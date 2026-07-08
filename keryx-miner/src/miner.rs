use std::collections::HashMap;
use std::num::Wrapping;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, sleep};
use std::time::{Duration, Instant};

use crate::{pow, watch, Error};
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use tokio::sync::mpsc::Sender;

use crate::pow::BlockSeed;
use keryx_miner::{PluginManager, WorkerSpec};

type MinerHandler = std::thread::JoinHandle<Result<(), Error>>;

// NOTA: aquí vivía el "freeze handler" (SIGUSR1 -> panic! / TerminateThread) que mataba a los
// workers que no cerraban en 1s. Eliminado: panicar desde un handler de señal sobre un frame de
// libcuda aborta el proceso entero (panic_cannot_unwind), y TerminateThread deja locks del driver
// colgados. El cierre ahora es espera acotada + detach (ver MinerManager::drop).

#[derive(Clone)]
enum WorkerCommand {
    Job(Box<pow::State>),
    Close,
}

#[allow(dead_code)]
pub struct MinerManager {
    handles: Vec<MinerHandler>,
    block_channel: watch::Sender<Option<WorkerCommand>>,
    send_channel: Sender<BlockSeed>,
    logger_stop: Arc<AtomicBool>,
    is_synced: bool,
    hashes_tried: Arc<AtomicU64>,
    hashes_by_worker: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
    current_state_id: AtomicUsize,
    opoi_challenge_active: Arc<AtomicBool>,
}

impl Drop for MinerManager {
    fn drop(&mut self) {
        info!("Closing miner");
        // Signal the detached hashrate logger to exit on its next wake (it polls this flag). We
        // don't join it — that would block shutdown up to LOG_RATE.
        self.logger_stop.store(true, Ordering::Release);
        match self.block_channel.send(Some(WorkerCommand::Close)) {
            Ok(_) => {}
            Err(_) => warn!("All workers are already dead"),
        }
        while !self.handles.is_empty() {
            let handle = self.handles.pop().expect("There should be at least one");
            // Espera ACOTADA y sin señales. El mecanismo anterior (SIGUSR1 -> panic! en el
            // handler tras 1s) abortaba el proceso entero: si la señal pilla al worker dentro
            // de libcuda (p. ej. subiendo el buffer inicial), el frame C no puede hacer unwind
            // -> panic_cannot_unwind -> abort. En GPUs con PCIe capado (CMP 170HX, x4 Gen1 y
            // 8 tarjetas serializadas) ese build inicial tarda decenas de segundos y el hilo
            // NO está colgado, solo lento. Si no termina en el plazo, se DETACHA: verá el
            // Close en el canal al acabar su operación CUDA en curso y saldrá solo.
            let deadline = Instant::now() + Duration::from_secs(30);
            while !handle.is_finished() && Instant::now() < deadline {
                sleep(Duration::from_millis(100));
            }
            if handle.is_finished() {
                match handle.join() {
                    Ok(res) => match res {
                        Ok(()) => {}
                        Err(e) => error!("Error when closing Worker: {}", e),
                    },
                    Err(_) => error!("Worker failed to close gracefully"),
                };
            } else {
                warn!(
                    "Worker still busy after 30s (slow CUDA op in flight?) — detaching; \
                     it exits on its own when the operation completes."
                );
            }
        }
    }
}

pub fn get_num_cpus(n_cpus: Option<u16>) -> u16 {
    n_cpus.unwrap_or_else(|| {
        num_cpus::get_physical().try_into().expect("Doesn't make sense to have more than 65,536 CPU cores")
    })
}

const LOG_RATE: Duration = Duration::from_secs(10);

impl MinerManager {
    pub fn new(send_channel: Sender<BlockSeed>, n_cpus: Option<u16>, manager: &PluginManager) -> Self {
        let hashes_tried = Arc::new(AtomicU64::new(0));
        let hashes_by_worker = Arc::new(Mutex::new(HashMap::<String, Arc<AtomicU64>>::new()));
        let opoi_challenge_active = Arc::new(AtomicBool::new(false));
        let (send, recv) = watch::channel(None);
        let mut handles =
            Self::launch_cpu_threads(send_channel.clone(), Arc::clone(&hashes_tried), recv.clone(), n_cpus)
                .collect::<Vec<MinerHandler>>();
        if manager.has_specs() {
            handles.append(&mut Self::launch_gpu_threads(
                send_channel.clone(),
                Arc::clone(&hashes_tried),
                recv,
                manager,
                hashes_by_worker.clone(),
            ));
        }
        let logger_stop = Arc::new(AtomicBool::new(false));
        let logger_stop_spawn = Arc::clone(&logger_stop);
        // Clone the counters the logger reads BEFORE the move-closure, so the originals stay
        // available for the struct fields below. The hashrate logger runs on a dedicated std::thread
        // (not a tokio task) so it never occupies one of the few async workers; it is detached and
        // exits on `logger_stop` (set in Drop) — no join (that would block shutdown up to LOG_RATE).
        let logger_hashes = Arc::clone(&hashes_tried);
        let logger_by_worker = hashes_by_worker.clone();
        let logger_challenge = Arc::clone(&opoi_challenge_active);
        thread::spawn(move || {
            Self::log_hashrate(logger_hashes, logger_by_worker, logger_challenge, logger_stop_spawn)
        });
        Self {
            handles,
            block_channel: send,
            send_channel,
            logger_stop,
            is_synced: true,
            hashes_tried,
            current_state_id: AtomicUsize::new(0),
            hashes_by_worker,
            opoi_challenge_active,
        }
    }

    fn launch_cpu_threads(
        send_channel: Sender<BlockSeed>,
        hashes_tried: Arc<AtomicU64>,
        work_channel: watch::Receiver<Option<WorkerCommand>>,
        n_cpus: Option<u16>,
    ) -> impl Iterator<Item = MinerHandler> {
        let n_cpus = get_num_cpus(n_cpus);
        info!("launching: {} cpu miners", n_cpus);
        (0..n_cpus)
            .map(move |_| Self::launch_cpu_miner(send_channel.clone(), work_channel.clone(), Arc::clone(&hashes_tried)))
    }

    fn launch_gpu_threads(
        send_channel: Sender<BlockSeed>,
        hashes_tried: Arc<AtomicU64>,
        work_channel: watch::Receiver<Option<WorkerCommand>>,
        manager: &PluginManager,
        hashes_by_worker: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
    ) -> Vec<MinerHandler> {
        let mut vec = Vec::<MinerHandler>::new();
        let specs = manager.build().unwrap();
        for spec in specs {
            let worker_hashes_tried = Arc::new(AtomicU64::new(0));
            hashes_by_worker.lock().unwrap().insert(spec.id(), worker_hashes_tried.clone());
            vec.push(Self::launch_gpu_miner(
                send_channel.clone(),
                work_channel.clone(),
                Arc::clone(&hashes_tried),
                spec,
                worker_hashes_tried,
            ));
        }
        vec
    }

    pub fn opoi_challenge_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.opoi_challenge_active)
    }

    pub async fn process_block(&mut self, block: Option<BlockSeed>) -> Result<(), Error> {
        let state = match block {
            Some(b) => {
                self.is_synced = true;
                let id = self.current_state_id.fetch_add(1, Ordering::SeqCst);
                Some(WorkerCommand::Job(Box::new(pow::State::new(id, b)?)))
            }
            None => {
                if !self.is_synced {
                    return Ok(());
                }
                self.is_synced = false;
                if self.opoi_challenge_active.load(Ordering::Relaxed) {
                    info!("OPoI challenge in progress — PoW template suspended, stand by");
                } else {
                    warn!("Keryxd is not synced, skipping current template");
                }
                None
            }
        };

        self.block_channel.send(state).map_err(|_e| "Failed sending block to threads")?;
        Ok(())
    }

    #[allow(unreachable_code)]
    fn launch_gpu_miner(
        send_channel: Sender<BlockSeed>,
        mut block_channel: watch::Receiver<Option<WorkerCommand>>,
        hashes_tried: Arc<AtomicU64>,
        spec: Box<dyn WorkerSpec>,
        worker_hashes_tried: Arc<AtomicU64>,
    ) -> MinerHandler {
        std::thread::spawn(move || {
            let mut box_ = spec.build();
            let gpu_work = box_.as_mut();
            (|| {
                info!("Spawned Thread for GPU {}", gpu_work.id());
                // CUDA device ordinal for this worker, parsed from the plugin id ("#0 (NVIDIA …)").
                // Each GPU thread mines PoM on its own candle device so all GPUs run in parallel.
                let pom_dev: u32 = gpu_work
                    .id()
                    .trim_start_matches('#')
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
                let mut nonces = vec![0u64; 1];

                let mut state = None;
                // PoM mining: nonce cursor + per-launch batch. The kernel grinds the whole batch
                // before returning, so BPS_max = hashrate / POM_BATCH. At 1<<22 this capped a
                // ~24 MH/s GPU at ~5.8 BPS. 1<<20 lifts the ceiling to ~23 BPS while staying well
                // above kernel-launch overhead (batch ≈ 43 ms at 24 MH/s).
                let mut pom_nonce: u64 = thread_rng().next_u64();
                const POM_BATCH: u64 = 1 << 20;

                loop {
                    nonces[0] = 0;
                    if state.is_none() {
                        state = match block_channel.wait_for_change() {
                            Ok(cmd) => match cmd {
                                Some(WorkerCommand::Job(s)) => Some(s),
                                Some(WorkerCommand::Close) => {return Ok(());}
                                None => None,
                            },
                            Err(e) => {
                                info!("{}: GPU thread crashed: {}", gpu_work.id(), e.to_string());
                                return Ok(());
                            }
                        };
                    }
                    // PoM possession mining (design A): when active, the walk runs on the GPU
                    // over the resident weights instead of kHeavyHash. On a winning nonce we build
                    // the proof (host) and submit; the legacy plugin path below is skipped.
                    if matches!(state.as_ref(), Some(s) if s.daa_score >= keryx_miner::pom::POM_ACTIVATION_DAA) {
                        let (pph, time, target_le, block_daa) = {
                            let s = state.as_ref().unwrap();
                            let mut pph = [0u8; 32];
                            pph.copy_from_slice(&s.pow_hash_header[0..32]);
                            let time = u64::from_le_bytes(s.pow_hash_header[32..40].try_into().unwrap());
                            (pph, time, s.target.to_le_bytes(), s.daa_score)
                        };
                        // An inference may have evicted the mining model (inference has priority).
                        // Rebuild the walk (reloads the model resident) before mining resumes.
                        if !keryx_miner::pom_gpu::is_installed(pom_dev) {
                            keryx_miner::pom_gpu::ensure_installed(pom_dev, block_daa);
                            // Still not resident (index/model still building, or an inference holds
                            // the GPU): mine() would no-op and return instantly. Counting a full
                            // POM_BATCH here would be a phantom — it inflates the reported hashrate
                            // by orders of magnitude. Pick up any new job, back off, and retry
                            // without counting until the GPU miner is actually grinding.
                            if !keryx_miner::pom_gpu::is_installed(pom_dev) {
                                if let Some(cmd) = block_channel.get_changed()? {
                                    state = match cmd {
                                        Some(WorkerCommand::Job(ns)) => Some(ns),
                                        Some(WorkerCommand::Close) => return Ok(()),
                                        None => state,
                                    };
                                }
                                std::thread::sleep(std::time::Duration::from_millis(200));
                                continue;
                            }
                        }
                        let h3 = block_daa >= keryx_miner::pom::POM_LEVEL_ACTIVATION_DAA;
                        let found = keryx_miner::pom_gpu::mine(pom_dev, &pph, time, &target_le, pom_nonce, POM_BATCH, h3);
                        pom_nonce = pom_nonce.wrapping_add(POM_BATCH);
                        hashes_tried.fetch_add(POM_BATCH, Ordering::AcqRel);
                        worker_hashes_tried.fetch_add(POM_BATCH, Ordering::AcqRel);
                        if let Some(nonce) = found {
                            info!("PoM: GPU candidate nonce {:#018x} ≤ target — re-walking on host to build proof", nonce);
                            let built = state.as_ref().and_then(|s| {
                                keryx_miner::pom::active_index(pom_dev).and_then(|idx| {
                                    // Tier is recomputed per block from its DAA (H2 reindexing).
                                    let tier = keryx_miner::pom_gpu::current_tier(pom_dev, s.daa_score)?;
                                    s.generate_block_if_pom(nonce, &idx, tier)
                                })
                            });
                            match built {
                                Some(block_seed) => {
                                    info!("PoM: proof built for nonce {:#018x} — submitting share", nonce);
                                    match send_channel.blocking_send(block_seed.clone()) {
                                        Ok(()) => {
                                            block_seed.report_block();
                                            if let BlockSeed::FullBlock(_) = block_seed {
                                                state = None;
                                            }
                                        }
                                        // Submit channel dead (pool dropped). Don't spin-flood: pause
                                        // this worker (state = None → block on the next job/Close).
                                        // listen() will have returned → the client reconnects and a
                                        // fresh MinerManager respawns workers on a live channel.
                                        Err(_) => {
                                            warn!("PoM submit channel closed — pausing GPU worker until reconnect");
                                            state = None;
                                        }
                                    }
                                }
                                None => {
                                    warn!("PoM: host re-walk REJECTED GPU candidate {:#018x} — GPU/host walk or target mismatch; NOT submitting", nonce);
                                }
                            }
                        } else if let Some(cmd) = block_channel.get_changed()? {
                            state = match cmd {
                                Some(WorkerCommand::Job(ns)) => Some(ns),
                                Some(WorkerCommand::Close) => return Ok(()),
                                None => state,
                            };
                        }
                        continue;
                    }

                    let state_ref = match &state {
                        Some(s) => {
                            s.load_to_gpu(gpu_work);
                            s
                        },
                        None => continue,
                    };
                    state_ref.pow_gpu(gpu_work);
                    if let Err(e) = gpu_work.sync() {
                        warn!("CUDA run ignored: {}", e);
                        continue
                    }

                    gpu_work.copy_output_to(&mut nonces)?;
                    // When PoM is active the GPU still runs kHeavyHash (3a is CPU-only); its
                    // solutions are NOT valid PoM blocks, so don't submit them. GPU PoM = 3b.
                    if nonces[0] != 0 && state_ref.daa_score < keryx_miner::pom::POM_ACTIVATION_DAA {
                        if let Some(block_seed) = state_ref.generate_block_if_pow(nonces[0]) {
                            match send_channel.blocking_send(block_seed.clone()) {
                                Ok(()) => {
                                    block_seed.report_block();
                                    if let BlockSeed::FullBlock(_) = block_seed {
                                        state = None;
                                    }
                                }
                                Err(_) => {
                                    warn!("Submit channel closed — pausing GPU worker until reconnect");
                                    state = None;
                                }
                            };
                            nonces[0] = 0;
                            hashes_tried.fetch_add(gpu_work.get_workload().try_into().unwrap(), Ordering::AcqRel);
                            worker_hashes_tried.fetch_add(gpu_work.get_workload().try_into().unwrap(), Ordering::AcqRel);
                            continue;
                        } else {
                            let hash = state_ref.calculate_pow(nonces[0]);
                            warn!("Something is wrong in GPU results! Got nonce {}, with hash real {:?}  (target: {}*2^196)", nonces[0], hash.0, state_ref.target.0[3]);
                            break;
                        }
                    }

                        /*
                        info!("Output should be: {:02X?}", state_ref.calculate_pow(nonces[0]).to_le_bytes());
                        info!("We got: {:02X?} (Nonces: {:02X?})", hashes[0], nonces[0].to_le_bytes());
                        assert!(state_ref.calculate_pow(nonces[0]).to_le_bytes() == hashes[0]);
                        */
                        /*
                        info!("Output should be: {}", state_ref.calculate_pow(nonces[nonces.len()-1]).0[3]);
                        info!("We got: {} (Nonces: {})", Uint256::from_le_bytes(hashes[nonces.len()-1]).0[3], nonces[nonces.len()-1]);
                        assert!(state_ref.calculate_pow(nonces[nonces.len()-1]).0[0] == Uint256::from_le_bytes(hashes[nonces.len()-1]).0[0]);
                         */
                        /*
                        if state_ref.calculate_pow(nonces[0]).0[0] != Uint256::from_le_bytes(hashes[0]).0[0] {
                            gpu_work.sync()?;
                            let mut nonce_vec = vec![nonces[0]; 1];
                            nonce_vec.append(&mut vec![0u64; gpu_work.workload-1]);
                            gpu_work.calculate_pow_hash(&state_ref.pow_hash_header, Some(&nonce_vec));
                            gpu_work.sync()?;
                            gpu_work.calculate_matrix_mul(&mut state_ref.matrix.clone().0.as_slice().as_dbuf().unwrap());
                            gpu_work.sync()?;
                            gpu_work.calculate_heavy_hash();
                            gpu_work.sync()?;
                            let mut hashes2  = vec![[0u8; 32]; out_size];
                            let mut nonces2= vec![0u64; out_size];
                            gpu_work.copy_output_to(&mut hashes2, &mut nonces2);
                            assert!(state_ref.calculate_pow(nonces[0]).to_le_bytes() == hashes2[0]);
                            assert!(nonces2[0] == nonces[0]);
                            assert!(hashes2 == hashes);
                            assert!(false);
                        }*/

                    hashes_tried.fetch_add(gpu_work.get_workload().try_into().unwrap(), Ordering::AcqRel);
                    worker_hashes_tried.fetch_add(gpu_work.get_workload().try_into().unwrap(), Ordering::AcqRel);

                    {
                        if let Some(new_cmd) = block_channel.get_changed()? {
                            state = match new_cmd {
                                Some(WorkerCommand::Job(s)) => Some(s),
                                Some(WorkerCommand::Close) => {return Ok(());}
                                None => None,
                            };
                        }
                    }
                }
                Ok(())
            })()
            .map_err(|e: Error| {
                error!("{}: GPU thread crashed: {}", gpu_work.id(), e.to_string());
                e
            })
        })
    }

    #[allow(unreachable_code)]
    fn launch_cpu_miner(
        send_channel: Sender<BlockSeed>,
        mut block_channel: watch::Receiver<Option<WorkerCommand>>,
        hashes_tried: Arc<AtomicU64>,
    ) -> MinerHandler {
        let mut nonce = Wrapping(thread_rng().next_u64());
        let mut mask = Wrapping(0);
        let mut fixed = Wrapping(0);
        std::thread::Builder::new()
            .name("cpu-miner".into())
            .stack_size(256 * 1024)
            .spawn(move || {
            (|| {
                let mut state = None;

                loop {
                    if state.is_none() {
                        state = match block_channel.wait_for_change() {
                            Ok(cmd) => match cmd {
                                Some(WorkerCommand::Job(s)) => Some(s),
                                Some(WorkerCommand::Close) => {
                                    return Ok(());
                                }
                                None => None,
                            },
                            Err(e) => {
                                info!("CPU thread crashed: {}", e.to_string());
                                return Ok(());
                            }
                        };
                        if let Some(s) = &state {
                            mask = Wrapping(s.nonce_mask);
                            fixed = Wrapping(s.nonce_fixed);
                        }
                    }
                    let state_ref = match state.as_mut() {
                        Some(s) => s,
                        None => continue,
                    };
                    nonce = (nonce & mask) | fixed;

                    // PoM possession path (CPU) once active; else legacy kHeavyHash.
                    let found = if state_ref.daa_score >= keryx_miner::pom::POM_ACTIVATION_DAA {
                        keryx_miner::pom::active_index(0).and_then(|idx| {
                            let tier = keryx_miner::pom_gpu::current_tier(0, state_ref.daa_score)?;
                            state_ref.generate_block_if_pom(nonce.0, &idx, tier)
                        })
                    } else {
                        state_ref.generate_block_if_pow(nonce.0)
                    };
                    if let Some(block_seed) = found {
                        match send_channel.blocking_send(block_seed.clone()) {
                            Ok(()) => {
                                block_seed.report_block();
                                if let BlockSeed::FullBlock(_) = block_seed {
                                    state = None;
                                }
                            }
                            Err(_) => {
                                warn!("Submit channel closed — pausing CPU worker until reconnect");
                                state = None;
                            }
                        };
                    }
                    nonce += Wrapping(1);
                    // TODO: Is this really necessary? can we just use Relaxed?
                    hashes_tried.fetch_add(1, Ordering::AcqRel);

                    if nonce.0 % 128 == 0 {
                        if let Some(new_cmd) = block_channel.get_changed()? {
                            state = match new_cmd {
                                Some(WorkerCommand::Job(s)) => Some(s),
                                Some(WorkerCommand::Close) => {
                                    return Ok(());
                                }
                                None => None,
                            };
                        }
                    }
                }
                Ok(())
            })()
            .map_err(|e: Error| {
                error!("CPU thread crashed: {}", e.to_string());
                e
            })
        }).expect("failed to spawn cpu-miner thread")
    }

    fn log_hashrate(
        hashes_tried: Arc<AtomicU64>,
        hashes_by_worker: Arc<Mutex<HashMap<String, Arc<AtomicU64>>>>,
        opoi_challenge_active: Arc<AtomicBool>,
        stop: Arc<AtomicBool>,
    ) {
        use std::io::{IsTerminal, Write};
        let start = Instant::now();
        let mut last_instant = Instant::now();
        // Lines the interactive panel drew last tick (0 = nothing yet) — the redraw moves the
        // cursor up this many lines and repaints in place, so the panel never scrolls.
        let mut drawn: usize = 0;
        while !stop.load(Ordering::Acquire) {
            thread::sleep(LOG_RATE);
            if stop.load(Ordering::Acquire) {
                break;
            }
            let duration = last_instant.elapsed().as_secs_f64();
            last_instant = Instant::now();
            // PoM model (re)load also intentionally pauses PoW — treat it like an inference pause.
            let challenge_active =
                opoi_challenge_active.load(Ordering::Relaxed) || keryx_miner::pom_gpu::is_loading();
            let hashes = hashes_tried.swap(0, Ordering::AcqRel);
            // Drenar los contadores por worker y capturar el ratio por-GPU (id = "#N (nombre)").
            let mut per_gpu: Vec<(u32, f64)> = Vec::new();
            for (id, counter) in &*hashes_by_worker.lock().unwrap() {
                let h = counter.swap(0, Ordering::AcqRel);
                if let Some(n) = id.trim_start_matches('#').split_whitespace().next().and_then(|s| s.parse::<u32>().ok()) {
                    per_gpu.push((n, h as f64 / duration));
                }
            }
            per_gpu.sort_by_key(|(n, _)| *n);
            // HiveOS / log mode (stdout not a TTY): emit newline-terminated, greppable stats lines
            // that h-stats.sh parses ("Device #N: X unit" per GPU + "Current hashrate is X unit").
            // println! DIRECTO, no info!: el nivel por defecto del logger es Warn (TUI limpio),
            // así que con info! estas líneas nunca llegaban al log de HiveOS → khs=0/stats=null
            // → dashboard vacío salvo con -d. El timestamp UTC replica el formato env_logger
            // porque h-stats.sh extrae el ISO-8601 de la línea para el chequeo de frescura.
            if !std::io::stdout().is_terminal() {
                let ts = Self::utc_stamp();
                for (n, rate) in &per_gpu {
                    let (r, u) = Self::hash_suffix(*rate);
                    println!("[{ts} INFO  keryx_miner::miner] Device #{}: {:.2} {}", n, r, u);
                }
                let (tr, tu) = if hashes == 0 { (0.0, "hash/s") } else { Self::hash_suffix(hashes as f64 / duration) };
                println!("[{ts} INFO  keryx_miner::miner] Current hashrate is {:.2} {}", tr, tu);
                let (acc, rej) = crate::client::stratum::share_counts();
                println!("[{ts} INFO  keryx_miner::miner] Shares total: {} accepted, {} rejected", acc, rej);
                let _ = std::io::stdout().flush();
                continue;
            }

            // ── Panel interactivo: bloque por-GPU repintado en sitio (sin scroll) ──────
            const CY: &str = "\x1b[36m"; // cian
            const BD: &str = "\x1b[1;36m"; // cian negrita
            const GR: &str = "\x1b[1;32m"; // verde negrita
            const YL: &str = "\x1b[1;33m"; // amarillo negrita
            const RD: &str = "\x1b[1;31m"; // rojo negrita
            const DM: &str = "\x1b[2m"; // atenuado
            const RS: &str = "\x1b[0m"; // reset

            let stats = Self::gpu_stats();
            let (acc, rej) = crate::client::stratum::share_counts();
            let up = start.elapsed().as_secs();
            let uptime = if up >= 3600 {
                format!("{}h {:02}m", up / 3600, (up / 60) % 60)
            } else {
                format!("{}m {:02}s", up / 60, up % 60)
            };
            let sep = format!("  {DM}{}{RS}", "─".repeat(66));

            let mut lines: Vec<String> = Vec::with_capacity(stats.len().max(per_gpu.len()) + 4);
            lines.push(format!(
                "  {BD}ddminer{RS} {DM}v{}{RS} {DM}·{RS} {CY}Keryx (KRX){RS} {DM}· PoM + OPoI · 0% fee · up {uptime}{RS}",
                env!("CARGO_PKG_VERSION")
            ));
            lines.push(sep.clone());
            // Filas por GPU: nvidia-smi manda (nombre/T/fan/W); el ratio viene del contador del
            // worker. Sin nvidia-smi (contenedores minimalistas) caemos a filas solo-hashrate.
            if stats.is_empty() {
                for (n, rate) in &per_gpu {
                    let (r, u) = Self::hash_suffix(*rate);
                    lines.push(format!("  {CY}GPU{n}{RS}   {GR}{r:>7.2} {u}{RS}"));
                }
            } else {
                for g in &stats {
                    let rate = per_gpu.iter().find(|(n, _)| *n == g.0).map(|(_, r)| *r).unwrap_or(0.0);
                    let hr_cell = if rate > 0.0 {
                        let (r, u) = Self::hash_suffix(rate);
                        format!("{GR}{r:>7.2} {u:<7}{RS}")
                    } else if challenge_active {
                        format!("{YL}{:<15}{RS}", "OPoI/PoM load")
                    } else {
                        format!("{DM}{:<15}{RS}", "—")
                    };
                    // Temperatura con semáforo: <70 atenuado, 70–84 amarillo, 85+ rojo.
                    let temp_cell = match g.2 {
                        Some(t) if t >= 85 => format!("{RD}{t:>3}°C{RS}"),
                        Some(t) if t >= 70 => format!("{YL}{t:>3}°C{RS}"),
                        Some(t) => format!("{DM}{t:>3}°C{RS}"),
                        None => format!("{DM}  --°C{RS}"),
                    };
                    let fan_cell = match g.3 {
                        Some(f) => format!("{DM}fan {f:>3}%{RS}"),
                        None => format!("{DM}fan  --{RS}"),
                    };
                    let pow_cell = match g.4 {
                        Some(w) => format!("{DM}{w:>3} W{RS}"),
                        None => format!("{DM} -- W{RS}"),
                    };
                    lines.push(format!(
                        "  {CY}GPU{:<2}{RS} {:<12} {}  {}  {}  {}",
                        g.0, g.1, hr_cell, temp_cell, fan_cell, pow_cell
                    ));
                }
            }
            lines.push(sep);
            let total_cell = if hashes == 0 {
                if challenge_active {
                    format!("{YL}OPoI challenge — PoW en pausa{RS}")
                } else {
                    format!("{DM}warming up…{RS}")
                }
            } else {
                let (r, u) = Self::hash_suffix(hashes as f64 / duration);
                format!("{GR}{r:.2} {u}{RS}")
            };
            let eff = if acc + rej > 0 {
                format!(" {DM}({:.1}%){RS}", 100.0 * acc as f64 / (acc + rej) as f64)
            } else {
                String::new()
            };
            let rej_cell =
                if rej > 0 { format!("{RD}✘ {rej}{RS}") } else { format!("{DM}✘ {rej}{RS}") };
            lines.push(format!(
                "  {DM}TOTAL{RS}  {total_cell}   shares {GR}✔ {acc}{RS} {rej_cell}{eff}"
            ));

            // Repintado: sube `drawn` líneas y limpia hasta el final de pantalla; así el panel
            // vive fijo bajo el banner aunque cambie de altura (p. ej. nvidia-smi intermitente).
            if drawn > 0 {
                print!("\x1b[{drawn}A");
            }
            print!("\x1b[0J");
            for l in &lines {
                println!("{l}");
            }
            let _ = std::io::stdout().flush();
            drawn = lines.len();
        }
    }

    /// Telemetría por GPU vía nvidia-smi: (índice, nombre corto, temp °C, fan %, potencia W).
    /// Los campos que nvidia-smi reporta como "[N/A]" (p. ej. fan en pasivas) llegan como None.
    fn gpu_stats() -> Vec<(u32, String, Option<i32>, Option<u32>, Option<u32>)> {
        match std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=index,name,temperature.gpu,fan.speed,power.draw",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter_map(|l| {
                    let f: Vec<&str> = l.split(',').map(|s| s.trim()).collect();
                    if f.len() >= 5 {
                        // nombre corto: "NVIDIA GeForce RTX 4090" -> "RTX 4090"
                        let name = f[1].replace("NVIDIA GeForce ", "").replace("NVIDIA ", "");
                        Some((
                            f[0].parse::<u32>().ok()?,
                            name,
                            f[2].parse::<i32>().ok(),
                            f[3].parse::<u32>().ok(),
                            f[4].parse::<f64>().ok().map(|w| w.round() as u32),
                        ))
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        }
    }

    #[allow(dead_code)] // sustituido por la línea de estado compacta (gpu_status + println)
    fn log_single_hashrate(
        counter: &Arc<AtomicU64>,
        prefix: String,
        warn_message: &str,
        duration: f64,
        keep_prefix: bool,
        challenge_active: bool,
    ) {
        let hashes = counter.swap(0, Ordering::AcqRel);
        let rate = (hashes as f64) / duration;
        if hashes == 0 {
            if challenge_active {
                if keep_prefix {
                    info!("{} OPoI challenge in progress — stand by", prefix);
                } else {
                    info!("OPoI challenge in progress — PoW paused, stand by");
                }
            } else {
                match keep_prefix {
                    true => warn!("{}{}", prefix, warn_message),
                    false => warn!("{}", warn_message),
                };
            }
        } else {
            let (rate, suffix) = Self::hash_suffix(rate);
            info!("{} {:.2} {}", prefix, rate, suffix);
        }
    }

    #[inline]
    /// Timestamp UTC "2026-07-05T18:00:00Z" — mismo formato que env_logger, para que el
    /// extractor ISO-8601 de h-stats.sh (frescura del dashboard) funcione sobre nuestras líneas.
    fn utc_stamp() -> String {
        use time::format_description::well_known::Rfc3339;
        time::OffsetDateTime::now_utc()
            .replace_nanosecond(0)
            .unwrap_or_else(|_| time::OffsetDateTime::now_utc())
            .format(&Rfc3339)
            .unwrap_or_default()
    }

    fn hash_suffix(n: f64) -> (f64, &'static str) {
        match n {
            n if n < 1_000.0 => (n, "hash/s"),
            n if n < 1_000_000.0 => (n / 1_000.0, "Khash/s"),
            n if n < 1_000_000_000.0 => (n / 1_000_000.0, "Mhash/s"),
            n if n < 1_000_000_000_000.0 => (n / 1_000_000_000.0, "Ghash/s"),
            n if n < 1_000_000_000_000_000.0 => (n / 1_000_000_000_000.0, "Thash/s"),
            _ => (n, "hash/s"),
        }
    }
}

#[cfg(all(test, feature = "bench"))]
mod benches {
    extern crate test;

    use self::test::{black_box, Bencher};
    use crate::pow::State;
    use crate::proto::{RpcBlock, RpcBlockHeader};
    use rand::{thread_rng, RngCore};

    #[bench]
    pub fn bench_mining(bh: &mut Bencher) {
        let mut state = State::new(
            0,
            RpcBlock {
                header: Some(RpcBlockHeader {
                    version: 1,
                    parents: vec![],
                    hash_merkle_root: "23618af45051560529440541e7dc56be27676d278b1e00324b048d410a19d764".to_string(),
                    accepted_id_merkle_root: "947d1a10378d6478b6957a0ed71866812dee33684968031b1cace4908c149d94"
                        .to_string(),
                    utxo_commitment: "ec5e8fc0bc0c637004cee262cef12e7cf6d9cd7772513dbd466176a07ab7c4f4".to_string(),
                    timestamp: 654654353,
                    bits: 0x1e7fffff,
                    nonce: 0,
                    daa_score: 654456,
                    blue_work: "d8e28a03234786".to_string(),
                    pruning_point: "be4c415d378f9113fabd3c09fcc84ddb6a00f900c87cb6a1186993ddc3014e2d".to_string(),
                    blue_score: 1164419,
                }),
                transactions: vec![],
                verbose_data: None,
            },
        )
        .unwrap();
        nonce = thread_rng().next_u64();
        bh.iter(|| {
            for _ in 0..100 {
                black_box(state.check_pow(nonce));
                nonce += 1;
            }
        });
    }
}
