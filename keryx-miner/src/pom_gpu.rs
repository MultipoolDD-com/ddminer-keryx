//! Proof-of-Model GPU mining — runs the `pom_mine` kernel in candle's CUDA context over the
//! resident weight blob to find a winning nonce. Foundation for the live mining loop (§6/3b).
//!
//! Loads the mining tier's GGUF raw (so we get per-tensor device pointers for the gather, like
//! `pom-q4-probe`) and builds the chunk-prefix gather index on the GPU. NOTE: this is a second
//! VRAM copy of the model (the inference engine holds its own). Fine for small tiers on the
//! testnet; the big tiers will share buffers later.
//!
//! The kernel's seed/pow folds are byte-identical to `pom::pom_block_seed`/`pom::pom_pow_value`,
//! so a nonce found here builds a `PomProof` (host) the node accepts.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use log::info;

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{CudaDevice, Device};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine.ptx"));
const CHUNK_BYTES: usize = 32;

fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

/// Total VRAM (MB) of every CUDA device, in **CUDA device order** — the same ordering
/// `Device::new_cuda(id)` uses — so an entry `(id, mb)` is the VRAM of the device the miner would
/// mine/serve on for that `id`. Sourced from the CUDA driver, NOT nvidia-smi: nvidia-smi orders by
/// PCI position, which disagrees with CUDA's default `FASTEST_FIRST` ordering on a mixed rig, so a
/// line-order mapping would read the wrong card's VRAM. Returns an empty vec when no CUDA driver is
/// present (CPU-only / AMD hosts). Never panics — a driver-load failure inside cudarc is caught and
/// treated as "no devices".
pub fn query_all_gpus_vram() -> Vec<(usize, u64)> {
    use candle_core::cuda_backend::cudarc::driver::result;
    std::panic::catch_unwind(|| {
        if result::init().is_err() {
            return Vec::new();
        }
        let count = result::device::get_count().unwrap_or(0);
        let mut out = Vec::with_capacity(count.max(0) as usize);
        for ordinal in 0..count {
            let Ok(dev) = result::device::get(ordinal) else {
                continue;
            };
            // SAFETY: `dev` is a valid device handle just returned by `device::get(ordinal)`.
            if let Ok(bytes) = unsafe { result::device::total_mem(dev) } {
                out.push((ordinal as usize, (bytes / (1024 * 1024)) as u64));
            }
        }
        out
    })
    .unwrap_or_default()
}

pub struct PomGpuMiner {
    cuda: CudaDevice,
    stream: Arc<CudaStream>,
    bases_dev: CudaSlice<u64>,
    prefix_dev: CudaSlice<u64>,
    t_count: u32,
    n_total_chunks: u64,
    _tensors: Vec<QTensor>, // raw-loaded tensors kept alive so the gather pointers stay valid
    _shared: Vec<Arc<QTensor>>, // shared-with-inference tensors kept alive (zero-dup, Option C)
}

/// CUDA walk-kernel block size, read once from POM_BLOCK (default 256), clamped to a warp-size
/// multiple in [32, 1024]. Occupancy benchmark knob only — does not affect results/proof validity.
fn pom_block_size() -> u32 {
    static CACHE: OnceLock<u32> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let b = std::env::var("POM_BLOCK").ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(256);
        let b = b.clamp(32, 1024) / 32 * 32;
        info!("PoM walk kernel block size = {} (POM_BLOCK)", b);
        b
    })
}

impl PomGpuMiner {
    /// Load the mining model's GGUF into candle on CUDA device `dev`, build the gather index, load
    /// the kernel. A standalone copy (used for device 0 fallback and for every non-zero device).
    pub fn load(gguf_path: &str, dev: u32) -> candle_core::Result<Self> {
        let device = Device::new_cuda(dev as usize)?;
        let cuda = match &device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: not a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical order — matches pom-rt-builder / the node R_T

        // Total de bytes de la sección de datos (para el % de subida a VRAM del panel).
        let data_total = file
            .metadata()
            .map(|m| m.len().saturating_sub(content.tensor_data_offset))
            .unwrap_or(0)
            .max(1);
        let mut data_done: u64 = 0;
        set_load_phase(dev, LoadPhase::Vram(0));

        let mut tensors: Vec<QTensor> = Vec::with_capacity(names.len());
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        for name in &names {
            let qt = content.tensor(&mut file, name, &device)?;
            data_done += qt.storage_size_in_bytes() as u64;
            set_load_phase(dev, LoadPhase::Vram((data_done * 100 / data_total).min(100) as u8));
            let chunks = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u64;
            if chunks == 0 {
                tensors.push(qt);
                continue;
            }
            bases.push(qt.device_ptr()? as usize as u64);
            prefix.push(prefix.last().unwrap() + chunks);
            tensors.push(qt);
        }
        clear_load_phase(dev);
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: model produced 0 chunks".into()));
        }

        let bases_dev = stream.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = stream.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        // Warm the module cache so mine() never compiles on the hot path.
        let _ = cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?;

        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: tensors, _shared: Vec::new() })
    }

    /// Zero-dup load (Option C): build the gather over the SAME canonical name-sorted layout as
    /// `R_T`, but for each tensor reuse the inference engine's resident VRAM buffer when it holds
    /// it quantized (`shared`, the big matrices) instead of loading a second copy. Only the
    /// dequantized-in-inference tensors (token_embd, norms) are read raw here — small. `device`
    /// MUST be the same candle device the `shared` tensors live on (pointers are context-bound).
    pub fn load_shared(
        gguf_path: &str,
        device: &Device,
        shared: &std::collections::HashMap<String, Arc<QTensor>>,
    ) -> candle_core::Result<Self> {
        let cuda = match device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: shared load requires a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical order — must match pom-rt-builder / the node R_T

        let mut raw: Vec<QTensor> = Vec::new();
        let mut kept_shared: Vec<Arc<QTensor>> = Vec::new();
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u64> = vec![0];
        let mut shared_hits = 0usize;
        for name in &names {
            let (ptr, chunks) = if let Some(qt) = shared.get(name) {
                // Matrix already resident for inference → reuse its buffer (zero dup).
                let c = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u64;
                let p = qt.device_ptr()? as usize as u64;
                kept_shared.push(qt.clone());
                shared_hits += 1;
                (p, c)
            } else {
                // Dequantized-in-inference (token_embd, norms): read the raw quantized bytes.
                let qt = content.tensor(&mut file, name, device)?;
                let c = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u64;
                if c == 0 {
                    raw.push(qt);
                    continue;
                }
                let p = qt.device_ptr()? as usize as u64;
                raw.push(qt);
                (p, c)
            };
            if chunks == 0 {
                continue;
            }
            bases.push(ptr);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap();
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: shared load produced 0 chunks".into()));
        }
        info!("PoM zero-dup gather: {} shared tensors, {} raw-loaded, N={} chunks", shared_hits, raw.len(), n_total_chunks);

        let bases_dev = stream.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = stream.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        let _ = cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?;

        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: raw, _shared: kept_shared })
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_total_chunks
    }

    /// Search nonces in `[start, start + batch)`. Returns the lowest nonce whose `pom_pow_value`
    /// is `<= target_le`, or None. `target_le` is the header's compact target as 32 LE bytes.
    /// `h3` salts the pph words host-side (POM_H3_PPH_SALT) — the kernel itself is era-agnostic,
    /// it folds whatever words it receives, so no PTX change at the H3 gate.
    pub fn mine(&self, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64, h3: bool) -> candle_core::Result<Option<u64>> {
        // CUDA block size for the walk kernel. The walk is memory-latency-bound (256 dependent
        // random VRAM reads/nonce), so occupancy — how many warps hide that latency — is the only
        // launch-side lever. Tunable via POM_BLOCK env (default 256); does NOT change the result
        // (proof stays valid), only how the same work is scheduled.
        self.mine_with_block(pom_block_size(), pre_pow_hash, timestamp, target_le, start, batch, h3)
    }

    /// Same as [`mine`] but with an explicit CUDA block size — used by the `--bench-pom` harness to
    /// sweep occupancy without the cached env value. Result is identical regardless of block size.
    pub fn mine_with_block(&self, block: u32, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64, h3: bool) -> candle_core::Result<Option<u64>> {
        let p = crate::pom::pph_words_for_era(pre_pow_hash, h3);
        let t = words4(target_le);
        let k = crate::pom::POM_WALK_STEPS;
        let winner = self.stream.clone_htod(&[u64::MAX]).map_err(candle_core::Error::wrap)?;
        let grid = ((batch + (block as u64) - 1) / (block as u64)) as u32;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (block, 1, 1), shared_mem_bytes: 0 };

        let func = self.cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?; // cached
        let mut b = func.builder();
        b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&self.n_total_chunks).arg(&k)
            .arg(&p[0]).arg(&p[1]).arg(&p[2]).arg(&p[3]).arg(&timestamp)
            .arg(&t[0]).arg(&t[1]).arg(&t[2]).arg(&t[3])
            .arg(&start).arg(&batch).arg(&winner);
        unsafe { b.launch(cfg).map_err(candle_core::Error::wrap)?; }
        self.stream.synchronize().map_err(candle_core::Error::wrap)?;

        let w = self.stream.clone_dtoh(&winner).map_err(candle_core::Error::wrap)?[0];
        Ok(if w == u64::MAX { None } else { Some(w) })
    }
}

/// Pool-independent benchmark of the PoM walk kernel: load `gguf` standalone on device `dev`, then
/// time `mine_with_block` over a fixed nonce batch for each block size. Prints MH/s per size. No
/// pool, no inference, no possession index — only the in-VRAM gather, so it isolates the kernel.
pub fn bench(gguf: &str, dev: u32) -> candle_core::Result<()> {
    use std::time::Instant;
    info!("PoM bench: loading '{}' on device {}…", gguf, dev);
    let m = PomGpuMiner::load(gguf, dev)?;
    info!("PoM bench: model resident — N={} chunks. Sweeping block sizes…", m.n_chunks());
    let pph = [0x11u8; 32];
    let target = [0u8; 32]; // imposible de cumplir → escanea el batch entero sin ganar
    let batch: u64 = 1 << 20;
    for &block in &[64u32, 128, 256, 384, 512, 768, 1024] {
        // Warm-up launch (módulo/caché) y luego promedio de varias iteraciones.
        let _ = m.mine_with_block(block, &pph, 0, &target, 0, batch, true)?;
        let iters = 8u64;
        let t0 = Instant::now();
        for i in 0..iters {
            let _ = m.mine_with_block(block, &pph, 0, &target, i * batch, batch, true)?;
        }
        let secs = t0.elapsed().as_secs_f64();
        let mhs = (iters * batch) as f64 / secs / 1.0e6;
        println!("  POM_BLOCK={block:>4}  →  {mhs:7.2} MH/s   ({:.2}s / {} nonces)", secs, iters * batch);
    }
    Ok(())
}

// Per-device GPU miner instances — one PoM miner per CUDA device so every GPU mines in parallel.
// Each device's miner sits behind its OWN mutex (inside the registry) so mine() on one device does
// not block mine() on another — only the brief registry lookup is shared. A miner can be
// uninstalled to free VRAM when inference (device 0 only) needs the GPU, then reinstalled.
static MINERS: Mutex<BTreeMap<u32, Arc<Mutex<Option<PomGpuMiner>>>>> = Mutex::new(BTreeMap::new());

/// The per-device slot (created on first use). The registry lock is held only for the lookup.
fn slot(dev: u32) -> Arc<Mutex<Option<PomGpuMiner>>> {
    MINERS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(dev)
        .or_insert_with(|| Arc::new(Mutex::new(None)))
        .clone()
}

/// Install device `dev`'s GPU miner (after loading/sharing the mining model's resident weights).
pub fn install(dev: u32, m: PomGpuMiner) {
    *slot(dev).lock().unwrap_or_else(|e| e.into_inner()) = Some(m);
}

/// Drop device `dev`'s GPU miner, releasing its hold on the mining model's VRAM (shared Arcs +
/// gather) so the inference engine can load another model. Inference is device-0 only.
pub fn uninstall(dev: u32) {
    *slot(dev).lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// Whether device `dev`'s GPU miner is currently installed.
pub fn is_installed(dev: u32) -> bool {
    slot(dev).lock().map(|g| g.is_some()).unwrap_or(false)
}

/// True while the GPU miner is being (re)built — a heavy one-time model load that blocks the
/// mining worker. The PoW stall watchdog treats this like an inference pause, not a crash.
static LOADING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Fase de la carga PoM de un device, para el panel de estado.
#[derive(Clone, Copy, PartialEq)]
pub enum LoadPhase {
    /// Construyendo el índice de posesión en host (compartido por modelo; el % vive en
    /// `pom::index_build_pct()` porque lo actualiza el builder, no este device).
    Index,
    /// Subiendo los pesos del modelo a la VRAM de ESTE device (0–100).
    Vram(u8),
}

/// Progreso de carga por device — el panel TUI lo lee para pintar "índice N% / modelo N%"
/// en las GPUs que aún no minan, en vez de un guion. Se limpia al terminar (ok o error).
static LOAD_PROGRESS: Mutex<BTreeMap<u32, LoadPhase>> = Mutex::new(BTreeMap::new());

fn set_load_phase(dev: u32, phase: LoadPhase) {
    LOAD_PROGRESS.lock().unwrap_or_else(|e| e.into_inner()).insert(dev, phase);
}

fn clear_load_phase(dev: u32) {
    LOAD_PROGRESS.lock().unwrap_or_else(|e| e.into_inner()).remove(&dev);
}

/// Fase de carga actual del device (None = no está cargando).
pub fn load_phase(dev: u32) -> Option<LoadPhase> {
    LOAD_PROGRESS.lock().unwrap_or_else(|e| e.into_inner()).get(&dev).copied()
}

/// Whether a PoM model load/rebuild is in progress (worker intentionally paused, not stalled).
pub fn is_loading() -> bool {
    LOADING.load(std::sync::atomic::Ordering::Relaxed)
}

/// Convenience: search a nonce batch via device `dev`'s installed miner. None if not installed or
/// no winner. Holds only this device's slot lock (not a global one) so devices mine concurrently.
pub fn mine(dev: u32, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64, h3: bool) -> Option<u64> {
    let s = slot(dev);
    let g = s.lock().ok()?;
    g.as_ref()?.mine(pre_pow_hash, timestamp, target_le, start, batch, h3).ok().flatten()
}

/// Mining-tier identity for rebuilds: (model_id, gguf_path). Set once at startup.
/// Per-device mining model (model_id + gguf path). A mixed rig assigns each GPU the highest PoM
/// tier whose model fits its VRAM, so the mining model is keyed by CUDA device ordinal.
static MINING_TIERS: Mutex<BTreeMap<u32, ([u8; 32], String)>> = Mutex::new(BTreeMap::new());

/// Record device `dev`'s mining tier so it can be rebuilt after an inference swapped the model away.
pub fn set_mining_tier(dev: u32, model_id: [u8; 32], gguf_path: String) {
    MINING_TIERS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(dev, (model_id, gguf_path));
}

/// Device `dev`'s mining model (model_id, gguf path). Falls back to the first assigned device's
/// model when this device has no explicit assignment (uniform rig: one tier set, all GPUs share it).
fn mining_tier(dev: u32) -> Option<([u8; 32], String)> {
    let g = MINING_TIERS.lock().ok()?;
    g.get(&dev).or_else(|| g.values().next()).cloned()
}

/// PoM tier index of device `dev`'s mining model at block `daa`. Recomputed PER BLOCK (not frozen
/// at index-build time) so the tier reindexing at the very-light hardfork (H2) is applied from the
/// crossing block onward — a CONSENSUS requirement (a frozen tier would diverge at the boundary).
pub fn current_tier(dev: u32, daa: u64) -> Option<u8> {
    let (model_id, _) = mining_tier(dev)?;
    crate::models::pom_tier_index(&model_id, daa)
}

/// The CUDA device that mines `model_id` (from the per-GPU tier assignment), if any. Inference for a
/// model is routed to the device that already holds it, so only that GPU pauses mining and the walk
/// can share the resident weights (zero-dup). Returns the lowest matching `dev` when several GPUs
/// mine the same tier; `None` when no GPU is assigned this model.
pub fn device_for_model(model_id: &[u8; 32]) -> Option<u32> {
    let g = MINING_TIERS.lock().ok()?;
    g.iter().filter(|(_, (id, _))| id == model_id).map(|(dev, _)| *dev).min()
}

/// Models that OOM'd when loading on a given GPU: `(dev, model_id)`. Once banlisted, that GPU never
/// retries that model (avoids a hot-spin reloading a model that doesn't fit); the OOM handler
/// downgrades the GPU to a smaller downloaded tier instead.
static OOM_BANLIST: Mutex<std::collections::BTreeSet<(u32, [u8; 32])>> =
    Mutex::new(std::collections::BTreeSet::new());

fn is_oom_banlisted(dev: u32, model_id: &[u8; 32]) -> bool {
    OOM_BANLIST.lock().unwrap_or_else(|e| e.into_inner()).contains(&(dev, *model_id))
}

fn oom_banlist_add(dev: u32, model_id: [u8; 32]) {
    OOM_BANLIST.lock().unwrap_or_else(|e| e.into_inner()).insert((dev, model_id));
}

/// After a GPU fails to load its assigned tier (OOM), reassign it to the largest **already-downloaded**
/// PoM model strictly smaller than the failed one that hasn't itself been banlisted on this GPU — so a
/// card whose VRAM estimate was optimistic (driver overhead + KV cache + fragmentation) mines a
/// smaller tier instead of idling. Returns true if a downgrade was applied. No extra prefetch is
/// needed: the candidate set is the served union (a mixed rig already downloaded the smaller tiers).
fn downgrade_after_oom(dev: u32, failed_model: &[u8; 32], daa: u64) -> bool {
    let Some(failed_tier) = crate::models::pom_tier_index(failed_model, daa) else {
        return false;
    };
    let pick = crate::slm::served_pom_specs()
        .into_iter()
        .filter_map(|s| crate::models::pom_tier_index(&s.model_id, daa).map(|t| (t, s)))
        .filter(|(t, s)| *t < failed_tier && !is_oom_banlisted(dev, &s.model_id))
        .max_by_key(|(t, _)| *t);
    match pick {
        Some((tier, spec)) => {
            let gguf = crate::slm::gguf_path_for(spec).to_string_lossy().into_owned();
            info!("PoM[dev {}]: OOM on tier {} — downgrading to tier {} ({}).", dev, failed_tier, tier, spec.name);
            set_mining_tier(dev, spec.model_id, gguf);
            true
        }
        None => {
            log::warn!("PoM[dev {}]: OOM and no smaller downloaded tier available — this GPU will not mine PoM (lower the tier flag or add VRAM).", dev);
            false
        }
    }
}

/// OOM al cargar `model_id` para INFERENCIA en `dev` (ruta SlmEngine, no PomGpuMiner). Si ese
/// modelo es además el tier de minado del device, aplica el mismo tratamiento que un OOM del
/// walk: banlist (dev, model) + downgrade al tier menor descargado — sin esto, el dev 0 de un
/// rig justo reintentaba la carga del mismo modelo para siempre (ensure_loaded → false → retry).
pub fn note_inference_oom(dev: u32, model_id: &[u8; 32]) {
    // Tier ranking evaluado en el gate H2: mainnet está permanentemente pasada H2 y el caller
    // (slm) no tiene el daa del bloque a mano; los índices de tier no cambian post-H2.
    let daa = crate::models::VERY_LIGHT_ACTIVATION_DAA;
    let is_mining_model = MINING_TIERS
        .lock()
        .map(|g| g.get(&dev).map_or(false, |(id, _)| id == model_id))
        .unwrap_or(false);
    if is_mining_model && !is_oom_banlisted(dev, model_id) {
        log::error!(
            "PoM[dev {}]: OOM cargando el modelo de minado para inferencia — banlist + downgrade.",
            dev
        );
        oom_banlist_add(dev, *model_id);
        downgrade_after_oom(dev, model_id, daa);
        // El índice/miner del modelo viejo quedan invalidados en el próximo ensure_installed
        // (invalidate_index_unless) — aquí solo liberamos el walk residente si lo hubiera.
        uninstall(dev);
    }
}

/// The set of distinct mining model ids assigned across all devices (for capability declaration).
pub fn assigned_model_ids() -> Vec<[u8; 32]> {
    let g = MINING_TIERS.lock().unwrap_or_else(|e| e.into_inner());
    let mut ids: Vec<[u8; 32]> = g.values().map(|(id, _)| *id).collect();
    ids.sort();
    ids.dedup();
    ids
}

/// Ensure the GPU miner is installed; if an inference evicted the mining model, reload it
/// (resident again) and rebuild the zero-dup gather. Heavy (model reload) but only when needed —
/// inference has priority, so mining reloads its model when it next gets the GPU. Returns true if
/// the miner is ready to mine.
/// Serializes the (global, one-shot) index build + miner install across GPU worker threads.
/// Without it, every GPU worker races into `WeightIndex::build_from_gguf` on the same gguf at
/// once: one wins and the rest short-read with "failed to fill whole buffer". On a multi-GPU rig
/// (e.g. 6× 3070) that means most GPUs fail to install the PoM miner. The first thread builds and
/// installs; the others block here and then find it ready via the re-checked `is_installed()`.
static INSTALL_LOCK: Mutex<()> = Mutex::new(());

/// Devices permanently disabled for PoM this run (e.g. Turing/sm_75: the PoM/inference PTX has an
/// sm_80 floor, so loading dies with CUDA_ERROR_INVALID_PTX — a hardware limit; retrying is spam).
static POM_UNSUPPORTED: Mutex<std::collections::BTreeSet<u32>> = Mutex::new(std::collections::BTreeSet::new());

pub fn ensure_installed(dev: u32, daa: u64) -> bool {
    if is_installed(dev) {
        return true;
    }
    if POM_UNSUPPORTED.lock().unwrap_or_else(|e| e.into_inner()).contains(&dev) {
        return false; // already diagnosed + logged once — stay quiet, let capable GPUs mine
    }
    // Recover from a poisoned lock (a prior worker panicked mid-build) rather than panicking the
    // whole miner — the worst case is one more rebuild attempt. The lock is global: it serializes
    // the one-shot host index build and the per-device model loads (a cheap startup cost) and
    // avoids candle multi-device init races.
    let _install = INSTALL_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    // Re-check under the lock: another worker may have finished this device's install while we waited.
    if is_installed(dev) {
        return true;
    }
    // Flag the heavy load so the stall watchdog stays benign while the worker is blocked here.
    LOADING.store(true, std::sync::atomic::Ordering::Relaxed);
    let ok = ensure_installed_inner(dev, daa);
    LOADING.store(false, std::sync::atomic::Ordering::Relaxed);
    clear_load_phase(dev); // también en error: que el panel no se quede clavado en un %
    ok
}

fn ensure_installed_inner(dev: u32, daa: u64) -> bool {
    // This device's assigned mining model (mixed rigs assign a different tier per GPU).
    let (model_id, gguf) = match mining_tier(dev) {
        Some(x) => x,
        None => return false,
    };
    if is_oom_banlisted(dev, &model_id) {
        return false; // this model OOM'd on this GPU before — don't retry (avoids a hot reload spin).
    }
    // An OOM downgrade may have reassigned this device's model — drop a stale (other-model) index
    // so the build below runs for the CURRENT model instead of tripping the N-guard forever.
    crate::pom::invalidate_index_unless(dev, &model_id);
    if crate::pom::active_index(dev).is_none() {
        set_load_phase(dev, LoadPhase::Index); // el % vive en pom::index_build_pct()
    }
    // Build THIS device's possession index once (host, heavy) the first time its PoM activates —
    // deferred from boot so the pre-PoM legacy phase starts immediately. The index is model-specific
    // so each device with a distinct model builds its own.
    if crate::pom::active_index(dev).is_none() {
        // Validate it's a PoM model (daa-independent); the actual tier index is computed per block.
        if !crate::models::is_pom_model(&model_id) {
            return false;
        }
        // Uniform rig: if another GPU already built this MODEL's (host-side) index, share it instead
        // of rebuilding — saves one ~heavy index build + one disk-backed merkle tree per extra GPU.
        if crate::pom::try_share_index(dev, &model_id) {
            info!("PoM[dev {}]: reusing the possession index already built by another GPU", dev);
        } else {
            info!("PoM[dev {}]: building possession index — this can take a while…", dev);
            match crate::pom::WeightIndex::build_from_gguf(&gguf) {
                Ok(idx) => {
                    info!("PoM[dev {}]: weight index ready — N={} chunks", dev, idx.n_chunks);
                    crate::pom::set_index(dev, model_id, idx);
                }
                Err(e) => {
                    log::error!("PoM[dev {}]: index build failed: {}", dev, e);
                    return false;
                }
            }
        }
    }
    // Device 0 shares the inference engine's resident weights (zero-dup) when they happen to be its
    // mining model, and is evicted whenever inference needs the GPU. Every other device loads its
    // OWN standalone copy of its mining model and mines PoM uninterrupted (no inference runs there).
    if dev == 0 {
        // Make device 0's mining model resident again (evicts whatever inference loaded), then share
        // it. If inference currently holds a DIFFERENT model (e.g. a challenge for another GPU's
        // tier), pom_shared returns None and we fall back to a standalone load below.
        if !crate::slm::ensure_loaded(&model_id) {
            log::warn!(
                "PoM: mining model {:02x?} not loadable yet (not in active lineup or load failed) — \
                 GPU0 PoM miner not installed; will retry",
                model_id
            );
            return false;
        }
    }
    // Load the miner (zero-dup on the inference GPU, else a standalone copy). A load OOM surfaces as
    // an Err or, in cudarc, a panic; catch both so the OOM handler can banlist + downgrade instead of
    // crashing the mining thread or hot-spinning on a model that doesn't fit this GPU.
    let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if dev == 0 {
            if let Some((device, shared)) = crate::slm::pom_shared(&model_id) {
                PomGpuMiner::load_shared(&gguf, &device, &shared)
            } else {
                PomGpuMiner::load(&gguf, 0)
            }
        } else {
            PomGpuMiner::load(&gguf, dev)
        }
    }));
    let gm = match loaded {
        Ok(Ok(gm)) => gm,
        Ok(Err(e)) => {
            let msg = e.to_string();
            // INVALID_PTX / PTX JIT failure = the GPU's compute capability is below the PTX floor
            // (sm_80): Turing (RTX 20xx) / Volta / Pascal. That's hardware, not transient — disable
            // this device for PoM with ONE clear message instead of an endless retry-error loop, so
            // capable GPUs in a mixed rig keep mining undisturbed.
            if msg.contains("INVALID_PTX") || msg.contains("PTX JIT") || msg.contains("UNSUPPORTED_PTX") {
                POM_UNSUPPORTED.lock().unwrap_or_else(|p| p.into_inner()).insert(dev);
                log::error!(
                    "PoM: GPU #{dev} does not support Proof-of-Model — its compute capability is below \
                     sm_80 (Ampere). RTX 20xx/Turing, V100/Volta and older CANNOT mine Keryx post-hardfork. \
                     Device #{dev} disabled for PoM; other GPUs keep mining. ({msg})"
                );
            } else if msg.contains("OUT_OF_MEMORY") || msg.contains("out of memory") || msg.contains("Oom") {
                // The model doesn't fit this GPU (driver overhead + KV cache + fragmentation ate the
                // estimate). Permanent for this (dev, model) — banlist and downgrade to a smaller tier.
                log::error!("PoM[dev {}]: load OOM ({}) — banlisting this model and downgrading.", dev, msg);
                oom_banlist_add(dev, model_id);
                downgrade_after_oom(dev, &model_id, daa);
            } else {
                // Transient (inference holds the GPU, file busy…): plain error, retry next block.
                log::error!("PoM: rebuild failed (dev {}): {}", dev, msg);
            }
            return false;
        }
        Err(_) => {
            log::error!("PoM[dev {}]: device miner load panicked (likely OOM) — banlisting this model and downgrading.", dev);
            oom_banlist_add(dev, model_id);
            downgrade_after_oom(dev, &model_id, daa);
            return false;
        }
    };
    let n = gm.n_chunks();
    // N-guard: the gather must match this device's host index, else blocks would be rejected.
    if let Some(idx) = crate::pom::active_index(dev) {
        if n != idx.n_chunks {
            log::error!("PoM: gather N={} != index N={} (dev {}) — refusing to mine (rejected blocks)", n, idx.n_chunks, dev);
            return false;
        }
    }
    install(dev, gm);
    info!("PoM: GPU miner ready on device {} — N={} chunks resident (matches index)", dev, n);
    true
}
