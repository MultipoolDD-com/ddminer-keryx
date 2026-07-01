//! Orquestador: une el cliente stratum con el motor RandomX.
//!
//! - Conecta al pool, recibe jobs → instala/actualiza el motor (reconstruye el
//!   dataset solo si cambia el seed_hash).
//! - Reenvía los shares que encuentra el motor al cliente para hacer `submit`.
//! - Reporta hashrate. Reconecta con backoff si el pool cae.
//!
//! Las piezas (RxEngine + cliente stratum) también se usan sueltas al integrar
//! en el ddminer; aquí se ofrecen empaquetadas para el minero CPU autónomo.

use crate::engine::{RxEngine, Share};
use crate::stratum::{run_client, PoolEvent, StratumConfig};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

/// Selecciona los cores a usar para minar (throttle por afinidad).
/// `threads` = nº de workers; se pinean a los primeros `threads` cores reales.
pub fn pick_cores(threads: usize) -> Vec<usize> {
    let ids: Vec<usize> = core_affinity::get_core_ids()
        .map(|v| v.into_iter().map(|c| c.id).collect())
        .unwrap_or_default();
    if ids.is_empty() {
        (0..threads).collect()
    } else {
        ids.into_iter().take(threads.max(1)).collect()
    }
}

/// Traduce un porcentaje de CPU (~70%) a un nº de threads sobre los cores reales.
pub fn threads_from_percent(percent: u8) -> usize {
    let total = core_affinity::get_core_ids().map(|v| v.len()).unwrap_or(4);
    ((total * percent as usize) / 100).max(1)
}

pub struct MinerConfig {
    pub stratum: StratumConfig,
    pub core_ids: Vec<usize>,
    /// Flag de pausa cooperativa (lo controla quien quiera ceder la CPU, p.ej.
    /// la inferencia OPoI). `false` = minando.
    pub pause_flag: Arc<AtomicBool>,
}

/// Configuración del fee del minero: el `fraction` (p.ej. 0.01 = 1%) del tiempo se mina al
/// `login` del desarrollador, en EL MISMO pool que use el usuario (time-share, estilo XMRig).
pub struct FeeConfig {
    /// Wallet del desarrollador para esta moneda (debe ser una dirección válida de la misma cadena).
    pub login: String,
    /// Fracción del tiempo minada al dev (0.0..1.0). 0 o login vacío ⇒ sin fee.
    pub fraction: f64,
}

/// Minutos por ronda de usuario antes de una ronda corta de fee. Largo a propósito para que el
/// coste de reconstruir el dataset (~pocos s) en cada cambio sea < 0.2% del tiempo de minado.
const USER_ROUND_MINS: u64 = 99;

/// Arranca el minero CPU y bloquea reconectando indefinidamente. Si `fee` está presente (fracción
/// > 0 y login no vacío), alterna rondas usuario/dev para cobrar el fee por time-share.
pub async fn run_miner(cfg: MinerConfig, fee: Option<FeeConfig>) -> ! {
    let fee = fee.filter(|f| f.fraction > 0.0 && !f.login.is_empty());
    let Some(fee) = fee else {
        // Sin fee: minar 100% al usuario (comportamiento original).
        loop {
            run_until(&cfg, None).await;
            eprintln!("rxminer: sesión cerrada; reconectando…");
        }
    };
    // Segundos de fee por ronda de usuario: user * frac/(1-frac). Al 1% ≈ 60 s por 99 min.
    let user_secs = USER_ROUND_MINS * 60;
    let fee_secs = (((user_secs as f64) * fee.fraction / (1.0 - fee.fraction)).ceil() as u64).max(1);
    let fee_cfg = MinerConfig {
        stratum: StratumConfig { login: fee.login.clone(), ..cfg.stratum.clone() },
        core_ids: cfg.core_ids.clone(),
        pause_flag: cfg.pause_flag.clone(),
    };
    println!(
        "  rxminer · fee {:.1}% activo → {} s de cada {} s minan al desarrollador (mismo pool)",
        fee.fraction * 100.0,
        fee_secs,
        user_secs + fee_secs,
    );
    loop {
        run_until(&cfg, Some(Instant::now() + Duration::from_secs(user_secs))).await;
        println!("  rxminer · ronda de fee ({fee_secs}s, {:.1}%) → wallet del desarrollador", fee.fraction * 100.0);
        run_until(&fee_cfg, Some(Instant::now() + Duration::from_secs(fee_secs))).await;
    }
}

/// Mina con `cfg` reconectando ante caídas hasta `deadline` (o indefinidamente si es `None`).
async fn run_until(cfg: &MinerConfig, deadline: Option<Instant>) {
    let mut backoff = 1u64;
    loop {
        if let Some(d) = deadline {
            if Instant::now() >= d {
                return;
            }
        }
        match run_session(cfg, deadline).await {
            Ok(()) => {
                backoff = 1;
                // Ok = deadline alcanzado o el pool cerró limpio. Si hay deadline y ya pasó, fin.
                if deadline.map_or(false, |d| Instant::now() >= d) {
                    return;
                }
            }
            Err(e) => {
                eprintln!("rxminer: error de sesión: {e}; reintento en {backoff}s");
                tokio::time::sleep(Duration::from_secs(backoff)).await;
                backoff = (backoff * 2).min(30);
            }
        }
    }
}

/// Una sesión: conecta, login, bombea jobs/shares hasta desconexión o `deadline`.
async fn run_session(cfg: &MinerConfig, deadline: Option<Instant>) -> Result<(), String> {
    let (events_tx, mut events_rx) = mpsc::unbounded_channel::<PoolEvent>();
    let stratum = StratumConfig {
        host: cfg.stratum.host.clone(),
        port: cfg.stratum.port,
        login: cfg.stratum.login.clone(),
        worker: cfg.stratum.worker.clone(),
        algo: cfg.stratum.algo.clone(),
    };
    let to_pool = run_client(stratum, events_tx).await?;

    let mut engine: Option<RxEngine> = None;
    let mut current_job = String::new();
    let mut accepted = 0u64;
    let mut rejected = 0u64;

    let mut report = tokio::time::interval(Duration::from_secs(10));
    report.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        // Rama de deadline para el time-share del fee: corta la sesión a la hora prevista.
        let until_deadline = async {
            match deadline {
                Some(d) => tokio::time::sleep_until(tokio::time::Instant::from_std(d)).await,
                None => std::future::pending::<()>().await,
            }
        };
        tokio::select! {
            _ = until_deadline => {
                return Ok(());
            }
            _ = report.tick() => {
                if let Some(e) = &engine {
                    let hps = e.take_hashes() as f64 / 10.0;
                    println!(
                        "  rxminer · {hps:.0} H/s · {} threads · job {} · shares {accepted} acc / {rejected} rej",
                        e.num_threads(),
                        &current_job.chars().take(8).collect::<String>(),
                    );
                }
            }
            ev = events_rx.recv() => {
                match ev {
                    Some(PoolEvent::NewJob(rxjob, _session)) => {
                        current_job = rxjob.job_id.clone();
                        let seed = rxjob.seed_hash.clone();
                        match engine.as_mut() {
                            None => {
                                // Primer job: construir motor + lanzar forwarder de shares.
                                print!("  construyendo dataset RandomX (2 GB)…");
                                let (e, shares_rx) = RxEngine::new(seed, cfg.core_ids.clone(), cfg.pause_flag.clone())?;
                                println!(" listo");
                                spawn_share_forwarder(shares_rx, to_pool.clone());
                                e.set_job(rxjob);
                                engine = Some(e);
                            }
                            Some(e) => {
                                if e.seed_hash() != seed.as_slice() {
                                    println!("  seed_hash nuevo → reconstruyendo dataset…");
                                    e.rebuild_for_seed(seed, cfg.core_ids.clone())?;
                                }
                                e.set_job(rxjob);
                            }
                        }
                    }
                    Some(PoolEvent::SubmitResult { accepted: ok }) => {
                        if ok { accepted += 1; } else { rejected += 1; }
                    }
                    Some(PoolEvent::Disconnected(reason)) => {
                        return Err(format!("desconectado: {reason}"));
                    }
                    None => return Ok(()),
                }
            }
        }
    }
}

/// Hilo que drena los shares del motor (canal síncrono) y los reenvía al cliente
/// para `submit`. El cliente añade el session-id internamente.
fn spawn_share_forwarder(
    shares_rx: std::sync::mpsc::Receiver<Share>,
    to_pool: mpsc::UnboundedSender<(String, u32, [u8; 32])>,
) {
    std::thread::Builder::new()
        .name("rx-share-fwd".into())
        .spawn(move || {
            while let Ok(s) = shares_rx.recv() {
                if to_pool.send((s.job_id, s.nonce, s.hash)).is_err() {
                    break; // el cliente murió
                }
            }
        })
        .expect("spawn share forwarder");
}
