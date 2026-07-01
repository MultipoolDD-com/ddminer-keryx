//! Subsistema de minado CPU RandomX (XMR / QRL) para el ddminer.
//!
//! Mina en paralelo al minado GPU de Keryx (kHeavyHash): la GPU sigue con KRX
//! y la CPU aporta RandomX a CUALQUIER pool que elija el usuario (vía `--cpu`).
//! Se PAUSA durante la inferencia OPoI en CPU vía [`pause`]/[`resume`] para no
//! pelear por el ancho de banda de memoria. Cobra un fee del 1% por time-share
//! (mina al wallet del desarrollador el 1% del tiempo, en el mismo pool).

use once_cell::sync::OnceCell;
use rxengine::orchestrator::{pick_cores, run_miner, threads_from_percent, FeeConfig, MinerConfig};
use rxengine::stratum::StratumConfig;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Pool por defecto si el usuario no especifica host en `--cpu`.
const DEFAULT_POOL_HOST: &str = "multipooldd.com";

/// Fee del minado CPU: fracción del tiempo minada al wallet del desarrollador (1%).
const FEE_FRACTION: f64 = 0.01;

/// Wallets de fee del desarrollador, POR MONEDA (cadenas distintas → direcciones distintas).
/// Vacío ⇒ sin fee para esa moneda (a prueba de fallos: nunca mina a una dirección inválida).
/// RELLENAR con las direcciones reales de Multipool DDMiner antes de distribuir.
const FEE_WALLET_XMR: &str = "";
const FEE_WALLET_QRL: &str = "";

/// Flag global de pausa del minado RandomX. La inferencia OPoI en CPU lo activa
/// para cederle la CPU y lo libera al terminar.
static RX_PAUSE: OnceCell<Arc<AtomicBool>> = OnceCell::new();

/// Moneda CPU seleccionable al lanzar.
#[derive(Clone, Copy, Debug)]
pub enum RxCoin {
    Xmr,
    Qrl,
}

impl RxCoin {
    /// Puerto del multipool del socio (confirmado en su código stratum-pool).
    fn port(self) -> u16 {
        match self {
            RxCoin::Xmr => 4334,
            RxCoin::Qrl => 4335,
        }
    }
    fn name(self) -> &'static str {
        match self {
            RxCoin::Xmr => "XMR",
            RxCoin::Qrl => "QRL",
        }
    }
    /// Puerto por defecto del multipool Multipool DDMiner para esta moneda.
    pub fn default_port(self) -> u16 {
        self.port()
    }
    /// Wallet de fee del desarrollador para esta moneda ("" ⇒ sin fee).
    fn fee_wallet(self) -> &'static str {
        match self {
            RxCoin::Xmr => FEE_WALLET_XMR,
            RxCoin::Qrl => FEE_WALLET_QRL,
        }
    }
}

/// Pausa el minado RandomX. Lo llama el código de inferencia OPoI en CPU para
/// cederle la CPU/memoria. No-op si el subsistema RandomX no está activo.
pub fn pause() {
    if let Some(p) = RX_PAUSE.get() {
        p.store(true, Ordering::SeqCst);
    }
}

/// Reanuda el minado RandomX tras la inferencia. No-op si no está activo.
pub fn resume() {
    if let Some(p) = RX_PAUSE.get() {
        p.store(false, Ordering::SeqCst);
    }
}

/// ¿Está activo el subsistema RandomX? (para decidir si pausar en la inferencia).
pub fn is_active() -> bool {
    RX_PAUSE.get().is_some()
}

/// Arranca el subsistema RandomX en una tarea de fondo. `wallet` = dirección de cobro (XMR/QRL).
/// `host`/`port` = pool elegido por el usuario (`None` → default Multipool DDMiner de la moneda).
/// `threads`: `None` → ~70% de los cores. Cobra el fee del 1% por time-share si hay wallet de fee.
pub fn spawn(
    coin: RxCoin,
    wallet: String,
    host: Option<String>,
    port: Option<u16>,
    worker: String,
    threads: Option<usize>,
    percent: u8,
) {
    let pause_flag = Arc::new(AtomicBool::new(false));
    // Si ya se inicializó (doble arranque), no duplicar.
    if RX_PAUSE.set(pause_flag.clone()).is_err() {
        eprintln!("  ! Minado CPU RandomX ya estaba activo; ignorando segundo arranque");
        return;
    }
    let host = host.unwrap_or_else(|| DEFAULT_POOL_HOST.to_string());
    let port = port.unwrap_or_else(|| coin.default_port());
    let n = threads.unwrap_or_else(|| threads_from_percent(percent));
    let core_ids = pick_cores(n);
    println!(
        "  + Minado CPU {} (RandomX) → {host}:{port} · {} threads · {} (pausa durante inferencia OPoI)",
        coin.name(),
        core_ids.len(),
        wallet.chars().take(12).collect::<String>() + "…",
    );
    let cfg = MinerConfig {
        stratum: StratumConfig {
            host,
            port,
            login: wallet,
            worker,
            algo: "rx/0".to_string(),
        },
        core_ids,
        pause_flag,
    };
    // Fee del 1% por time-share (mismo pool, wallet del dev). Sin wallet de fee ⇒ None (sin fee).
    let fee = {
        let w = coin.fee_wallet();
        if w.is_empty() {
            None
        } else {
            Some(FeeConfig { login: w.to_string(), fraction: FEE_FRACTION })
        }
    };
    // El RxEngine contiene el dataset de RandomX (puntero FFI, no Send), así que
    // NO puede ir en el runtime multi-hilo principal. Corre en su propio hilo
    // con un runtime tokio current-thread dedicado.
    std::thread::Builder::new()
        .name("rx-cpu-miner".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("rx-cpu-miner: no pude crear el runtime tokio");
            rt.block_on(run_miner(cfg, fee));
        })
        .expect("rx-cpu-miner: no pude crear el hilo");
}
