use clap::Parser;
use log::LevelFilter;

use crate::Error;

#[derive(Parser, Debug)]
#[clap(name = "ddminer", version, about = "ddminer — Keryx (KRX) GPU miner por Multipool DDMiner (kHeavyHash→PoM @ DAA 37,780,000 + OPoI)\n\nUnder PoM each tier mines AND serves exactly ONE model (1 GPU = 1 tier). No flag = AUTO (each GPU runs the max tier its VRAM fits):\n  --very-light Qwen3-1.7B — tiny GPUs (4GB+), PoM tier 0 post-H2\n  --light      Gemma-3-4B — any GPU (6GB+)\n  --default    Dolphin-3.0-Llama-3.1-8B — RTX 3060 12GB / 3070 / 3080\n  --high       Qwen3-32B (Q4_K_M) — RTX 3090 / 4090 / 5090 (24GB+)\n  --very-high  Llama-3.3-70B — Q4 48GB (pre-H2) → Q2_K_L 32GB / RTX 5090 (post-H2)", term_width = 0)]
pub struct Opt {
    // ── OPoI / Inference ─────────────────────────────────────────────────────

    #[clap(
        long = "very-light",
        help = "Model tier: Qwen3-1.7B — tiny GPUs (4GB+). PoM tier 0 post-H2.",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["light", "default_tier", "high", "very_high"]
    )]
    pub very_light: bool,

    #[clap(
        long = "light",
        help = "Model tier: Gemma-3-4B only — any GPU (6GB+ VRAM)",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["high", "very_high"]
    )]
    pub light: bool,

    #[clap(
        long = "high",
        help = "Model tier: Qwen3-32B (Q4_K_M) — RTX 3090 / 4090 / 5090 (24GB+)",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["light", "very_high"]
    )]
    pub high: bool,

    #[clap(
        long = "very-high",
        help = "Model tier: Llama-3.3-70B — 48GB+ single-GPU (RTX 6000 Ada / A6000 / L40S)",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["light", "high", "default_tier"]
    )]
    pub very_high: bool,

    #[clap(
        long = "default",
        help = "Model tier: Dolphin-3.0-Llama-3.1-8B (8GB+). Pins this tier instead of auto-detecting the max per GPU",
        help_heading = "OPoI / Inference",
        conflicts_with_all = &["light", "high", "very_high"]
    )]
    pub default_tier: bool,

    #[clap(
        long = "cpu-inference",
        help = "Run OPoI inference on the CPU instead of the GPU — frees the GPU for hashing and avoids weak-fp16 GPUs (e.g. GTX 1060). Required on AMD / non-NVIDIA GPUs, where inference cannot use CUDA (--cuda-disable only affects PoW, not inference). Pairs well with --light.",
        help_heading = "OPoI / Inference"
    )]
    pub cpu_inference: bool,

    #[clap(
        long = "ipfs-url",
        help = "IPFS Kubo API URL for uploading inference results",
        help_heading = "OPoI / Inference",
        default_value = "http://127.0.0.1:5001"
    )]
    pub ipfs_url: String,

    #[clap(
        long = "escrow-key-file",
        help = "Path to the OPoI escrow private key file (auto-generated if absent)",
        help_heading = "OPoI / Inference",
        default_value = "escrow.key"
    )]
    pub escrow_key_file: String,

    #[clap(
        long = "escrow-state-file",
        help = "Path to the escrow claim state file",
        help_heading = "OPoI / Inference",
        default_value = "escrow_state.json"
    )]
    pub escrow_state_file: String,

    #[clap(
        long = "recover-escrow",
        help = "Rebuild escrow_state.json by querying the Keryx public API. Exits after recovery.",
        help_heading = "OPoI / Inference"
    )]
    pub recover_escrow: bool,

    #[clap(
        long = "recover-escrow-api",
        help = "Base URL of the Keryx API to use for escrow recovery",
        help_heading = "OPoI / Inference",
        default_value = "https://keryx-labs.com"
    )]
    pub recover_escrow_api: String,

    // ── Mining ────────────────────────────────────────────────────────────────

    #[clap(short, long, help = "Enable debug logging level")]
    pub debug: bool,

    #[clap(short = 'a', long = "mining-address", help = "The Keryx address for the miner reward")]
    pub mining_address: Option<String>,

    #[clap(short = 's', long = "keryxd-address", default_value = "stratum+tcp://multipooldd.com:5555", help = "Stratum pool URL (stratum+tcp://host:port). Default: Multipool DDMiner pool. Use grpc://<keryxd> for SOLO/Proof-of-Model mining")]
    pub keryxd_address: String,

    #[clap(long = "devfund-percent", help = "The percentage of blocks to send to the devfund (minimum 2%)", default_value = "2", parse(try_from_str = parse_devfund_percent))]
    pub devfund_percent: u16,

    #[clap(short, long, help = "Keryxd port [default: Mainnet = 22110, Testnet = 22211]")]
    port: Option<u16>,

    #[clap(long, help = "Use testnet instead of mainnet [default: false]")]
    testnet: bool,

    #[clap(short = 't', long = "threads", help = "Amount of CPU miner threads to launch [default: 0]")]
    pub num_threads: Option<u16>,

    #[clap(
        long = "mine-when-not-synced",
        help = "Mine even when keryxd says it is not synced",
        long_help = "Mine even when keryxd says it is not synced, only useful when passing `--allow-submit-block-when-not-synced` to keryxd  [default: false]"
    )]
    pub mine_when_not_synced: bool,

    #[clap(
        long = "cpu",
        value_name = "CPU_SPEC",
        help = "Also mine RandomX on the CPU, in parallel with GPU mining, to ANY pool. \
                Format: \"<coin> <wallet> <host>[:port]\" — coin = xmr|qrl. \
                Example: --cpu \"xmr 4xWALLET... pool.supportxmr.com:3333\". \
                Port defaults per coin if omitted. A 1% dev fee applies (time-share).",
        help_heading = "CPU / RandomX"
    )]
    pub cpu: Option<String>,

    #[clap(
        long = "xmr",
        value_name = "XMR_ADDRESS",
        help = "[legacy] Mine Monero (XMR) on the CPU to the default Multipool DDMiner pool. Prefer --cpu for any pool.",
        help_heading = "CPU / RandomX"
    )]
    pub xmr: Option<String>,

    #[clap(
        long = "qrl",
        value_name = "QRL_ADDRESS",
        help = "[legacy] Mine QRL on the CPU to the default Multipool DDMiner pool. Prefer --cpu for any pool.",
        help_heading = "CPU / RandomX"
    )]
    pub qrl: Option<String>,

    #[clap(
        long = "cpu-percent",
        default_value = "70",
        help = "Percent of CPU cores to use for RandomX mining (--xmr/--qrl) [default: 70]"
    )]
    pub cpu_percent: u8,

    #[clap(
        long = "cpu-threads",
        help = "Exact RandomX thread count (overrides --cpu-percent)"
    )]
    pub cpu_threads: Option<usize>,

    #[clap(skip)]
    pub devfund_address: String,

    #[clap(
        long = "bench-pom",
        help = "Benchmark the PoM walk kernel across CUDA block sizes (pool-independent) and exit. Loads the Dolphin-8B weights on GPU 0.",
        hide = true
    )]
    pub bench_pom: bool,
}

fn parse_devfund_percent(s: &str) -> Result<u16, &'static str> {
    let err = "devfund-percent should be --devfund-percent=XX.YY up to 2 numbers after the dot";
    let mut splited = s.split('.');
    let prefix = splited.next().ok_or(err)?;
    // if there's no postfix then it's 0.
    let postfix = splited.next().ok_or(err).unwrap_or("0");
    // error if there's more than a single dot
    if splited.next().is_some() {
        return Err(err);
    };
    // error if there are more than 2 numbers before or after the dot
    if prefix.len() > 2 || postfix.len() > 2 {
        return Err(err);
    }
    let postfix: u16 = postfix.parse().map_err(|_| err)?;
    let prefix: u16 = prefix.parse().map_err(|_| err)?;
    // can't be more than 99.99%,
    if prefix >= 100 || postfix >= 100 {
        return Err(err);
    }
    if prefix < 2 {
        // Force at least 2 percent
        return Ok(200u16);
    }
    // DevFund is out of 10_000
    Ok(prefix * 100 + postfix)
}

impl Opt {
    pub fn process(&mut self) -> Result<(), Error> {
        if self.recover_escrow {
            return Ok(());
        }
        if self.mining_address.is_none() {
            return Err("--mining-address is required".into());
        }
        if self.xmr.is_some() && self.qrl.is_some() {
            return Err("Use either --xmr or --qrl, not both (one RandomX coin at a time on the CPU)".into());
        }
        // ── Dirección de pool ────────────────────────────────────────────────
        // Por defecto mina contra el pool de Multipool DDMiner, pero acepta CUALQUIER
        // host stratum (`stratum+tcp://host:puerto`). EXCEPCIÓN: un address
        // `grpc://` apunta a un keryxd local para minado SOLO, que es la ÚNICA vía
        // válida bajo Proof-of-Model — un share de pool (PartialBlock) no puede
        // transportar la prueba de posesión por-minero, así que PoM exige
        // FullBlock + prueba vía grpc.
        const DEFAULT_POOL_HOST: &str = "multipooldd.com";
        if self.keryxd_address.is_empty() {
            self.keryxd_address = format!("stratum+tcp://{}:5555", DEFAULT_POOL_HOST);
        }
        if self.keryxd_address.starts_with("grpc://") {
            // Solo PoM contra keryxd local: minado en solitario.
            log::warn!(
                "Modo SOLO (grpc) habilitado: {} — requerido para minar Proof-of-Model (el pool stratum no puede enviar la prueba de posesión).",
                self.keryxd_address
            );
        } else {
            // Normaliza a stratum+tcp://host:puerto, aceptando cualquier pool.
            let no_scheme = self
                .keryxd_address
                .strip_prefix("stratum+tcp://")
                .unwrap_or(self.keryxd_address.as_str());
            let mut it = no_scheme.split(':');
            let host = it.next().unwrap_or("").to_string();
            let port = it.next().unwrap_or("5555").to_string();
            if host.is_empty() {
                return Err(format!(
                    "Dirección de pool inválida: '{}'. Usa stratum+tcp://<host>:<puerto>, o grpc://<keryxd> para minado SOLO (Proof-of-Model).",
                    self.keryxd_address
                )
                .into());
            }
            self.keryxd_address = format!("stratum+tcp://{}:{}", host, port);
            log::info!("Pool stratum: {}", self.keryxd_address);
        }

        if self.num_threads.is_none() {
            self.num_threads = Some(0);
        }

        // Devfund fully disabled (Multipool DDMiner miner): 0% and no devfund address
        // hardcoded anywhere. main.rs skips devfund entirely when percent == 0,
        // so devfund_address is left empty and never used.
        self.devfund_percent = 0;
        Ok(())
    }

    #[allow(dead_code)] // pool-locked: solo/grpc port no longer used, kept for the -p flag
    fn port(&mut self) -> u16 {
        *self.port.get_or_insert(if self.testnet { 22211 } else { 22110 })
    }

    pub fn log_level(&self) -> LevelFilter {
        if self.debug {
            LevelFilter::Debug
        } else {
            // Salida limpia: solo warnings/errores del logger. Banner, info de GPU
            // y la línea de estado (temp/fans/hashrate/shares) se imprimen aparte.
            LevelFilter::Warn
        }
    }
}
