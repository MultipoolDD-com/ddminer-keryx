#![cfg_attr(all(test, feature = "bench"), feature(test))]

use std::env::consts::DLL_EXTENSION;
use std::env::current_exe;
use std::error::Error as StdError;
use std::ffi::OsStr;

use clap::{App, FromArgMatches, IntoApp};
use keryx_miner::PluginManager;
use log::{error, info, warn};
use rand::{thread_rng, RngCore};
use std::fs;
use std::sync::atomic::AtomicU16;
use std::sync::Arc;
use std::time::Duration;

use crate::cli::Opt;
use crate::client::grpc::KeryxdHandler;
use crate::client::stratum::StratumHandler;
use crate::client::Client;
use crate::miner::MinerManager;
use crate::target::Uint256;

mod cli;
mod client;
mod escrow;
mod ipfs;
mod keryxd_messages;
mod miner;
mod pow;
mod target;
mod watch;

const WHITELIST: [&str; 4] = ["libkeryxcuda", "libkeryxopencl", "keryxcuda", "keryxopencl"];

pub mod proto {
    #![allow(clippy::derive_partial_eq_without_eq)]
    tonic::include_proto!("protowire");
    // include!("protowire.rs"); // FIXME: https://github.com/intellij-rust/intellij-rust/issues/6579
}

pub type Error = Box<dyn StdError + Send + Sync + 'static>;

type Hash = Uint256;

/// Attempt to install the CUDA runtime libraries candle needs, on a Debian/Ubuntu host (HiveOS).
///
/// OPoI GPU inference needs cuBLAS, cuBLASLt and cuRAND — candle creates handles for all three
/// when it opens the CUDA device. These ship with the CUDA toolkit but not with the bare NVIDIA
/// driver that mining rigs usually have. Rather than forcing miners to run apt by hand, we add
/// the NVIDIA CUDA repo and install `libcublas-12-2` (cuBLAS + cuBLASLt) and `libcurand-12-2`
/// ourselves, then register their directory with ldconfig. Runs as root on HiveOS, so no sudo.
///
/// Version 12-2 (not 12-6) is deliberate: the binary's candle kernels are compiled with the
/// CUDA 12.2 toolkit so they JIT on driver >= 535 (typical HiveOS), and the cuBLAS runtime must
/// match that minimum. Installing 12-6 here would pull a runtime needing driver >= 560.
/// Returns true on success.
#[cfg(target_os = "linux")]
fn install_cuda_libs() -> bool {
    use std::process::Command;
    // Only meaningful where apt-get exists (Debian/Ubuntu, incl. HiveOS).
    let has_apt = Command::new("sh")
        .args(["-c", "command -v apt-get"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !has_apt {
        error!("CUDA lib auto-install needs apt-get (Debian/Ubuntu) — not found on this system.");
        return false;
    }
    // The CUDA libs install into /usr/local/cuda-*/targets/x86_64-linux/lib, which is NOT in
    // the default loader search path. Installing alone is not enough: we must register that
    // directory with ldconfig so dlopen("libcublas.so.12" / "libcurand.so.10") resolves it.
    let script = r#"set -e
cd /tmp
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring.deb
dpkg -i cuda-keyring.deb
apt-get update -qq
apt-get install -y -qq libcublas-12-2 libcurand-12-2
CUBLAS_PATH=$(find /usr/local /usr/lib -name 'libcublas.so.12' 2>/dev/null | head -1)
if [ -z "$CUBLAS_PATH" ]; then echo "libcublas.so.12 not found after install"; exit 1; fi
echo "$(dirname "$CUBLAS_PATH")" > /etc/ld.so.conf.d/keryx-cuda.conf
ldconfig
ldconfig -p | grep -q libcublas.so.12 || { echo "libcublas still not in loader cache"; exit 1; }
ldconfig -p | grep -q libcurand.so   || { echo "libcurand still not in loader cache"; exit 1; }
rm -f cuda-keyring.deb"#;
    Command::new("bash")
        .args(["-c", script])
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

#[cfg(target_os = "windows")]
fn adjust_console() -> Result<(), Error> {
    let console = win32console::console::WinConsole::input();
    let mut mode = console.get_mode()?;
    mode = (mode & !win32console::console::ConsoleMode::ENABLE_QUICK_EDIT_MODE)
        | win32console::console::ConsoleMode::ENABLE_EXTENDED_FLAGS;
    console.set_mode(mode)?;
    Ok(())
}

fn filter_plugins(dirname: &str) -> Vec<String> {
    match fs::read_dir(dirname) {
        Ok(readdir) => readdir
            .map(|entry| entry.unwrap().path())
            .filter(|fname| {
                fname.is_file()
                    && fname.extension().is_some()
                    && fname.extension().and_then(OsStr::to_str).unwrap_or_default().starts_with(DLL_EXTENSION)
            })
            .filter(|fname| WHITELIST.iter().any(|lib| *lib == fname.file_stem().and_then(OsStr::to_str).unwrap()))
            .map(|path| path.to_str().unwrap().to_string())
            .collect::<Vec<String>>(),
        _ => Vec::<String>::new(),
    }
}

/// Parse a `--cpu` spec: `"<coin> <wallet> <host>[:port]"` → (coin, wallet, Some(host), port).
/// coin ∈ {xmr, qrl} (case-insensitive). Port is optional (defaults per coin downstream).
fn parse_cpu_spec(
    spec: &str,
) -> Result<(keryx_miner::cpu_randomx::RxCoin, String, Option<String>, Option<u16>), String> {
    let parts: Vec<&str> = spec.split_whitespace().collect();
    if parts.len() != 3 {
        return Err(format!("esperaba 3 campos \"<coin> <wallet> <host>[:port]\", recibí {}", parts.len()));
    }
    let coin = match parts[0].to_lowercase().as_str() {
        "xmr" | "monero" => keryx_miner::cpu_randomx::RxCoin::Xmr,
        "qrl" => keryx_miner::cpu_randomx::RxCoin::Qrl,
        other => return Err(format!("moneda desconocida '{other}' (usa xmr o qrl)")),
    };
    let wallet = parts[1].to_string();
    if wallet.is_empty() {
        return Err("wallet vacío".into());
    }
    let (host, port) = match parts[2].rsplit_once(':') {
        Some((h, p)) => {
            let port: u16 = p.parse().map_err(|_| format!("puerto inválido '{p}'"))?;
            (h.to_string(), Some(port))
        }
        None => (parts[2].to_string(), None),
    };
    if host.is_empty() {
        return Err("host vacío".into());
    }
    Ok((coin, wallet, Some(host), port))
}

/// Per-GPU total VRAM in MB, indexed by CUDA device ordinal (one nvidia-smi line per GPU).
/// Empty on failure → callers fall back to assigning every GPU the same (highest staged) tier.
fn per_gpu_vram_mb() -> Vec<u64> {
    match std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .filter_map(|l| l.trim().parse::<u64>().ok())
            .collect(),
        _ => Vec::new(),
    }
}

/// Query GPU stats via nvidia-smi and warn on power/VRAM issues for the selected model tier.
///
/// VRAM requirements (GGUF weights only, not counting CUDA workspace):
///   TinyLlama-1.1B  →  ~1.5 GB
///   DeepSeek-R1-8B  →  ~5 GB
///   DeepSeek-R1-32B → ~19 GB   (requires ≥24 GB card)
///   LLaMA-3.3-70B   → ~28 GB   (requires ≥40 GB card — does NOT fit on RTX 3090)
///
/// Power thresholds empirically derived: Xid 32 observed at ≤300W on RTX 3090 with 32B GGUF.
fn check_gpu_power_limit(needs_high: bool, needs_very_high: bool) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=power.limit,power.max_limit,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    // nvidia-smi prints one line per GPU; the power + VRAM check applies to GPU 0
    // (the device the miner mines/serves on).
    let (current_w, vram_mb) = match output {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            let mut cur = 0u32;
            let mut vram = 0u64;
            for (i, line) in s.trim().lines().take(1).enumerate() {
                let mut parts = line.split(',');
                let line_cur: f32 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0.0);
                let _max: f32 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0.0);
                let line_vram: u64 = parts.next().unwrap_or("0").trim().parse().unwrap_or(0);
                if i == 0 {
                    cur = line_cur as u32;
                }
                vram += line_vram;
            }
            (cur, vram)
        }
        _ => return,
    };

    // VRAM sufficiency for the selected tier (Q4_K_M weights + KV cache + CUDA workspace).
    // Insufficient VRAM means GPU inference for this tier will OOM. This is non-fatal — a
    // host/CPU path can still serve it — so warn rather than error, and do NOT then claim the
    // model is "ready" on the same GPU (the contradictory ERROR-then-ready pair).
    let (model_label, min_vram_mb): (&str, u64) = if needs_very_high {
        ("Llama-3.3-70B (--very-high)", 46_000)
    } else if needs_high {
        ("Qwen3-32B (--high)", 20_000)
    } else {
        ("Dolphin-8B (default)", 8_000)
    };

    if vram_mb < min_vram_mb {
        log::warn!(
            "⚠  {} needs ≥{} GB VRAM but only {} GB on this GPU — GPU inference for this tier \
             will OOM. Use a smaller tier (--high Qwen3-32B / --light Gemma-3-4B) or serve it \
             via a host/CPU path.",
            model_label,
            min_vram_mb / 1024,
            vram_mb / 1024,
        );
    } else {
        log::info!("GPU: {}W PL, {} MB VRAM — ready for {}", current_w, vram_mb, model_label);
    }
}

/// GPU 0 total VRAM (MB) via nvidia-smi, or None when nvidia-smi is unavailable or
/// unparseable (e.g. AMD-only machines). GPU 0 is the device the miner mines/serves on.
fn query_vram_mb() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .and_then(|l| l.trim().parse::<u64>().ok())
}

/// OPoI capability gate (layer A): drop the models this machine cannot actually
/// serve on GPU 0, so the `ai:cap` announcement never promises a model the miner
/// would fail to load. Skipped when nvidia-smi is unavailable (CPU-fallback setups
/// keep working).
fn filter_specs_by_vram(
    specs: &'static [&'static keryx_miner::models::ModelSpec],
) -> &'static [&'static keryx_miner::models::ModelSpec] {
    let Some(gpu0_mb) = query_vram_mb() else {
        log::warn!("Cannot query GPU VRAM (nvidia-smi) — skipping the model capability gate.");
        return specs;
    };
    let kept: Vec<&'static keryx_miner::models::ModelSpec> = specs
        .iter()
        .copied()
        .filter(|spec| {
            if spec.min_vram_mb <= gpu0_mb {
                true
            } else {
                log::warn!(
                    "✗  '{}' needs ≥{} MB VRAM but only {} MB on GPU 0 — model NOT announced (ai:cap) and not downloaded.",
                    spec.name,
                    spec.min_vram_mb,
                    gpu0_mb,
                );
                false
            }
        })
        .collect();
    if kept.len() == specs.len() {
        specs
    } else {
        // Leaked once at startup to keep the &'static API of init_supported.
        Box::leak(kept.into_boxed_slice())
    }
}

/// Build the staged lineup at `daa` as the deduped UNION of `specs_for` over every tier in `tiers`,
/// then VRAM-filter it. For a heterogeneous rig this stages each GPU's model (e.g. Qwen3-32B for a
/// 4090 and Gemma for a 3060Ti) so per-GPU assignment can give each device its best fit. Leaked
/// once at startup to keep the `&'static` lineup API.
fn union_lineup(
    daa: u64,
    tiers: &[keryx_miner::models::Tier],
) -> &'static [&'static keryx_miner::models::ModelSpec] {
    let mut v: Vec<&'static keryx_miner::models::ModelSpec> = Vec::new();
    for &t in tiers {
        for &s in keryx_miner::models::specs_for(daa, t) {
            if !v.iter().any(|x| x.model_id == s.model_id) {
                v.push(s);
            }
        }
    }
    filter_specs_by_vram(Box::leak(v.into_boxed_slice()))
}

/// Banner de arranque. Usa println! para verse siempre, aunque el logger esté en Warn.
fn print_banner(mining_address: &str, pool_address: &str) {
    use std::io::IsTerminal;
    // Color ANSI solo en terminal real; plano si va a log/redirección.
    let (cy, bd, dm, rs) = if std::io::stdout().is_terminal() {
        ("\x1b[36m", "\x1b[1;36m", "\x1b[2m", "\x1b[0m")
    } else {
        ("", "", "", "")
    };
    println!();
    println!(r"{cy}   ____  ____  __  __ _                 {rs}");
    println!(r"{cy}  |  _ \|  _ \|  \/  (_)_ __   ___ _ __ {rs}");
    println!(r"{cy}  | | | | | | | |\/| | | '_ \ / _ \ '__|{rs}");
    println!(r"{cy}  | |_| | |_| | |  | | | | | |  __/ |   {rs}");
    println!(r"{cy}  |____/|____/|_|  |_|_|_| |_|\___|_|   {rs}");
    println!("{bd}        por Multipool DDMiner{rs}  ·  Keryx (KRX) GPU miner v{}", env!("CARGO_PKG_VERSION"));
    // Show the pool actually in use (opt.process() already normalized it). grpc:// = solo mining.
    let pool_display = if let Some(rest) = pool_address.strip_prefix("grpc://") {
        format!("SOLO (grpc): {}", rest)
    } else {
        pool_address.strip_prefix("stratum+tcp://").unwrap_or(pool_address).to_string()
    };
    println!("{dm}        Pool: {pool_display}  ·  kHeavyHash→PoM (DAA 37,780,000) + OPoI{rs}");
    println!("{dm}        Mining to: {}{rs}", mining_address);
    if let Ok(o) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,name,driver_version,power.limit", "--format=csv,noheader,nounits"])
        .output()
    {
        if o.status.success() {
            for line in String::from_utf8_lossy(&o.stdout).lines() {
                let f: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if f.len() >= 4 {
                    println!("{dm}        GPU{} {}  ·  driver {}  ·  {}W{rs}", f[0], f[1], f[2], f[3]);
                }
            }
        }
    }
    println!();
}

async fn get_client(
    keryxd_address: String,
    mining_address: String,
    mine_when_not_synced: bool,
    block_template_ctr: Arc<AtomicU16>,
    escrow_privkey: Option<String>,
    escrow_state_file: String,
    ipfs_url: String,
) -> Result<Box<dyn Client + 'static>, Error> {
    if keryxd_address.starts_with("stratum+tcp://") {
        let (_schema, address) = keryxd_address.split_once("://").unwrap();
        Ok(StratumHandler::connect(
            address.to_string().clone(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
            ipfs_url.clone(),
        )
        .await?)
    } else if keryxd_address.starts_with("grpc://") {
        Ok(KeryxdHandler::connect(
            keryxd_address.clone(),
            mining_address.clone(),
            mine_when_not_synced,
            Some(block_template_ctr.clone()),
            escrow_privkey,
            escrow_state_file,
            ipfs_url,
        )
        .await?)
    } else {
        Err("Did not recognize pool/grpc address schema".into())
    }
}

/// Worker name para el minado CPU RandomX = hostname saneado (primer label),
/// para que el pool identifique este rig. Fallback "rig".
fn cpu_worker_name() -> String {
    std::env::var("HOSTNAME")
        .ok()
        .or_else(|| std::fs::read_to_string("/etc/hostname").ok())
        .map(|s| s.split('.').next().unwrap_or("rig").trim().to_string())
        .map(|s| {
            s.chars()
                .filter(|c| c.is_alphanumeric() || *c == '-' || *c == '_')
                .take(32)
                .collect::<String>()
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "rig".into())
}

async fn client_main(
    opt: &Opt,
    block_template_ctr: Arc<AtomicU16>,
    plugin_manager: &PluginManager,
    escrow_privkey: Option<String>,
) -> Result<(), Error> {
    let ipfs_url = opt.ipfs_url.clone();
    tokio::task::spawn_blocking(move || crate::ipfs::ensure_daemon(&ipfs_url)).await.ok();

    let mut client = get_client(
        opt.keryxd_address.clone(),
        opt.mining_address.clone().unwrap_or_default(),
        opt.mine_when_not_synced,
        block_template_ctr.clone(),
        escrow_privkey,
        opt.escrow_state_file.clone(),
        opt.ipfs_url.clone(),
    )
    .await?;

    if opt.devfund_percent > 0 {
        client.add_devfund(opt.devfund_address.clone(), opt.devfund_percent);
    }
    client.register().await?;
    let mut miner_manager = MinerManager::new(client.get_block_channel(), opt.num_threads, plugin_manager);
    client.listen(&mut miner_manager).await?;
    drop(miner_manager);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Error> {
    #[cfg(target_os = "windows")]
    adjust_console().unwrap_or_else(|e| {
        eprintln!("WARNING: Failed to protect console ({}). Any selection in console will freeze the miner.", e)
    });
    let mut path = current_exe().unwrap_or_default();
    path.pop(); // Getting the parent directory
    let plugins = filter_plugins(path.to_str().unwrap_or("."));
    let (app, mut plugin_manager): (App, PluginManager) = keryx_miner::load_plugins(Opt::into_app(), &plugins)?;

    let matches = app.get_matches();

    let worker_count = plugin_manager.process_options(&matches)?;
    let mut opt: Opt = Opt::from_arg_matches(&matches)?;
    // Pool-independent kernel benchmark: load Dolphin-8B on GPU 0, sweep block sizes, exit.
    // Runs before opt.process() so it needs neither a mining address nor a pool.
    if opt.bench_pom {
        env_logger::builder().filter_level(log::LevelFilter::Info).parse_default_env().init();
        let gguf = keryx_miner::slm::gguf_path_for(&keryx_miner::models::DOLPHIN_LLAMA3_8B)
            .to_string_lossy()
            .into_owned();
        println!("=== PoM walk-kernel benchmark (GPU 0, Dolphin-8B, pool-independent) ===");
        keryx_miner::pom_gpu::bench(&gguf, 0).map_err(|e| -> Error { format!("bench-pom: {e}").into() })?;
        return Ok(());
    }
    opt.process()?;
    env_logger::builder().filter_level(opt.log_level()).parse_default_env().init();
    print_banner(opt.mining_address.as_deref().unwrap_or("(recovery mode)"), &opt.keryxd_address);

    // Recovery mode: rebuild escrow_state.json from the Keryx public API, then exit.
    // Must run before escrow key loading to avoid creating a new random key on disk.
    // Uses escrow.key to derive the pubkey — only claimable UTXOs are returned.
    if opt.recover_escrow {
        let escrow_privkey = match escrow::load_key(&opt.escrow_key_file) {
            Ok(k) => k,
            Err(e) => {
                error!("{}", e);
                return Err(e.into());
            }
        };
        let pubkey_hex = match escrow::pubkey_hex_from_privkey(&escrow_privkey) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to derive pubkey from escrow key: {}", e);
                return Err(e.into());
            }
        };
        let url = format!("{}/api/v1/escrow/{}", opt.recover_escrow_api.trim_end_matches('/'), pubkey_hex);
        info!("Querying escrow UTXOs from {}", url);

        #[derive(serde::Deserialize)]
        struct ApiEscrowEntry {
            coinbase_txid: String,
            block_hash: String,
            confirm_daa: i64,
            amount_sompi: i64,
            output_index: i64,
        }

        let url_clone = url.clone();
        let api_entries: Vec<ApiEscrowEntry> = tokio::task::spawn_blocking(move || {
            let response = ureq::get(&url_clone)
                .call()
                .map_err(|e| format!("HTTP request failed: {}", e))?;
            serde_json::from_reader::<_, Vec<ApiEscrowEntry>>(response.into_reader())
                .map_err(|e| format!("JSON parse error: {}", e))
        })
        .await
        .map_err(|e| format!("spawn_blocking failed: {}", e))??;

        let entries: Vec<escrow::EscrowEntry> = api_entries
            .into_iter()
            .map(|a| escrow::EscrowEntry {
                coinbase_txid: a.coinbase_txid,
                block_hash: a.block_hash,
                confirm_daa: a.confirm_daa as u64,
                amount_sompi: a.amount_sompi as u64,
                output_index: a.output_index as u32,
                claimed: false,
                slashed: false,
                orphan_slashed: false,
                orphan_retries: 0,
                orphan_retry_after_daa: None,
                is_inference: false,
            })
            .collect();

        let total_sompi: u64 = entries.iter().map(|e| e.amount_sompi).sum();
        let count = entries.len();
        let state = escrow::EscrowState { entries };
        let json = serde_json::to_string_pretty(&state)?;
        fs::write(&opt.escrow_state_file, &json)?;

        info!(
            "Recovered {} escrow entries — claimable: {:.4} KRX",
            count,
            total_sompi as f64 / 1e8
        );
        info!("State saved to '{}'.", opt.escrow_state_file);
        return Ok(());
    }

    // Resolve OPoI escrow private key (once, before the reconnect loop).
    let escrow_privkey: Option<String> = match escrow::load_or_generate_key(&opt.escrow_key_file) {
        Ok(k) => {
            info!("OPoI: escrow key loaded from '{}'.", opt.escrow_key_file);
            Some(k)
        }
        Err(e) => {
            error!("Failed to load/generate OPoI escrow key: {}", e);
            return Err(e.into());
        }
    };

    // Phase-3 OPoI / PoM: load inference models before mining starts. Under PoM each tier
    // mines AND serves exactly ONE model (1 GPU = 1 tier); the lineup is DAA-gated so the
    // legacy kHeavyHash lineup runs until the hardfork and the uncensored PoM lineup swaps
    // in at OPOI_V2_ACTIVATION_DAA without a restart.
    //   --light      → Gemma-3-4B        --high       → Qwen3-32B
    //   --very-high  → Llama-3.3-70B     (no flag)    → AUTO per GPU
    //
    // AUTO (default, no flag): detect each GPU's VRAM and run the highest tier it can fit; the
    // staged lineup is the UNION of all GPUs' tiers, so a heterogeneous rig (e.g. 4090 + 3060Ti)
    // mines Qwen3-32B on the 4090 and Dolphin-8B on the 3060Ti — no flags, each GPU at its max.
    use keryx_miner::models::Tier;
    let explicit = if opt.very_high {
        Some(Tier::VeryHigh)
    } else if opt.high {
        Some(Tier::High)
    } else if opt.default_tier {
        Some(Tier::Default)
    } else if opt.light {
        Some(Tier::Light)
    } else if opt.very_light {
        Some(Tier::VeryLight)
    } else {
        None
    };
    let tiers: Vec<Tier> = match explicit {
        Some(t) => {
            info!("Tier forced by flag: {:?} (one model on every GPU).", t);
            vec![t]
        }
        None => {
            let vrams = per_gpu_vram_mb();
            for (dev, v) in vrams.iter().enumerate() {
                info!("Auto-tier: GPU {} ({} MB VRAM) → {:?}", dev, v, keryx_miner::models::tier_for_vram(*v));
            }
            let mut ts: Vec<Tier> = vrams.iter().map(|&v| keryx_miner::models::tier_for_vram(v)).collect();
            if ts.is_empty() {
                warn!("No per-GPU VRAM info — defaulting to the Dolphin-8B tier.");
                ts.push(Tier::Default);
            }
            ts.sort();
            ts.dedup();
            ts
        }
    };
    let max_tier = tiers.iter().copied().max().unwrap_or(Tier::Default);
    // Warn if GPU power limit / VRAM is below the safe threshold for the heaviest selected tier.
    check_gpu_power_limit(max_tier >= Tier::High, max_tier >= Tier::VeryHigh);
    // DD legacy --cpu-inference: kept so existing launch configs keep working. Ineffective
    // under PoM (inference must stay GPU-resident for zero-dup with the mined weights).
    keryx_miner::slm::set_cpu_inference(opt.cpu_inference);
    if opt.cpu_inference {
        warn!("--cpu-inference is deprecated and ineffective under PoM (inference is GPU-only); ignoring for the PoM lineup.");
    }
    // Stage BOTH lineups for this tier, each filtered by what this GPU can serve, so the
    // chain crossing OPOI_V2_ACTIVATION_DAA hot-swaps without a restart.
    let specs_v1 = union_lineup(0, &tiers);
    let specs_v2 = union_lineup(keryx_miner::models::OPOI_V2_ACTIVATION_DAA, &tiers);
    // PoM candidate models (those with a pinned R_T) staged in this lineup. Captured before
    // specs_v2 is consumed, so each GPU can be assigned the best-fitting one by VRAM below.
    let pom_candidates: Vec<&'static keryx_miner::models::ModelSpec> =
        if keryx_miner::pom::POM_ACTIVATION_DAA != u64::MAX {
            specs_v2
                .iter()
                .copied()
                .filter(|s| keryx_miner::models::is_pom_model(&s.model_id))
                .collect()
        } else {
            Vec::new()
        };
    // Serve the uncensored lineup FROM THE START (as upstream v0.3.4 does): mainnet is permanently
    // past OPOI_V2_ACTIVATION_DAA, so booting on the legacy lineup only creates a broken window on
    // fresh installs — legacy files are never downloaded (lazy), so the initial declare is empty and
    // every bridge challenge loops "model not ready — sending empty response" until a daa-carrying
    // notify triggers the swap (which some bridges never send). Starting on v2 declares the real
    // mining model immediately; set_v2_lineup keeps the crossing swap a consistent no-op (v2 → v2).
    keryx_miner::slm::set_v2_lineup(specs_v2);
    keryx_miner::slm::init_supported(specs_v2);
    let _ = specs_v1; // legacy lineup dropped post-fork (never downloaded, never served)
    info!(
        "OPoI Phase-3 active — {} uncensored model(s) staged (legacy lineup dropped, post-fork).",
        specs_v2.len(),
    );
    info!("Prefetching model files before mining starts…");
    match tokio::task::spawn_blocking(move || keryx_miner::slm::prefetch_models(specs_v2)).await {
        Ok(Ok(())) => info!("Model files ready — starting mining."),
        Ok(Err(e)) => {
            error!("OPoI v2 prefetch failed — refusing to mine without the post-hardfork lineup: {}", e);
            return Err(e.into());
        }
        Err(e) => {
            error!("Model prefetch task panicked: {}", e);
            return Err(e.into());
        }
    }
    // PoM possession setup is LAZY: the possession index + GPU walk are built by the mining
    // loop the first time PoM is active. Here we only record cheap config.
    // Per-GPU PoM assignment: each GPU mines the highest staged PoM tier whose model fits its VRAM.
    // On a uniform rig every GPU resolves to the same model (mining_tier falls back to the single
    // assignment). For a mixed rig (e.g. 4090 + 3060Ti) stage every tier you want a smaller GPU to
    // run, so each device picks the best it can fit.
    if !pom_candidates.is_empty() {
        keryx_miner::slm::set_pom_force_split(true);
        let pick = |vram_mb: u64| -> &'static keryx_miner::models::ModelSpec {
            pom_candidates
                .iter()
                .copied()
                .filter(|s| s.min_vram_mb <= vram_mb)
                .max_by_key(|s| s.min_vram_mb)
                .or_else(|| pom_candidates.iter().copied().min_by_key(|s| s.min_vram_mb))
                .expect("pom_candidates is non-empty")
        };
        let vram = per_gpu_vram_mb();
        if vram.is_empty() {
            // No per-GPU VRAM info: assign device 0 the highest staged tier; other GPUs fall back to it.
            let spec = pom_candidates.iter().copied().max_by_key(|s| s.min_vram_mb).unwrap();
            keryx_miner::pom_gpu::set_mining_tier(
                0,
                spec.model_id,
                keryx_miner::slm::gguf_path_for(spec).to_string_lossy().into_owned(),
            );
            // Tier index is daa-dependent (H2) → computed per block, not shown here.
            info!("PoM: no per-GPU VRAM info — all GPUs mine {}.", spec.dir_name);
        } else {
            for (dev, &vram_mb) in vram.iter().enumerate() {
                let spec = pick(vram_mb);
                keryx_miner::pom_gpu::set_mining_tier(
                    dev as u32,
                    spec.model_id,
                    keryx_miner::slm::gguf_path_for(spec).to_string_lossy().into_owned(),
                );
                info!(
                    "PoM: GPU {} ({} MB VRAM) → {}; index + walk load lazily at activation (DAA {}).",
                    dev, vram_mb, spec.dir_name, keryx_miner::pom::POM_ACTIVATION_DAA
                );
            }
        }
    }
    // Verify GPU inference works before mining. OPoI challenges are mandatory, so a miner
    // that cannot run inference must fail fast with a clear message rather than spam panics.
    if opt.cpu_inference {
        info!("--cpu-inference: skipping GPU inference probe (inference runs on CPU).");
    } else {
    info!("Probing GPU inference (cuBLAS) before mining…");
    match tokio::task::spawn_blocking(keryx_miner::slm::probe_gpu_inference).await {
        Ok(keryx_miner::slm::GpuProbe::Ok) => info!("GPU inference verified — cuBLAS loaded successfully."),
        Ok(keryx_miner::slm::GpuProbe::NoCuda) => {
            warn!("No CUDA device detected — inference will run on CPU (small models only, slow).");
        }
        Ok(keryx_miner::slm::GpuProbe::CublasMissing) => {
            warn!("CUDA GPU detected but a CUDA runtime lib is missing — installing them automatically (one-time)…");
            #[cfg(target_os = "linux")]
            {
                let installed = tokio::task::spawn_blocking(install_cuda_libs).await.unwrap_or(false);
                if !installed {
                    error!("Automatic CUDA lib install failed — install them manually then restart:");
                    error!("  apt-get install -y libcublas-12-2 libcurand-12-2");
                    return Err("CUDA runtime libs missing — cannot start OPoI mining".into());
                }
                // Re-probe in-process. The dynamic loader may still hold a stale cache, so if
                // the freshly-installed libs aren't picked up here, exit cleanly and let the
                // supervisor (HiveOS/PM2) relaunch us with a fresh loader cache.
                match tokio::task::spawn_blocking(keryx_miner::slm::probe_gpu_inference).await {
                    Ok(keryx_miner::slm::GpuProbe::Ok) => {
                        info!("CUDA libs installed — GPU inference verified, starting mining.");
                    }
                    _ => {
                        info!("CUDA libs installed successfully — restarting miner to activate them.");
                        std::process::exit(0);
                    }
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                error!("CUDA GPU detected but a CUDA runtime lib failed to load — install the CUDA 12.6 toolkit and restart.");
                return Err("CUDA runtime libs missing — cannot start OPoI mining".into());
            }
        }
        Err(e) => {
            error!("GPU probe task panicked: {}", e);
            return Err(e.into());
        }
    }
    }
    info!("Found plugins: {:?}", plugins);
    info!("Plugins found {} workers", worker_count);
    if worker_count == 0 && opt.num_threads.unwrap_or(0) == 0 {
        error!("No workers specified");
        return Err("No workers specified".into());
    }

    let block_template_ctr = Arc::new(AtomicU16::new((thread_rng().next_u64() % 10_000u64) as u16));
    if opt.devfund_percent > 0 {
        info!(
            "devfund enabled, mining {}.{}% of the time to devfund address: {} ",
            opt.devfund_percent / 100,
            opt.devfund_percent % 100,
            opt.devfund_address
        );
    }
    // CPU RandomX: arranca el minado CPU en paralelo al GPU de Keryx. Corre en su propia tarea y se
    // pausa durante la inferencia OPoI en CPU. `--cpu "<coin> <wallet> <host>[:port]"` permite
    // cualquier pool; `--xmr/--qrl` (legacy) usan el pool Multipool DDMiner por defecto.
    use keryx_miner::cpu_randomx::RxCoin;
    if let Some(spec) = opt.cpu.clone() {
        match parse_cpu_spec(&spec) {
            Ok((coin, wallet, host, port)) => {
                keryx_miner::cpu_randomx::spawn(
                    coin, wallet, host, port, cpu_worker_name(), opt.cpu_threads, opt.cpu_percent,
                );
            }
            Err(e) => {
                error!("--cpu inválido: {e}. Formato: --cpu \"<xmr|qrl> <wallet> <host>[:port]\"");
                return Err(e.into());
            }
        }
    } else if let Some(addr) = opt.xmr.clone() {
        keryx_miner::cpu_randomx::spawn(
            RxCoin::Xmr, addr, None, None, cpu_worker_name(), opt.cpu_threads, opt.cpu_percent,
        );
    } else if let Some(addr) = opt.qrl.clone() {
        keryx_miner::cpu_randomx::spawn(
            RxCoin::Qrl, addr, None, None, cpu_worker_name(), opt.cpu_threads, opt.cpu_percent,
        );
    }

    let mut fails: u32 = 0;
    loop {
        match client_main(&opt, block_template_ctr.clone(), &plugin_manager, escrow_privkey.clone()).await {
            Ok(_) => {
                fails = 0;
                info!("Client closed gracefully");
            }
            Err(e) => {
                fails += 1;
                // Mensaje CORTO (unas palabras), no el log entero. Detalle real solo con -d.
                match fails {
                    1..=2 => error!("Pool connection dropped — retrying. (run with -d for details)"),
                    3 => error!("Pool connection keeps dropping — check your -a address / GPU. Retrying quietly. (-d for details)"),
                    _ => {}
                }
                log::debug!("Client closed with error {:?}", e);
            }
        }
        // Backoff creciente con tope (evita el spam de miles de reconexiones por segundo).
        let secs = if fails == 0 { 1 } else { std::cmp::min(3 * fails as u64, 30) };
        tokio::time::sleep(Duration::from_secs(secs)).await;
    }
}
