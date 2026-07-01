use std::env;
use time::{format_description, OffsetDateTime};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let format = format_description::parse("[year repr:last_two][month][day][hour][minute]")?;
    let dt = OffsetDateTime::now_utc().format(&format)?;
    println!("cargo:rustc-env=PACKAGE_COMPILE_TIME={}", dt);

    println!("cargo:rerun-if-changed=proto");
    println!("cargo:rerun-if-changed=src/keccakf1600_x86-64.s");
    tonic_build::configure()
        .build_server(false)
        // .type_attribute(".", "#[derive(Debug)]")
        .compile(
            &["proto/rpc.proto", "proto/p2p.proto", "proto/messages.proto"],
            &["proto"],
        )?;
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    if target_arch == "x86_64" && target_os != "windows" && target_os != "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64.s").compile("libkeccak.a");
    }
    if target_arch == "x86_64" && target_os == "macos" {
        cc::Build::new().flag("-c").file("src/keccakf1600_x86-64-osx.s").compile("libkeccak.a");
    }

    // PoM mining kernel → PTX (loaded at runtime into candle's CUDA context).
    // Skipped on non-CUDA builds (e.g. macOS / CI without nvcc). Configurable via env:
    //   NVCC=/path/to/nvcc   SM_ARCH=86   (PTX is JIT-forward-compatible to newer GPUs)
    println!("cargo:rerun-if-changed=cuda/pom_mine.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=SM_ARCH");
    if target_os != "macos" {
        let nvcc = env::var("NVCC").ok().unwrap_or_else(find_nvcc);
        // sm_86 PTX runs on every Ampere+ GPU (30xx/40xx/50xx) via driver JIT.
        // Override with SM_ARCH for older cards (e.g. 75 for Turing).
        let sm = env::var("SM_ARCH").unwrap_or_else(|_| "86".to_string());
        let out_dir = env::var("OUT_DIR").unwrap();
        let ptx = format!("{out_dir}/pom_mine.ptx");
        let status = std::process::Command::new(&nvcc)
            .args(["-ptx", "-O3", &format!("-arch=sm_{sm}"), "cuda/pom_mine.cu", "-o", &ptx])
            .status()
            .unwrap_or_else(|e| panic!("nvcc ({nvcc}) failed to run: {e}. Set NVCC=/path/to/nvcc."));
        assert!(status.success(), "nvcc failed to compile cuda/pom_mine.cu (arch sm_{sm})");
    }
    Ok(())
}

/// Locate an `nvcc` binary: prefer a CUDA 12.x toolkit (candle 0.9 / cudarc target),
/// else fall back to whatever `nvcc` is on PATH.
fn find_nvcc() -> String {
    for dir in [
        "/usr/local/cuda-12.6/bin",
        "/usr/local/cuda-12/bin",
        "/usr/local/cuda/bin",
    ] {
        let p = format!("{dir}/nvcc");
        if std::path::Path::new(&p).exists() {
            return p;
        }
    }
    "nvcc".to_string()
}
