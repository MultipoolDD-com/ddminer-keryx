use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed=src/compatibility.cuh");
    println!("cargo::rerun-if-changed=src/cuda_utils.cuh");
    println!("cargo::rerun-if-changed=src/binary_op_macros.cuh");

    // Portable PTX virtual-arch floor. We ship ONE binary that must run on any
    // miner GPU (GTX 10xx = sm_61, RTX 20 = sm_75, RTX 30 = sm_86, RTX 40 = sm_89,
    // RTX 50 = sm_120…). bindgen_cuda::default() otherwise bakes the *build
    // machine's* arch, which fails with CUDA_ERROR_INVALID_PTX on lower GPUs
    // because a PTX .target is a *minimum* capability. The standard kernel set is
    // arch-guarded (bf16 paths gated by __CUDA_ARCH__ >= 800), so sm_61 compiles
    // clean and the driver JIT-compiles it up to whatever real GPU loads it. The
    // MoE bf16-WMMA static lib below keeps its own sm_80 floor — those kernels
    // need it to COMPILE, and are only launched for MoE models (none in the PoM
    // lineup), so they stay dead code on older GPUs. Override with
    // CUDA_COMPUTE_CAP=<n> for a single-arch optimised build.
    println!("cargo::rerun-if-env-changed=CUDA_COMPUTE_CAP");
    let compute_cap: usize = env::var("CUDA_COMPUTE_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(61);
    let moe_compute_cap = compute_cap.max(80);

    // Build for PTX — ONLY the standard kernel set. The moe/*.cu WMMA kernels need sm_80 to
    // compile, so they must not enter this (sm_61-floor) PTX pass; they are built separately below
    // as a static lib at their own sm_80 floor.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("ptx.rs");
    let builder = bindgen_cuda::Builder::default()
        .kernel_paths(vec![
            "src/affine.cu",
            "src/binary.cu",
            "src/cast.cu",
            "src/conv.cu",
            "src/fill.cu",
            "src/indexing.cu",
            "src/quantized.cu",
            "src/reduce.cu",
            "src/sort.cu",
            "src/ternary.cu",
            "src/unary.cu",
        ])
        .compute_cap(compute_cap)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");
    let bindings = builder.build_ptx().unwrap();
    bindings.write(&ptx_path).unwrap();

    // Remove unwanted MOE PTX constants from ptx.rs
    remove_lines(&ptx_path, &["MOE_GGUF", "MOE_WMMA", "MOE_WMMA_GGUF"]);

    let mut moe_builder = bindgen_cuda::Builder::default()
        .compute_cap(moe_compute_cap)
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3");

    // Build for FFI binding (must use custom bindgen_cuda, which supports simutanously build PTX and lib)
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut is_target_msvc = false;
    if let Ok(target) = std::env::var("TARGET") {
        if target.contains("msvc") {
            is_target_msvc = true;
            moe_builder = moe_builder.arg("-D_USE_MATH_DEFINES");
        }
    }

    if !is_target_msvc {
        moe_builder = moe_builder.arg("-Xcompiler").arg("-fPIC");
    }

    let moe_builder = moe_builder.kernel_paths(vec![
        "src/moe/moe_gguf.cu",
        "src/moe/moe_wmma.cu",
        "src/moe/moe_wmma_gguf.cu",
    ]);
    moe_builder.build_lib(out_dir.join("libmoe.a"));
    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rustc-link-lib=moe");
    println!("cargo:rustc-link-lib=dylib=cudart");
    if !is_target_msvc {
        println!("cargo:rustc-link-lib=stdc++");
    }
}

fn remove_lines<P: AsRef<std::path::Path>>(file: P, patterns: &[&str]) {
    let content = std::fs::read_to_string(&file).unwrap();
    let filtered = content
        .lines()
        .filter(|line| !patterns.iter().any(|p| line.contains(p)))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(file, filtered).unwrap();
}
