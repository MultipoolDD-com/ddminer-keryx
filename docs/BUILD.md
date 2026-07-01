# Building from source

You do not need this to mine. The bundles in this repo and in Releases are prebuilt and self contained. This page is for building the miner yourself.

The full source is in [`keryx-miner/`](../keryx-miner/). The crate builds the `ddminer` binary; the `rxengine` sub-crate is the RandomX (XMR/QRL) CPU miner. It is a Multipool DDMiner build of the Keryx miner (kHeavyHash + OPoI) with these differences from the upstream Keryx-Labs miner:

1. The pool is locked to `multipooldd.com`. The miner refuses any other pool or solo mode.
2. A single status line that refreshes in place (ASCII banner once, then hashrate, per-GPU temperature and fan, accepted/rejected shares, and 0% miner fee). Everything else is quiet unless you pass `-d`.
3. The rig hostname is sent to the pool as the worker name (`address.HOSTNAME`).
4. The upstream dev fund is disabled, so 100% of block rewards go to the operator wallet.
5. Optional CPU mining of Monero (`--xmr`) or QRL (`--qrl`) with RandomX, in parallel with KRX GPU mining, locked to the Multipool DDMiner ports 4334 (XMR) and 4335 (QRL).

## Requirements

* Rust and Cargo ([rustup.rs](https://rustup.rs))
* `protoc` (the `protobuf-compiler` package)
* The CUDA 12.x toolkit for the GPU build. CUDA 12.4 is recommended. Do not use CUDA 13.x, it breaks the cudarc build.
* `cmake`, a C++ compiler, and `git` for the RandomX engine. It builds the upstream RandomX C++ with CMake (`apt-get install cmake build-essential git`).

## Build

Set `CUDA_COMPUTE_CAP` for your GPU generation (RTX 30xx = 86, RTX 40xx = 89, RTX 50xx = 100). Build in two steps; building the binary and the plugin in one `cargo build` invocation fails on recent Cargo.

```bash
cd keryx-miner
export CUDA_ROOT=/usr/local/cuda-12.4 CUDA_PATH=/usr/local/cuda-12.4 PROTOC=/usr/bin/protoc
export CUDA_COMPUTE_CAP=89 PATH=/usr/local/cuda-12.4/bin:$PATH
cargo build --release --bin ddminer      # host binary, includes the RandomX engine
cargo build --release -p keryxcuda        # CUDA GPU mining plugin
```

Outputs:

* `target/release/ddminer` (the binary)
* `target/release/libkeryxcuda.so` (the GPU mining plugin)

Build the binary and the plugin together with the same toolchain, they must match (a mismatch makes the binary abort when it loads the plugin). To make a runnable bundle, place `ddminer` and `libkeryxcuda.so` in one folder with the CUDA runtime libraries (`libcudart.so.12`, and for GPU inference also `libcublas.so.12`, `libcublasLt.so.12`, `libcurand.so.10`) plus a `start.sh` that sets `LD_LIBRARY_PATH` to that folder, exactly like the prebuilt bundles.

## Runtime libraries

Proof of work needs only `libcuda.so.1` (the driver). GPU inference additionally loads `libcublas.so.12` and `libcurand.so.10` at runtime, which the FULL bundle ships. The LITE bundle runs inference on CPU and does not need them.

RandomX mining (`--xmr`/`--qrl`) links `libstdc++.so.6`. This is standard on every Linux install, so the bundles do not ship it. The required symbol version is low (`GLIBCXX_3.4.21`, from GCC 5.1), so any modern distro satisfies it.

## Portable bundles (low glibc)

The glibc floor of a binary is set by the glibc of the machine you compile on. To make bundles that run on older distros, build inside a low-glibc container, for example Ubuntu 20.04 (glibc 2.31). Mount your CUDA 12.4 toolkit into the container and build there. The resulting `ddminer` has a glibc 2.30 floor and needs only `libc`, `libm`, `libgcc_s`, `libstdc++`, `libdl`, and `libpthread` from the host, so it runs on Ubuntu 18.04+, Debian 10+, and Rocky/RHEL 8+.
