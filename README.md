# DDMiner — Keryx (KRX)

GPU miner for **Keryx (KRX)** by **Multipool DDMiner**. It mines the Keryx
proof-of-work — **kHeavyHash → Proof-of-Model (PoM)** after the hardfork — and
serves the OPoI inference lineup. It connects to **any Keryx stratum pool**
(defaults to the Multipool DDMiner pool if you don't pass one).

```
GPU (KRX):  stratum+tcp://<any pool>:<port>   default: multipooldd.com:5555
CPU (opt):  RandomX (XMR/QRL) to ANY pool     opt-in with --cpu
Address:    your own KRX payout address (-a)
Fee:        0% on GPU Keryx mining. CPU RandomX (--cpu) carries a 1% dev fee (time-share).
```

## Quick start

Pick a bundle from [Releases](https://github.com/MultiPoolDD/ddminer-keryx/releases), extract, and run:

```bash
tar -xzf ddminer-keryx-0.5.2-pom-linux-x64.tar.gz
cd ddminer-pom

# no flags = AUTO: each GPU mines the highest tier its VRAM can fit
./start.sh -a keryx:YOUR_ADDRESS

# or connect to a specific pool
./start.sh -a keryx:YOUR_ADDRESS -s stratum+tcp://your-pool:5555
```

On first run it downloads the model lineup into `./models` (via a bundled IPFS
daemon) and, once PoM is active, builds the possession index. Models are **not**
shipped in the bundle — they download automatically.

## Which bundle?

| Bundle | Size | Use it if… |
|---|---|---|
| `ddminer-keryx-0.5.2-pom-linux-x64.tar.gz` | ~10 MB | You have the **CUDA toolkit** installed (libcurand/libcublas). |
| `ddminer-keryx-0.5.2-pom-full-linux-x64.tar.gz` | ~475 MB | You do **not** have CUDA installed — it bundles the CUDA libraries and runs on just the NVIDIA driver. |
| `keryx-miner-0.5.2-hiveos.tgz` | ~475 MB | **HiveOS** — self-contained custom miner (see below). |

All contain the same `ddminer` binary and mine at the same speed. The `full`
and `hiveos` bundles add the CUDA runtime libraries (`libcurand`, `libcublas`,
`libcublasLt`) so you don't need the CUDA toolkit. If you hit
`libcurand.so.10: cannot open shared object file`, use the **full** bundle.

## Model tiers (auto or forced)

**No flag → AUTO:** the miner detects each GPU's VRAM and mines the highest tier
it fits. A mixed rig (e.g. RTX 4090 + RTX 3060 Ti) runs a different model per GPU,
each at its max. Force a tier if you prefer:

| Flag | Model | Min VRAM |
|---|---|---|
| `--very-light` | Qwen3-1.7B | 4 GB (PoM tier 0 post-H2) |
| `--light` | Gemma-3-4B | any GPU |
| `--default` | Dolphin-3.0-Llama-3.1-8B | 8 GB (RTX 3060/3070) |
| `--high` | Qwen3-32B (Q4_K_M) | 24 GB (RTX 3090/4090/5090) |
| `--very-high` | Llama-3.3-70B (Q4 pre-H2 → Q2_K_L post-H2) | 48 GB Q4 / 32 GB Q2 (RTX 5090) |

Under PoM each GPU mines **and** proves possession of exactly one model.

## Any pool

Pass `-s stratum+tcp://host:port` to mine to any Keryx pool. Omit it to use the
default Multipool DDMiner pool. For solo mining against a local `keryxd`, use
`-s grpc://<keryxd>` (required for solo Proof-of-Model).

## Mine XMR or QRL on the CPU (optional)

RandomX on otherwise-idle CPU cores, in parallel with GPU Keryx mining, to **any**
RandomX pool:

```bash
./start.sh -a keryx:YOUR_KRX_ADDRESS --cpu "xmr YOUR_XMR_ADDRESS pool.supportxmr.com:3333"
./start.sh -a keryx:YOUR_KRX_ADDRESS --cpu "qrl YOUR_QRL_ADDRESS qrlpool.com:3333"
```

Format: `--cpu "<coin> <wallet> <host>[:port]"` (coin = `xmr` | `qrl`). It pauses
during OPoI inference so it never fights for memory bandwidth. A **1% dev fee**
applies via time-share (mines to the dev wallet ~1% of the time on the same pool).
Tune with `--cpu-percent N` (default 70) or `--cpu-threads N`. The legacy
`--xmr`/`--qrl` flags still work (default pool).

## HiveOS

Use `keryx-miner-0.5.2-hiveos.tgz` (self-contained — HiveOS ships only the NVIDIA
driver, not the CUDA toolkit). In the Flight Sheet → Miner: **Custom**:

| Field | Value |
|---|---|
| Miner name | `keryx-miner` |
| Installation URL | the URL of `keryx-miner-0.5.2-hiveos.tgz` from Releases |
| Wallet and worker template | `%WAL%` |
| Pool URL | `stratum+tcp://multipooldd.com:5555` (or your pool) |
| Pass | `x` |
| Extra config arguments | *(empty = auto-tier)* or `--high`, `--very-high`, `--cpu "..."` |

HiveOS shows per-GPU hashrate, total, temp/fan, and accepted/rejected shares.

## GPU support

Turing (RTX 20xx, sm_75) · Ampere (RTX 30xx) · Ada (RTX 40xx / RTX Ada) ·
Blackwell (RTX 50xx, sm_120, native PTX) and datacenter Blackwell. The mining
kernel ships PTX for sm_61/75/80/86/89/100/120 and JIT-adapts to your GPU.

> **Note:** PoM inference requires bf16 tensor cores (sm_80+). Turing (RTX 20xx,
> sm_75) can run the mining kernel but **not** PoM inference, so it cannot mine
> Keryx after the PoM hardfork.

## Common options

| Flag | Meaning |
|---|---|
| `-a, --mining-address keryx:...` | your KRX payout address (required) |
| `-s, --keryxd-address` | pool URL `stratum+tcp://host:port` (or `grpc://` for solo) |
| *(no tier flag)* | auto — each GPU mines its max tier |
| `--very-light` / `--light` / `--default` / `--high` / `--very-high` | force a tier |
| `--cpu "<coin> <wallet> <host>[:port]"` | also mine XMR/QRL on the CPU to any pool |
| `--cpu-percent N` / `--cpu-threads N` | CPU cores for RandomX (default 70%) |
| `-d` | verbose debug logging |

Run `./ddminer --help` for the full list.

## Requirements

* NVIDIA GPU + NVIDIA driver (`libcuda.so.1`).
* Linux x64, glibc 2.39+ (Ubuntu 24.04+ / recent distros).
* CUDA runtime libraries (`libcurand`, `libcublas`, `libcublasLt`) — **bundled**
  in the `full` and `hiveos` packages; on the small bundle you need the CUDA
  toolkit installed.
* Disk for the model lineup (downloaded on first run) + RAM for the RandomX
  dataset if using `--cpu` (~2.5 GB; huge pages raise hashrate 10–30%).

## Your escrow key

On first run the miner generates an OPoI escrow keypair as `escrow.key` next to
the binary. It is yours and tied to your escrow rewards — keep it, back it up,
never share it. Bundles ship without any escrow key, so every miner gets its own.

## Verify your download

```bash
sha256sum -c SHA256SUMS
```

## Build from source

Source is in [`keryx-miner/`](keryx-miner/): the crate builds `ddminer`, and the
`rxengine` sub-crate is the RandomX (XMR/QRL) miner. The inference engine builds
against CUDA (13.2 tested). The candle inference PTX is pinned to sm_80 so one
binary runs any GPU ≥ sm_80.

```bash
cd keryx-miner
cargo build --release --bin ddminer      # host binary (includes RandomX)
cargo build --release -p keryxcuda        # CUDA plugin (libkeryxcuda.so)
```

Build the binary and the plugin with the same toolchain. Re-apply RPATH
`$ORIGIN` (`patchelf --set-rpath '$ORIGIN' target/release/ddminer`) so co-located
libraries load without `LD_LIBRARY_PATH`. For older distros, build inside a
low-glibc container.

## License and credits

MIT or Apache 2.0 (your choice), a fork of the upstream Keryx miner. See
`LICENSE-MIT`, `LICENSE-APACHE`, and `NOTICE.md`.
