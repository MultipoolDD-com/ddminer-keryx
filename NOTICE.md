# Third-party notices and attribution

DDMiner Keryx is made by Multipool DDMiner. It is a fork of the upstream Keryx miner and builds on third-party open-source work. This notice credits those projects.

## Keryx miner (upstream)

This is a fork of [Keryx-Labs/keryx-miner](https://github.com/Keryx-Labs/keryx-miner), the reference Keryx GPU miner (kHeavyHash proof of work plus OPoI inference). The upstream project is itself derived from the Kaspa miner family (kHeavyHash). Upstream is licensed under MIT or Apache 2.0, and this fork keeps the same license (see `LICENSE-MIT` and `LICENSE-APACHE`).

## Multipool DDMiner changes

The Multipool DDMiner fork adds:

* An optimized kHeavyHash CUDA kernel (about 31% faster on an RTX 4090), bit exact with the stock kernel.
* Stratum compatibility with the Multipool DDMiner Keryx pool bridge (NiceHash style extranonce handling).
* The upstream dev fund disabled, so 100% of block rewards go to the operator wallet supplied with `-a`.

## Bundled components

* The GPU mining plugin (`libkeryxcuda.so`) depends only on the NVIDIA driver (`libcuda.so.1`).
* The FULL bundle also ships the NVIDIA cuBLAS, cuBLASLt, cuRAND, and cudart runtime libraries, used only for GPU inference. These are redistributed under the NVIDIA CUDA toolkit end user license agreement.
* On first run the miner downloads the TinyLlama model and starts a bundled IPFS (kubo) daemon for uploading inference results.

## Dev fee

This miner takes no dev fee. 100% of block rewards go to the wallet you pass with `-a`. The pool charges its own fee, disclosed on the pool site.
