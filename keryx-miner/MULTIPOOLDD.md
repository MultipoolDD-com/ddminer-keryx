# Keryx miner — fork optimizado Multipool DDMiner

Fork de [Keryx-Labs/keryx-miner](https://github.com/Keryx-Labs/keryx-miner) (kHeavyHash + OPoI)
con optimizaciones y fixes de Multipool DDMiner. Mina **Keryx (KRX)** en GPU NVIDIA.

## Cambios respecto al upstream

### 1. Kernel kHeavyHash optimizado — **+31%** (1576 → 2068 MH/s en RTX 4090, bit-exacto)
- **Keccak-f1600 residente en registros** (`plugins/cuda/kaspa-cuda-native/src/keccak-fast.c`):
  25 lanes, rho/pi explícito sin lookups de byte ni arrays en local memory. Reemplaza el
  `keccak-tiny.c` del upstream. Baja de 80 → 72 registros.
- **Ocupancia**: el `suggested_launch_configuration` del upstream elegía block 512 (33% ocupancia).
  El óptimo es block 384 + workload grande (+21%).
- PTX optimizado ya generado en `plugins/cuda/resources/keryx-cuda-sm*.ptx` (sm_61/75/80/86/89).
- Kernel optimizado: `kaspa-cuda-opt.cu` (= stock con el include de keccak cambiado).
- Verificación bit-exacta: `bench.cu` (modo `verify` compara hashes de nonces fijos stock vs opt;
  modo benchmark mide MH/s; flags `-DBLK=N -DGM=N` para barrer launch config).

### 2. Compatibilidad con el bridge stratum de Multipool DDMiner (`src/client/stratum.rs`)
- **`set_extranonce` estilo EthereumStratum/NiceHash**: el bridge manda `set_extranonce(["00d8"])`
  sin `nonce_size`; se deriva de la longitud del extranonce. Sin esto: "Unexpected stratum message".

### 3. Devfund del upstream desactivado (`src/cli.rs`)
- El upstream fuerza un 2% mínimo a la wallet de Keryx-Labs. Desactivado (`devfund_percent = 0`):
  100% de las recompensas a la wallet del operador. (El dev fee de Multipool DDMiner se gestiona en
  `../coins.json`, como el resto de monedas.)
- **Fix crítico asociado**: con `devfund_percent=0` la lógica de time-slice user↔dev del
  `listen()` entraba en reconnect-storm cuando el contador de bloques ciclaba a 0. El early-return
  ahora está guardado a que el devfund esté habilitado.

## Build (CUDA 12.6, gcc 13 OK)

```bash
CUDA_COMPUTE_CAP=89 CUDA_PATH=/usr/local/cuda-12.6 PROTOC=/usr/bin/protoc \
  cargo build --release            # binario + plugins/cuda (libkeryxcuda.so)
```
Recompilar el PTX optimizado tras tocar el kernel:
```bash
cd plugins/cuda/kaspa-cuda-native/src
nvcc -O3 -arch=sm_89 -ptx kaspa-cuda-opt.cu -o ../../resources/keryx-cuda-sm89.ptx
```

## Ejecutar

```bash
LD_LIBRARY_PATH=.:/ruta/cuda-12.6/targets/x86_64-linux/lib ./keryx-miner \
  -a keryx:TU_WALLET -s stratum+tcp://multipooldd.com:5555 --light
```
- `--light` = TinyLlama, inferencia OPoI en GPU (rápida, gana carreras de escrow).
- `--cpu-inference` = inferencia en CPU (no pausa el PoW; usar solo con muchísimos AiRequests).
- Multi-GPU automático (`--cuda-device 0,1` para elegir).

## Bundle autocontenido (para desplegar sin CUDA del sistema)

Las librerías CUDA (`libcublas/cublasLt/curand/cudart`) NO se versionan en git (≈725 MB).
Para un bundle portable: copiar el binario + `libkeryxcuda.so` + esas libs + symlinks sin
versión (`libcublas.so → libcublas.so.12`, etc.) en una carpeta, con un `start.sh` que fije
`LD_LIBRARY_PATH` a esa carpeta. El binario se compila con `RPATH=$ORIGIN`. Requisito en destino:
solo el driver NVIDIA (`libcuda.so.1`). El modelo TinyLlama y el daemon IPFS se auto-descargan
en la primera ejecución.

## OPoI (escrow 20%)
El minero declara capacidades (`mining.declare_capabilities`), responde a `mining.challenge`
con inferencia real de TinyLlama (~1s en GPU) y sube el AiResponse a IPFS (kubo auto-instalado,
puerto 5001), devolviendo el CID en `mining.submit`. Sin OPoI el bridge no entrega trabajo.

Licencia upstream: MIT/Apache (ver `LICENSE-MIT` / `LICENSE-APACHE`).
