#!/usr/bin/env bash
# ddminer-dual: dual-mining Pearl (GPU) + Keryx (CPU) en el mismo rig.
# Pearl satura la GPU; Keryx kHeavyHash corre en los cores ociosos de la CPU.
# Procesos separados + afinidad de cores (taskset) para que no se peleen.
#
#   Pearl  -> GPU (tensor/dp4a)  + unos pocos cores para su orquestacion
#   Keryx  -> CPU (keccak asm), --no-opoi (sin inferencia, PoW puro)
#
# Config por variables de entorno (rellena las tuyas):
set -euo pipefail

# --- Pearl (ddminer) ---
: "${PEARL_CMD:=/home/threadripper/pearl-miner-ubuntu22/pearlminer}"   # comando del minero Pearl
: "${PEARL_CORES:=0-15}"                                                # cores para orquestacion Pearl

# --- Keryx ---
: "${KERYX_BIN:=/home/threadripper/keryx-miner/target-cuda/release/keryx-miner}"
: "${KERYX_ADDR:?define KERYX_ADDR=keryx:tu_direccion}"                 # OBLIGATORIO
: "${KERYX_POOL:=}"                                                     # opcional: --pool host:port via flags propios
: "${KERYX_CORES:=16-127}"                                             # cores dedicados a Keryx CPU
: "${KERYX_THREADS:=112}"                                              # nº de hilos = nº de cores del rango de arriba
: "${DEVFUND:=2}"

cleanup(){ echo "[dual] parando..."; kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "=============================================="
echo " ddminer-dual  |  Pearl(GPU) + Keryx(CPU)"
echo "  Pearl cores : $PEARL_CORES"
echo "  Keryx cores : $KERYX_CORES  (threads=$KERYX_THREADS)"
echo "=============================================="

# 1) Pearl en la GPU (cores de orquestacion acotados)
echo "[dual] arrancando Pearl (GPU)..."
taskset -c "$PEARL_CORES" $PEARL_CMD &
PEARL_PID=$!

# 2) Keryx en CPU: --no-opoi (PoW puro, sin inferencia ni GPU), afinidad a cores libres
echo "[dual] arrancando Keryx (CPU, $KERYX_THREADS hilos)..."
taskset -c "$KERYX_CORES" "$KERYX_BIN" \
    --mining-address "$KERYX_ADDR" \
    --threads "$KERYX_THREADS" \
    --no-opoi \
    --devfund-percent "$DEVFUND" \
    ${KERYX_POOL:+$KERYX_POOL} &
KERYX_PID=$!

# 3) Supervisar: si uno muere, avisa (no mata al otro — mineria independiente)
while true; do
    if ! kill -0 $PEARL_PID 2>/dev/null; then echo "[dual] ⚠ Pearl murio (pid $PEARL_PID)"; PEARL_PID=0; fi
    if ! kill -0 $KERYX_PID 2>/dev/null; then echo "[dual] ⚠ Keryx murio (pid $KERYX_PID)"; KERYX_PID=0; fi
    [ "$PEARL_PID" = 0 ] && [ "$KERYX_PID" = 0 ] && { echo "[dual] ambos pararon, salgo"; break; }
    sleep 10
done
