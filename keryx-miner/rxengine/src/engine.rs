//! Motor de minado RandomX: pool de threads con afinidad, pausa cooperativa
//! (para cederle la CPU a la inferencia OPoI) y detección de shares.
//!
//! Diseñado para portarse al ddminer: el `RxEngine` es autónomo y se controla
//! con métodos simples (start/stop/pause/resume/set_job/rebuild_for_seed).

use crate::job::RxJob;
use randomx_rs::{RandomXCache, RandomXDataset, RandomXFlag, RandomXVM};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, RwLock};
use std::thread::JoinHandle;
use std::time::Duration;

/// El dataset de RandomX es read-only durante el minado; compartir su puntero
/// entre threads para lectura es seguro (mismo modelo que usa xmrig). El crate
/// randomx-rs no marca Send por el raw pointer interno, así que lo envolvemos.
struct ShareableDataset(RandomXDataset);
unsafe impl Send for ShareableDataset {}
unsafe impl Sync for ShareableDataset {}

/// Un share encontrado por un worker.
#[derive(Clone, Debug)]
pub struct Share {
    pub job_id: String,
    pub nonce: u32,
    pub hash: [u8; 32],
}

/// Cuántos nonces toma cada worker por lote del dispatcher global.
const NONCE_BATCH: u64 = 256;

/// Estado compartido entre el engine y sus workers (todo Arc, se clona barato).
#[derive(Clone)]
struct Shared {
    job: Arc<RwLock<Option<Arc<RxJob>>>>,
    job_epoch: Arc<AtomicU64>,
    nonce_dispatch: Arc<AtomicU64>,
    paused: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    hashes: Arc<AtomicU64>,
    shares_tx: Sender<Share>,
}

pub struct RxEngine {
    flags: RandomXFlag,
    seed_hash: Vec<u8>,
    dataset: RandomXDataset,
    shared: Shared,
    handles: Vec<JoinHandle<()>>,
    num_threads: usize,
}

impl RxEngine {
    /// Crea el motor para un `seed_hash` dado, construyendo cache + dataset (fast
    /// mode, ~2 GB) y arrancando un worker pineado a cada core de `core_ids`
    /// (su longitud = nº de threads de minado, p.ej. ~70% de los cores físicos).
    /// `paused` es un flag de pausa INYECTABLE: el orquestador (o el ddminer)
    /// lo conserva para poder pausar el minado RandomX durante la inferencia
    /// OPoI en CPU. Pásalo en `false` para arrancar minando.
    pub fn new(
        seed_hash: Vec<u8>,
        core_ids: Vec<usize>,
        paused: Arc<AtomicBool>,
    ) -> Result<(Self, Receiver<Share>), String> {
        let flags = RandomXFlag::get_recommended_flags() | RandomXFlag::FLAG_FULL_MEM;
        let cache = RandomXCache::new(flags, &seed_hash).map_err(|e| format!("cache: {e}"))?;
        let dataset = build_dataset(flags, cache)?;
        let (shares_tx, shares_rx) = std::sync::mpsc::channel();
        let shared = Shared {
            job: Arc::new(RwLock::new(None)),
            job_epoch: Arc::new(AtomicU64::new(0)),
            nonce_dispatch: Arc::new(AtomicU64::new(0)),
            paused,
            running: Arc::new(AtomicBool::new(true)),
            hashes: Arc::new(AtomicU64::new(0)),
            shares_tx,
        };
        let handles = spawn_workers(flags, &dataset, &shared, &core_ids);
        Ok((
            RxEngine {
                flags,
                seed_hash,
                dataset,
                shared,
                handles,
                num_threads: core_ids.len(),
            },
            shares_rx,
        ))
    }

    /// Instala un nuevo job y resetea el dispatcher de nonces. Si el seed_hash
    /// cambió, llamar antes a `rebuild_for_seed`.
    pub fn set_job(&self, job: RxJob) {
        *self.shared.job.write().unwrap() = Some(Arc::new(job));
        self.shared.nonce_dispatch.store(0, Ordering::SeqCst);
        self.shared.job_epoch.fetch_add(1, Ordering::SeqCst);
    }

    /// Pausa cooperativa: los workers dejan de hashear pero conservan VM y
    /// dataset (resume instantáneo). Para cederle la CPU a la inferencia OPoI.
    pub fn pause(&self) {
        self.shared.paused.store(true, Ordering::SeqCst);
    }
    pub fn resume(&self) {
        self.shared.paused.store(false, Ordering::SeqCst);
    }
    pub fn is_paused(&self) -> bool {
        self.shared.paused.load(Ordering::SeqCst)
    }

    /// Hashes calculados desde la última lectura (resetea el contador).
    pub fn take_hashes(&self) -> u64 {
        self.shared.hashes.swap(0, Ordering::AcqRel)
    }

    pub fn seed_hash(&self) -> &[u8] {
        &self.seed_hash
    }
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Reconstruye cache + dataset para un nuevo seed (cambio de época) y relanza
    /// los workers. Operación cara (segundos): solo cuando el pool cambia el
    /// seed_hash. No-op si el seed no cambió.
    pub fn rebuild_for_seed(&mut self, new_seed: Vec<u8>, core_ids: Vec<usize>) -> Result<(), String> {
        if new_seed == self.seed_hash {
            return Ok(());
        }
        self.stop();
        let cache = RandomXCache::new(self.flags, &new_seed).map_err(|e| format!("cache: {e}"))?;
        self.dataset = build_dataset(self.flags, cache)?;
        self.seed_hash = new_seed;
        self.shared.running.store(true, Ordering::SeqCst);
        self.shared.nonce_dispatch.store(0, Ordering::SeqCst);
        self.handles = spawn_workers(self.flags, &self.dataset, &self.shared, &core_ids);
        self.num_threads = core_ids.len();
        Ok(())
    }

    /// Detiene todos los workers y espera a que terminen.
    pub fn stop(&mut self) {
        self.shared.running.store(false, Ordering::SeqCst);
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

impl Drop for RxEngine {
    fn drop(&mut self) {
        self.stop();
    }
}

fn build_dataset(flags: RandomXFlag, cache: RandomXCache) -> Result<RandomXDataset, String> {
    // El init del dataset es paralelo internamente y solo ocurre al arrancar o
    // al cambiar de época.
    RandomXDataset::new(flags, cache, 0).map_err(|e| format!("dataset: {e}"))
}

/// Lanza un worker por cada core en `core_ids`, pineado por afinidad.
fn spawn_workers(
    flags: RandomXFlag,
    dataset: &RandomXDataset,
    shared: &Shared,
    core_ids: &[usize],
) -> Vec<JoinHandle<()>> {
    let mut handles = Vec::with_capacity(core_ids.len());
    for &core in core_ids {
        let ds = ShareableDataset(dataset.clone());
        let shared = shared.clone();
        let handle = std::thread::Builder::new()
            .name(format!("rx-worker-{core}"))
            .spawn(move || {
                if let Some(cid) = core_affinity::get_core_ids()
                    .and_then(|ids| ids.into_iter().find(|c| c.id == core))
                {
                    core_affinity::set_for_current(cid);
                }
                worker_loop(flags, ds, shared);
            })
            .expect("spawn worker");
        handles.push(handle);
    }
    handles
}

fn worker_loop(flags: RandomXFlag, dataset: ShareableDataset, shared: Shared) {
    // Cada thread crea su propia VM enlazada al dataset compartido.
    let vm = match RandomXVM::new(flags, None, Some(dataset.0.clone())) {
        Ok(vm) => vm,
        Err(e) => {
            eprintln!("rx-worker: no pude crear VM: {e}");
            return;
        }
    };

    let mut local_epoch = u64::MAX;
    let mut blob: Vec<u8> = Vec::new();
    let mut nonce_offset = 0usize;
    let mut target = [0u8; 32];
    let mut job_id = String::new();

    while shared.running.load(Ordering::Relaxed) {
        if shared.paused.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }

        // ¿Job nuevo? Recargamos blob/target a buffers locales.
        let epoch = shared.job_epoch.load(Ordering::Relaxed);
        if epoch != local_epoch {
            match shared.job.read().unwrap().as_ref() {
                Some(j) => {
                    blob = j.blob.clone();
                    nonce_offset = j.nonce_offset;
                    target = j.target;
                    job_id = j.job_id.clone();
                    local_epoch = epoch;
                }
                None => {
                    std::thread::sleep(Duration::from_millis(5));
                    continue;
                }
            }
        }
        if blob.len() < nonce_offset + 4 {
            std::thread::sleep(Duration::from_millis(5));
            continue;
        }

        // Lote de nonces del dispatcher global (sin colisiones entre threads).
        let start = shared.nonce_dispatch.fetch_add(NONCE_BATCH, Ordering::Relaxed);
        for i in 0..NONCE_BATCH {
            let nonce = (start + i) as u32;
            blob[nonce_offset..nonce_offset + 4].copy_from_slice(&nonce.to_le_bytes());
            if let Ok(h) = vm.calculate_hash(&blob) {
                shared.hashes.fetch_add(1, Ordering::Relaxed);
                if h.len() == 32 && RxJob::meets_target(&h, &target) {
                    let mut hash = [0u8; 32];
                    hash.copy_from_slice(&h);
                    let _ = shared.shares_tx.send(Share {
                        job_id: job_id.clone(),
                        nonce,
                        hash,
                    });
                }
            }
            // Salida temprana del lote si llega job nuevo, pausa o parada.
            if i % 64 == 0
                && (shared.job_epoch.load(Ordering::Relaxed) != local_epoch
                    || shared.paused.load(Ordering::Relaxed)
                    || !shared.running.load(Ordering::Relaxed))
            {
                break;
            }
        }
    }
}
