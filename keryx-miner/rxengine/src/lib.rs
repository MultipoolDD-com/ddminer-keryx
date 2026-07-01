//! rxengine — motor de minado RandomX (XMR/QRL) portable al ddminer.
//!
//! - [`job`]: el trabajo de minado y la lógica de target (256-bit LE).
//! - [`engine`]: pool de threads con afinidad, pausa cooperativa y shares.

pub mod engine;
pub mod job;
pub mod orchestrator;
pub mod stratum;

pub use engine::{RxEngine, Share};
pub use job::{target_from_difficulty, RxJob};
pub use orchestrator::{run_miner, MinerConfig};
pub use stratum::{PoolEvent, StratumConfig};
