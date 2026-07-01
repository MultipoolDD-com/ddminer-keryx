//! Cliente stratum para RandomX (protocolo Monero, el universal de RandomX).
//!
//! Flujo: `login` → respuesta con session-id + primer job → push de `job` →
//! `submit` de shares → `keepalived`. El framing es JSON separado por '\n'.
//!
//! NOTA: el envoltorio de mensajes (nombres de método, params) puede variar
//! según el pool; las funciones puras de parsing (target, job→RxJob, submit)
//! están aisladas y testeadas para poder amoldarlas al dialecto del socio en
//! cuanto su backend RandomX responda.

use crate::job::RxJob;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;
use tokio::sync::mpsc;

/// Offset del nonce dentro del blob en Monero/CryptoNote (bytes 39..43).
pub const MONERO_NONCE_OFFSET: usize = 39;

// ─────────────────────────── Tipos de mensaje ───────────────────────────────

#[derive(Serialize, Debug)]
pub struct LoginParams {
    pub login: String,
    pub pass: String,
    pub agent: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rigid: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub algo: Vec<String>,
}

/// Job tal como lo manda el pool (campos hex).
#[derive(Deserialize, Debug, Clone, Default)]
pub struct StratumJob {
    pub job_id: String,
    pub blob: String,
    pub target: String,
    #[serde(default)]
    pub seed_hash: String,
    #[serde(default)]
    pub height: u64,
    #[serde(default)]
    pub algo: String,
}

#[derive(Serialize, Debug)]
pub struct SubmitParams {
    /// session-id devuelto por el login (¡NO el job_id! error clásico).
    pub id: String,
    pub job_id: String,
    /// nonce de 4 bytes en hex (8 chars).
    pub nonce: String,
    /// hash resultante en hex (64 chars).
    pub result: String,
}

// ─────────────────────────── Funciones puras ────────────────────────────────

fn decode_hex(s: &str) -> Result<Vec<u8>, String> {
    if s.len() % 2 != 0 {
        return Err(format!("hex impar: {} chars", s.len()));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

pub fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Convierte el `target` compacto del pool (hex little-endian) al umbral u64
/// que representa, reproduciendo EXACTAMENTE la lógica del pool de Multipool DDMiner
/// (crates/plugins/cryptonote/src/validator.rs::pool_target_u64_from_hex):
///   - 4 bytes: se interpretan como u64 LE y se desplazan 32 bits a la parte
///     alta (`lo << 32`) — el compacto lleva los top-32-bits de u64::MAX/diff.
///   - otro tamaño (≤8): se usa como u64 LE directo.
/// Luego se coloca ese u64 en los 8 bytes más significativos del target de
/// 256 bits LE ([24..32]), para que el `meets_target` del motor (comparación
/// 256-bit LE) sea equivalente a `u64(hash[24..32]) < target_u64` del pool.
pub fn parse_target(hex_le: &str) -> Result<[u8; 32], String> {
    let bytes = decode_hex(hex_le)?;
    if bytes.is_empty() || bytes.len() > 8 {
        return Err(format!("longitud de target no soportada: {} bytes", bytes.len()));
    }
    let mut le = [0u8; 8];
    le[..bytes.len()].copy_from_slice(&bytes);
    let lo = u64::from_le_bytes(le);
    let target_u64 = if bytes.len() == 4 { lo << 32 } else { lo };
    let mut t = [0u8; 32];
    t[24..32].copy_from_slice(&target_u64.to_le_bytes());
    Ok(t)
}

/// El umbral u64 que el pool compara contra `u64(hash[24..32])` — expuesto
/// para tests y para validación cruzada con el pool.
pub fn pool_target_u64(hex_le: &str) -> Result<u64, String> {
    let t = parse_target(hex_le)?;
    Ok(u64::from_le_bytes(t[24..32].try_into().unwrap()))
}

/// Mapea un job del pool a un `RxJob` del motor.
pub fn job_to_rxjob(job: &StratumJob) -> Result<RxJob, String> {
    let blob = decode_hex(&job.blob)?;
    if blob.len() < MONERO_NONCE_OFFSET + 4 {
        return Err(format!("blob demasiado corto: {} bytes", blob.len()));
    }
    let target = parse_target(&job.target)?;
    let seed_hash = decode_hex(&job.seed_hash).unwrap_or_default();
    Ok(RxJob {
        job_id: job.job_id.clone(),
        blob,
        target,
        nonce_offset: MONERO_NONCE_OFFSET,
        seed_hash,
        height: job.height,
    })
}

/// Construye los params de un `submit` a partir de un share del motor.
pub fn build_submit(session_id: &str, job_id: &str, nonce: u32, hash: &[u8; 32]) -> SubmitParams {
    SubmitParams {
        id: session_id.to_string(),
        job_id: job_id.to_string(),
        nonce: encode_hex(&nonce.to_le_bytes()),
        result: encode_hex(hash),
    }
}

// ─────────────────────────── Cliente async ──────────────────────────────────

/// Eventos que el cliente emite hacia el orquestador.
#[derive(Debug, Clone)]
pub enum PoolEvent {
    /// Nuevo job (ya mapeado). El segundo campo es el session-id para submits.
    NewJob(RxJob, String),
    /// El pool aceptó/rechazó un share.
    SubmitResult { accepted: bool },
    Disconnected(String),
}

#[derive(Clone)]
pub struct StratumConfig {
    pub host: String,
    pub port: u16,
    pub login: String,
    pub worker: String,
    pub algo: String,
}

/// Conecta, hace login y bombea jobs al canal `events`. Devuelve un sender por
/// el que el orquestador envía shares para submit. Bucle hasta desconexión.
pub async fn run_client(
    cfg: StratumConfig,
    events: mpsc::UnboundedSender<PoolEvent>,
) -> Result<mpsc::UnboundedSender<(String, u32, [u8; 32])>, String> {
    let stream = TcpStream::connect((cfg.host.as_str(), cfg.port))
        .await
        .map_err(|e| format!("connect: {e}"))?;
    let (rd, mut wr) = stream.into_split();
    let mut reader = BufReader::new(rd);

    // login
    let login = serde_json::json!({
        "id": 1, "method": "login",
        "params": LoginParams {
            login: cfg.login.clone(),
            pass: cfg.worker.clone(),
            agent: "ddminer/0.3.5".into(),
            rigid: None,
            algo: if cfg.algo.is_empty() { vec![] } else { vec![cfg.algo.clone()] },
        }
    });
    wr.write_all(format!("{login}\n").as_bytes()).await.map_err(|e| e.to_string())?;

    let (share_tx, mut share_rx) = mpsc::unbounded_channel::<(String, u32, [u8; 32])>();

    // tarea de submit
    let mut wr_submit = wr;
    let session = std::sync::Arc::new(tokio::sync::Mutex::new(String::new()));
    let session_w = session.clone();
    tokio::spawn(async move {
        let mut sub_id = 100u64;
        while let Some((job_id, nonce, hash)) = share_rx.recv().await {
            let sid = session_w.lock().await.clone();
            if sid.is_empty() {
                continue;
            }
            let p = build_submit(&sid, &job_id, nonce, &hash);
            let msg = serde_json::json!({"id": sub_id, "method": "submit", "params": p});
            sub_id += 1;
            if wr_submit.write_all(format!("{msg}\n").as_bytes()).await.is_err() {
                break;
            }
        }
    });

    // tarea de lectura
    tokio::spawn(async move {
        let mut line = String::new();
        loop {
            line.clear();
            match reader.read_line(&mut line).await {
                Ok(0) => {
                    let _ = events.send(PoolEvent::Disconnected("eof".into()));
                    break;
                }
                Ok(_) => {
                    if let Some(ev) = handle_line(line.trim(), &session).await {
                        if events.send(ev).is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    let _ = events.send(PoolEvent::Disconnected(e.to_string()));
                    break;
                }
            }
        }
    });

    Ok(share_tx)
}

/// Procesa una línea JSON del pool. Maneja: respuesta de login (result.id +
/// result.job), push de `job`, y resultado de submit.
async fn handle_line(line: &str, session: &std::sync::Arc<tokio::sync::Mutex<String>>) -> Option<PoolEvent> {
    if line.is_empty() {
        return None;
    }
    let v: Value = serde_json::from_str(line).ok()?;

    // Respuesta de login: { result: { id, job, status } }
    if let Some(result) = v.get("result") {
        if let Some(id) = result.get("id").and_then(|x| x.as_str()) {
            *session.lock().await = id.to_string();
            if let Some(job_v) = result.get("job") {
                if let Ok(job) = serde_json::from_value::<StratumJob>(job_v.clone()) {
                    if let Ok(rx) = job_to_rxjob(&job) {
                        return Some(PoolEvent::NewJob(rx, id.to_string()));
                    }
                }
            }
            return None;
        }
        // resultado de un submit: { result: { status: "OK" } } o booleano
        if let Some(status) = result.get("status").and_then(|x| x.as_str()) {
            return Some(PoolEvent::SubmitResult { accepted: status == "OK" });
        }
    }

    // Push de job: { method: "job", params: {...} }
    if v.get("method").and_then(|m| m.as_str()) == Some("job") {
        if let Some(params) = v.get("params") {
            if let Ok(job) = serde_json::from_value::<StratumJob>(params.clone()) {
                if let Ok(rx) = job_to_rxjob(&job) {
                    let sid = session.lock().await.clone();
                    return Some(PoolEvent::NewJob(rx, sid));
                }
            }
        }
    }
    None
}

/// Lanza un keepalive periódico (keepalived) para que el pool no corte.
pub async fn keepalive_loop(mut wr: tokio::net::tcp::OwnedWriteHalf, session: String, secs: u64) {
    let mut ticker = tokio::time::interval(Duration::from_secs(secs));
    loop {
        ticker.tick().await;
        let msg = serde_json::json!({"id": 1, "method": "keepalived", "params": {"id": session}});
        if wr.write_all(format!("{msg}\n").as_bytes()).await.is_err() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::RxJob;

    #[test]
    fn target_4byte_pool_formula() {
        // 4 bytes "ffffffff" → lo = 0xFFFFFFFF → <<32 = 0xFFFFFFFF00000000
        // (replica pool_target_u64_from_hex del validador del pool).
        assert_eq!(pool_target_u64("ffffffff").unwrap(), 0xFFFF_FFFF_0000_0000);
        let t = parse_target("ffffffff").unwrap();
        assert_eq!(&t[0..24], &[0u8; 24]);
        assert_eq!(u64::from_le_bytes(t[24..32].try_into().unwrap()), 0xFFFF_FFFF_0000_0000);
    }

    #[test]
    fn target_harder_is_smaller() {
        let easy = pool_target_u64("ffffffff").unwrap();
        let hard = pool_target_u64("ffff0000").unwrap(); // lo = 0x0000ffff
        assert!(hard < easy, "hard {hard:#x} < easy {easy:#x}");
        assert_eq!(hard, 0x0000_FFFF_0000_0000);
    }

    #[test]
    fn target_8byte_direct() {
        let t = parse_target("0102030405060708").unwrap();
        assert_eq!(&t[24..32], &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(pool_target_u64("0102030405060708").unwrap(), 0x0807_0605_0403_0201);
    }

    #[test]
    fn meets_target_equivalent_to_pool() {
        // El motor (256-bit LE estricto) debe dar el MISMO veredicto que el
        // pool: u64(hash[24..32] LE) < target_u64, para cualquier cola.
        let hex = "00010000"; // 4 bytes
        let t = parse_target(hex).unwrap();
        let tu = pool_target_u64(hex).unwrap();
        for tail in [0u64, 1, tu.wrapping_sub(1), tu, tu.wrapping_add(1), u64::MAX] {
            let mut h = [0u8; 32];
            h[24..32].copy_from_slice(&tail.to_le_bytes());
            assert_eq!(
                RxJob::meets_target(&h, &t),
                tail < tu,
                "discrepancia en tail={tail:#x} tu={tu:#x}"
            );
        }
    }

    #[test]
    fn submit_format() {
        let hash = [0xABu8; 32];
        let s = build_submit("sess123", "job7", 0x12345678, &hash);
        assert_eq!(s.id, "sess123");
        assert_eq!(s.job_id, "job7");
        // nonce 0x12345678 en LE = 78 56 34 12
        assert_eq!(s.nonce, "78563412");
        assert_eq!(s.result, "ab".repeat(32));
    }

    #[test]
    fn job_mapping_rejects_short_blob() {
        let job = StratumJob {
            job_id: "j".into(),
            blob: "00".repeat(10), // 10 bytes < 43
            target: "ffffffff".into(),
            seed_hash: "00".repeat(32),
            height: 1,
            algo: "rx/0".into(),
        };
        assert!(job_to_rxjob(&job).is_err());
    }

    #[test]
    fn job_mapping_ok() {
        let job = StratumJob {
            job_id: "j1".into(),
            blob: "11".repeat(76),
            target: "cdef0000".into(),
            seed_hash: "aa".repeat(32),
            height: 42,
            algo: "rx/0".into(),
        };
        let rx = job_to_rxjob(&job).unwrap();
        assert_eq!(rx.job_id, "j1");
        assert_eq!(rx.blob.len(), 76);
        assert_eq!(rx.nonce_offset, MONERO_NONCE_OFFSET);
        assert_eq!(rx.seed_hash.len(), 32);
        assert_eq!(rx.height, 42);
    }
}
