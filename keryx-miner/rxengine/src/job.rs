//! Trabajo de minado RandomX, genérico (independiente del pool).
//! La conversión del target "compacto" del stratum del pool a este target de
//! 256 bits se hará en la Fase 2 (cliente stratum), cuando veamos el formato
//! exacto del multipool del socio. Aquí el motor trabaja siempre con 256-bit LE.

/// Un trabajo de minado RandomX.
#[derive(Clone, Debug)]
pub struct RxJob {
    pub job_id: String,
    /// Block template (blob) sobre el que se inyecta el nonce.
    pub blob: Vec<u8>,
    /// Umbral de dificultad como entero de 256 bits en little-endian.
    /// Un hash es válido si, interpretado como entero LE, es <= target.
    pub target: [u8; 32],
    /// Offset (en bytes) donde va el nonce de 4 bytes dentro del blob.
    /// En Monero/CryptoNote el nonce ocupa los bytes 39..43.
    pub nonce_offset: usize,
    /// Seed hash que selecciona la caché/dataset de RandomX (cambia ~cada época).
    pub seed_hash: Vec<u8>,
    pub height: u64,
}

impl RxJob {
    /// ¿El hash (salida RandomX de 32 bytes, interpretada como entero
    /// little-endian) cumple el target? Válido si `hash < target` (estricto,
    /// convención Monero/CryptoNote — coincide con el validador del pool).
    #[inline]
    pub fn meets_target(hash: &[u8], target: &[u8; 32]) -> bool {
        debug_assert_eq!(hash.len(), 32);
        // Comparación de enteros de 256 bits en little-endian: del byte más
        // significativo (índice 31) al menos significativo (0).
        for i in (0..32).rev() {
            let h = hash[i];
            let t = target[i];
            if h < t {
                return true;
            }
            if h > t {
                return false;
            }
        }
        false // exactamente igual: NO válido (hash debe ser estrictamente menor)
    }
}

/// Construye un target de 256-bit LE a partir de una dificultad entera.
/// target = floor(2^256 - 1 / difficulty). Útil para tests y para derivar el
/// umbral cuando el pool envía dificultad en vez de target empaquetado.
pub fn target_from_difficulty(difficulty: u64) -> [u8; 32] {
    if difficulty <= 1 {
        return [0xFF; 32];
    }
    // 2^256 / difficulty mediante división larga byte a byte sobre 0xFFFF...FF.
    // Aproximación suficiente (igual que cryptonote): dividimos 2^256-1.
    let mut rem: u128 = 0;
    let mut out = [0u8; 32];
    // Procesamos del byte más significativo al menos significativo en big-endian
    // y luego volcamos a little-endian.
    let mut be = [0u8; 32];
    for i in 0..32 {
        // tomamos 0xFF como dividendo de este byte arrastrando el resto
        rem = (rem << 8) | 0xFF;
        be[i] = (rem / difficulty as u128) as u8;
        rem %= difficulty as u128;
    }
    for i in 0..32 {
        out[i] = be[31 - i];
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_ordering() {
        let easy = [0xFFu8; 32]; // dificultad 1: casi todo hash vale
        let zero_hash = [0u8; 32];
        assert!(RxJob::meets_target(&zero_hash, &easy));

        // target medio: hash con byte alto por encima no vale
        let mut tgt = [0u8; 32];
        tgt[31] = 0x0F;
        let mut high = [0u8; 32];
        high[31] = 0x10;
        assert!(!RxJob::meets_target(&high, &tgt));
        let mut low = [0u8; 32];
        low[31] = 0x0E;
        assert!(RxJob::meets_target(&low, &tgt));
    }

    #[test]
    fn difficulty_target_monotonic() {
        // a mayor dificultad, target más pequeño (más FF arriba en LE = byte 31 menor)
        let t1 = target_from_difficulty(1);
        let t1000 = target_from_difficulty(1000);
        assert!(RxJob::meets_target(&[0u8; 32], &t1000));
        // el byte más significativo del target fácil >= el del difícil
        assert!(t1[31] >= t1000[31]);
    }
}
