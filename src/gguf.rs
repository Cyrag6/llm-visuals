use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Architecture fields from a GGUF header (no tensors loaded).
#[derive(Debug, Clone, Default)]
pub struct GgufInfo {
    pub name: String,
    pub architecture: String,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_experts: usize,
    pub n_experts_used: usize,
    pub ctx_train: usize,
    #[allow(dead_code)]
    pub n_embd: usize,
    /// llama.cpp `nextn_predict_layers` (MTP draft depth). 0 if not MTP.
    pub n_mtp: usize,
}

impl GgufInfo {
    pub fn is_moe(&self) -> bool {
        self.n_experts > 1
    }
}

pub fn read_info(path: &Path) -> Result<GgufInfo, String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"GGUF" {
        return Err("not a GGUF file".into());
    }
    let version = read_u32(&mut f)?;
    if version < 2 {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let _n_tensors = read_u64(&mut f)?;
    let n_kv = read_u64(&mut f)? as usize;

    let mut map = Vec::with_capacity(n_kv.min(48));
    for _ in 0..n_kv {
        let key = read_string(&mut f)?;
        let ty = read_u32(&mut f)?;
        // Tokenizer tables are huge; architecture keys come first.
        if key.starts_with("tokenizer.") {
            break;
        }
        let val = read_value(&mut f, ty)?;
        map.push((key, val));
    }

    let architecture = kv_str(&map, "general.architecture").unwrap_or_default();
    let prefix = if architecture.is_empty() {
        String::new()
    } else {
        format!("{architecture}.")
    };

    let n_layers = kv_usize(&map, &format!("{prefix}block_count"))
        .or_else(|| kv_usize(&map, "llama.block_count"))
        .unwrap_or(0);
    let n_heads = kv_usize(&map, &format!("{prefix}attention.head_count")).unwrap_or(0);
    let n_kv_heads = kv_usize(&map, &format!("{prefix}attention.head_count_kv")).unwrap_or(n_heads);
    let n_experts = kv_usize(&map, &format!("{prefix}expert_count")).unwrap_or(0);
    let n_experts_used = kv_usize(&map, &format!("{prefix}expert_used_count")).unwrap_or(0);
    let ctx_train = kv_usize(&map, &format!("{prefix}context_length")).unwrap_or(0);
    let n_embd = kv_usize(&map, &format!("{prefix}embedding_length")).unwrap_or(0);
    let n_mtp = kv_usize(&map, &format!("{prefix}nextn_predict_layers")).unwrap_or(0);
    let name = kv_str(&map, "general.name")
        .or_else(|| kv_str(&map, "general.basename"))
        .unwrap_or_else(|| {
            path.file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default()
        });

    Ok(GgufInfo {
        name,
        architecture,
        n_layers,
        n_heads,
        n_kv_heads,
        n_experts,
        n_experts_used,
        ctx_train,
        n_embd,
        n_mtp,
    })
}

/// Where the bytes of a GGUF live: total, expert (MoE) tensors, the token
/// embedding table, and per-block totals. Sizes come from the gap between
/// consecutive tensor offsets, so no quantisation type table is needed.
#[derive(Debug, Clone, Default)]
pub struct TensorSummary {
    pub total_bytes: u64,
    /// `ffn_*_exps` tensors: only `n_experts_used / n_experts` of these are
    /// touched per token.
    pub expert_bytes: u64,
    /// `token_embd`: a row lookup, not a matmul, so it is not streamed per token.
    pub embd_bytes: u64,
    /// Bytes per transformer block (`blk.N.*`), indexed by N.
    pub block_bytes: Vec<u64>,
    pub n_tensors: usize,
}

impl TensorSummary {
    /// Weight bytes read to produce one token (or one verification step of a
    /// speculative decoder): everything except the embedding lookup, with the
    /// expert tensors scaled by the routed fraction.
    pub fn active_bytes_per_token(&self, n_experts: usize, n_experts_used: usize) -> u64 {
        let dense = self.total_bytes.saturating_sub(self.expert_bytes).saturating_sub(self.embd_bytes);
        let experts = if n_experts > 0 && n_experts_used > 0 && n_experts_used < n_experts {
            (self.expert_bytes as f64 * n_experts_used as f64 / n_experts as f64) as u64
        } else {
            self.expert_bytes
        };
        dense + experts
    }
}

/// Walk the whole header (including tokenizer arrays) to reach the tensor
/// info table. Takes ~100 ms on a 25 GB file; call it once per detection.
pub fn read_tensor_summary(path: &Path) -> Result<TensorSummary, String> {
    let file_len = std::fs::metadata(path).map_err(|e| e.to_string())?.len();
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic).map_err(|e| e.to_string())?;
    if &magic != b"GGUF" {
        return Err("not a GGUF file".into());
    }
    let version = read_u32(&mut f)?;
    if version < 2 {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let n_tensors = read_u64(&mut f)? as usize;
    let n_kv = read_u64(&mut f)? as usize;
    let mut alignment: u64 = 32;
    for _ in 0..n_kv {
        let key = read_string(&mut f)?;
        let ty = read_u32(&mut f)?;
        let val = read_value(&mut f, ty)?;
        if key == "general.alignment" {
            if let Val::U(a) = val {
                if a > 0 {
                    alignment = a;
                }
            }
        }
    }
    let mut tensors: Vec<(String, u64)> = Vec::with_capacity(n_tensors.min(4096));
    for _ in 0..n_tensors {
        let name = read_string(&mut f)?;
        let n_dims = read_u32(&mut f)?;
        if n_dims > 8 {
            return Err("bad tensor dimension count".into());
        }
        for _ in 0..n_dims {
            let _ = read_u64(&mut f)?;
        }
        let _ty = read_u32(&mut f)?;
        let offset = read_u64(&mut f)?;
        tensors.push((name, offset));
    }
    let header_end = f.stream_position().map_err(|e| e.to_string())?;
    let data_start = (header_end + alignment - 1) / alignment * alignment;
    let data_len = file_len.saturating_sub(data_start);
    tensors.sort_by_key(|(_, off)| *off);
    let mut out = TensorSummary {
        n_tensors: tensors.len(),
        ..Default::default()
    };
    for i in 0..tensors.len() {
        let (name, off) = &tensors[i];
        let next = tensors.get(i + 1).map(|(_, o)| *o).unwrap_or(data_len);
        let size = next.saturating_sub(*off);
        out.total_bytes += size;
        if name.contains("_exps") {
            out.expert_bytes += size;
        }
        if name.starts_with("token_embd") {
            out.embd_bytes += size;
        }
        if let Some(rest) = name.strip_prefix("blk.") {
            if let Some(n) = rest.split('.').next().and_then(|n| n.parse::<usize>().ok()) {
                if n < 4096 {
                    if out.block_bytes.len() <= n {
                        out.block_bytes.resize(n + 1, 0);
                    }
                    out.block_bytes[n] += size;
                }
            }
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
enum Val {
    U(u64),
    I(i64),
    F(f64),
    #[allow(dead_code)]
    B(bool),
    S(String),
    Other,
}

fn kv_str(map: &[(String, Val)], key: &str) -> Option<String> {
    map.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        Val::S(s) => Some(s.clone()),
        _ => None,
    })
}

fn kv_usize(map: &[(String, Val)], key: &str) -> Option<usize> {
    map.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        Val::U(n) => Some(*n as usize),
        Val::I(n) if *n >= 0 => Some(*n as usize),
        Val::F(n) if *n >= 0.0 => Some(*n as usize),
        _ => None,
    })
}

fn read_u32(f: &mut File) -> Result<u32, String> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(f: &mut File) -> Result<u64, String> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(b))
}

fn read_string(f: &mut File) -> Result<String, String> {
    let n = read_u64(f)? as usize;
    if n > 1_000_000 {
        return Err("GGUF string too large".into());
    }
    let mut buf = vec![0u8; n];
    f.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn read_value(f: &mut File, ty: u32) -> Result<Val, String> {
    match ty {
        0 => Ok(Val::U(read_exact_n::<1>(f)?[0] as u64)),
        1 => Ok(Val::I(read_exact_n::<1>(f)?[0] as i8 as i64)),
        2 => {
            let b = read_exact_n::<2>(f)?;
            Ok(Val::U(u16::from_le_bytes(b) as u64))
        }
        3 => {
            let b = read_exact_n::<2>(f)?;
            Ok(Val::I(i16::from_le_bytes(b) as i64))
        }
        4 => Ok(Val::U(read_u32(f)? as u64)),
        5 => Ok(Val::I(read_u32(f)? as i32 as i64)),
        6 => {
            let b = read_exact_n::<4>(f)?;
            Ok(Val::F(f32::from_le_bytes(b) as f64))
        }
        7 => Ok(Val::B(read_exact_n::<1>(f)?[0] != 0)),
        8 => Ok(Val::S(read_string(f)?)),
        9 => {
            skip_array(f)?;
            Ok(Val::Other)
        }
        10 => Ok(Val::U(read_u64(f)?)),
        11 => {
            let b = read_exact_n::<8>(f)?;
            Ok(Val::I(i64::from_le_bytes(b)))
        }
        12 => {
            let b = read_exact_n::<8>(f)?;
            Ok(Val::F(f64::from_le_bytes(b)))
        }
        _ => Err(format!("unknown GGUF value type {ty}")),
    }
}

fn read_exact_n<const N: usize>(f: &mut File) -> Result<[u8; N], String> {
    let mut b = [0u8; N];
    f.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(b)
}

fn skip_array(f: &mut File) -> Result<(), String> {
    let at = read_u32(f)?;
    let n = read_u64(f)?;
    for _ in 0..n {
        let _ = read_value(f, at)?;
    }
    Ok(())
}

/// Which GPU a layer lives on given llama.cpp `--tensor-split` percentages.
pub fn layer_device(layer: usize, n_layers: usize, split: &[f32]) -> usize {
    if split.is_empty() || n_layers == 0 {
        return 0;
    }
    let total: f32 = split.iter().copied().sum::<f32>().max(1.0);
    let t = (layer as f32 + 0.5) / n_layers as f32 * total;
    let mut acc = 0.0;
    for (i, s) in split.iter().enumerate() {
        acc += *s;
        if t <= acc {
            return i;
        }
    }
    split.len() - 1
}

#[allow(dead_code)]
pub fn skip_rest(f: &mut File) -> Result<(), String> {
    f.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_split_63_37() {
        let split = [63.0, 37.0];
        assert_eq!(layer_device(0, 41, &split), 0);
        assert_eq!(layer_device(25, 41, &split), 0);
        assert_eq!(layer_device(40, 41, &split), 1);
    }

    #[test]
    /// Set LLM_VISUALS_TEST_GGUF to a local MoE GGUF to exercise the reader
    /// against a real file; skipped otherwise.
    fn read_local_gguf_if_configured() {
        let Ok(p) = std::env::var("LLM_VISUALS_TEST_GGUF") else {
            return;
        };
        let path = Path::new(&p);
        if !path.exists() {
            return;
        }
        let info = read_info(path).expect("gguf header");
        assert!(info.n_layers > 0);
        assert!(info.n_heads > 0);
        assert!(!info.architecture.is_empty());
        let t = read_tensor_summary(path).expect("tensor table");
        let file_len = std::fs::metadata(path).unwrap().len();
        // Every byte of the data section belongs to some tensor.
        assert!(t.total_bytes > file_len / 2 && t.total_bytes <= file_len);
        // block_count may or may not include the MTP/nextn blocks.
        assert!(t.block_bytes.len() >= info.n_layers && t.block_bytes.len() <= info.n_layers + info.n_mtp + 1);
        if info.is_moe() {
            assert!(t.expert_bytes > t.total_bytes / 2);
        }
        assert!(t.active_bytes_per_token(info.n_experts, info.n_experts_used) < t.total_bytes);
    }

    #[test]
    fn active_bytes_scales_experts_only() {
        let t = TensorSummary {
            total_bytes: 1000,
            expert_bytes: 800,
            embd_bytes: 50,
            block_bytes: vec![],
            n_tensors: 3,
        };
        assert_eq!(t.active_bytes_per_token(0, 0), 950);
        assert_eq!(t.active_bytes_per_token(256, 8), 150 + 25);
    }
}
