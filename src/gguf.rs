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
    }
}
