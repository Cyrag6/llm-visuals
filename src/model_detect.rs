use std::path::{Path, PathBuf};

use crate::gguf::{self, GgufInfo};

/// Information about a detected running LLM process
#[derive(Debug, Clone)]
pub struct DetectedModel {
    pub name: String,
    pub path: Option<PathBuf>,
    pub pid: u32,
    #[allow(dead_code)]
    pub process_name: String,
    pub engine: String,
    pub gpu_indices: Vec<u32>,
    pub mem_used_mb: u64,
    pub port: Option<u16>,
    pub ctx_max: Option<usize>,
    pub spec_type: Option<String>,
    #[allow(dead_code)]
    pub n_gpu_layers: Option<u32>,
    pub tensor_split: Vec<f32>,
    pub cmdline: String,
    pub gguf: Option<GgufInfo>,
    /// Weight byte layout from the tensor table (for bandwidth estimates).
    pub tensors: Option<gguf::TensorSummary>,
}

impl std::fmt::Display for DetectedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gpus = if self.gpu_indices.is_empty() {
            "?".into()
        } else {
            self.gpu_indices
                .iter()
                .map(|g| g.to_string())
                .collect::<Vec<_>>()
                .join(",")
        };
        write!(
            f,
            "{} (PID {} · {} · GPU {} · {} MB)",
            self.name, self.pid, self.engine, gpus, self.mem_used_mb
        )
    }
}

impl DetectedModel {
    pub fn n_layers(&self) -> usize {
        self.gguf.as_ref().map(|g| g.n_layers).unwrap_or(0)
    }

    pub fn n_heads(&self) -> usize {
        self.gguf.as_ref().map(|g| g.n_heads).unwrap_or(0)
    }

    pub fn n_experts(&self) -> usize {
        self.gguf.as_ref().map(|g| g.n_experts).unwrap_or(0)
    }

    pub fn n_experts_used(&self) -> usize {
        self.gguf.as_ref().map(|g| g.n_experts_used).unwrap_or(0)
    }
}

/// Scan GPU compute apps + process cmdlines for inference servers.
pub fn detect_models() -> Vec<DetectedModel> {
    let gpu_procs = nvidia_compute_apps();
    let mut by_pid: std::collections::HashMap<u32, DetectedModel> = std::collections::HashMap::new();

    for app in gpu_procs {
        let cmdline = read_cmdline(app.pid).unwrap_or_else(|| app.process_name.clone());
        if !looks_like_llm(&app.process_name, &cmdline) {
            continue;
        }
        let parsed = parse_cmdline(&app.process_name, &cmdline);
        let entry = by_pid.entry(app.pid).or_insert_with(|| DetectedModel {
            name: parsed.name.clone(),
            path: parsed.path.clone(),
            pid: app.pid,
            process_name: app.process_name.clone(),
            engine: parsed.engine.clone(),
            gpu_indices: Vec::new(),
            mem_used_mb: 0,
            port: parsed.port,
            ctx_max: parsed.ctx_max,
            spec_type: parsed.spec_type.clone(),
            n_gpu_layers: parsed.n_gpu_layers,
            tensor_split: parsed.tensor_split.clone(),
            cmdline: cmdline.clone(),
            gguf: None,
            tensors: None,
        });
        if !entry.gpu_indices.contains(&app.gpu_index) {
            entry.gpu_indices.push(app.gpu_index);
        }
        entry.mem_used_mb = entry.mem_used_mb.saturating_add(app.mem_used_mb);
    }

    // /proc scan for engines that might not appear in nvidia-smi
    for (pid, name, cmdline) in walk_proc_llms() {
        if by_pid.contains_key(&pid) {
            continue;
        }
        let parsed = parse_cmdline(&name, &cmdline);
        by_pid.insert(
            pid,
            DetectedModel {
                name: parsed.name,
                path: parsed.path,
                pid,
                process_name: name,
                engine: parsed.engine,
                gpu_indices: Vec::new(),
                mem_used_mb: 0,
                port: parsed.port,
                ctx_max: parsed.ctx_max,
                spec_type: parsed.spec_type,
                n_gpu_layers: parsed.n_gpu_layers,
                tensor_split: parsed.tensor_split,
                cmdline,
                gguf: None,
            tensors: None,
            },
        );
    }

    let mut models: Vec<DetectedModel> = by_pid.into_values().collect();
    for m in &mut models {
        if let Some(path) = m.path.clone() {
            if path.extension().and_then(|e| e.to_str()) == Some("gguf") {
                if let Ok(info) = gguf::read_info(&path) {
                    if m.name.starts_with('[') || m.name == "llama-server" || m.name.is_empty() {
                        m.name = info.name.clone();
                    }
                    m.gguf = Some(info);
                    m.tensors = gguf::read_tensor_summary(&path).ok();
                }
            }
        }
        m.gpu_indices.sort_unstable();
    }

    // GPU memory first; when nvidia-smi is unavailable that is zero for all,
    // so fall back to "has a model loaded on its command line" and then to a
    // serving engine over a resident daemon (ollama with nothing loaded).
    models.sort_by_key(|m| {
        let has_model = m.path.is_some() || m.gguf.is_some();
        let engine_rank = match m.engine.as_str() {
            "llama.cpp" | "vllm" | "sglang" | "exllamav2" => 2,
            "ollama" => 0,
            _ => 1,
        };
        std::cmp::Reverse((m.mem_used_mb, has_model, engine_rank))
    });
    models
}

#[derive(Default)]
struct ParsedCmd {
    name: String,
    path: Option<PathBuf>,
    engine: String,
    port: Option<u16>,
    ctx_max: Option<usize>,
    spec_type: Option<String>,
    n_gpu_layers: Option<u32>,
    tensor_split: Vec<f32>,
}

fn looks_like_llm(process_name: &str, cmdline: &str) -> bool {
    let p = process_name.to_lowercase();
    let c = cmdline.to_lowercase();
    let keys = [
        "llama-server",
        "llama-cli",
        "llama.cpp",
        "ollama",
        "vllm",
        "sglang",
        "exllama",
        "text-generation",
        "aphrodite",
        "tensorrt-llm",
        "lmdeploy",
        "kobold",
        "tabbyapi",
        "transformers",
        ".gguf",
    ];
    keys.iter().any(|k| p.contains(k) || c.contains(k))
        || c.contains("--model")
        || c.contains(" -m ")
}

fn engine_from(process_name: &str, cmdline: &str) -> String {
    let blob = format!("{process_name} {cmdline}").to_lowercase();
    if blob.contains("llama-server") || blob.contains("llama.cpp") {
        "llama.cpp".into()
    } else if blob.contains("ollama") {
        "ollama".into()
    } else if blob.contains("vllm") {
        "vllm".into()
    } else if blob.contains("sglang") {
        "sglang".into()
    } else if blob.contains("exllama") {
        "exllamav2".into()
    } else if blob.contains("ollama") {
        "ollama".into()
    } else {
        "llm".into()
    }
}

fn parse_cmdline(process_name: &str, cmdline: &str) -> ParsedCmd {
    let tokens: Vec<&str> = cmdline.split_whitespace().collect();
    let mut parsed = ParsedCmd {
        engine: engine_from(process_name, cmdline),
        ..Default::default()
    };

    let mut i = 0;
    while i < tokens.len() {
        let t = tokens[i];
        let (key, inline) = if let Some((k, v)) = t.split_once('=') {
            (k, Some(v))
        } else {
            (t, None)
        };
        let next = || inline.map(|s| s.to_string()).or_else(|| {
            tokens.get(i + 1).map(|s| s.to_string())
        });

        match key {
            "--model" | "-m" | "--model-path" => {
                if let Some(v) = next() {
                    let p = PathBuf::from(&v);
                    parsed.name = p
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or(v.clone());
                    parsed.path = Some(p);
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            "--alias" => {
                if let Some(v) = next() {
                    parsed.name = v;
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            "--port" => {
                if let Some(v) = next() {
                    parsed.port = v.parse().ok();
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            "--ctx-size" | "--ctx_size" | "-c" => {
                if let Some(v) = next() {
                    parsed.ctx_max = v.parse().ok();
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            "--spec-type" | "--spec_type" => {
                if let Some(v) = next() {
                    parsed.spec_type = Some(v);
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            "--n-gpu-layers" | "-ngl" => {
                if let Some(v) = next() {
                    parsed.n_gpu_layers = v.parse().ok();
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            "--tensor-split" | "--tensor_split" => {
                if let Some(v) = next() {
                    parsed.tensor_split = v
                        .split(',')
                        .filter_map(|s| s.trim().parse().ok())
                        .collect();
                    if inline.is_none() {
                        i += 1;
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    if parsed.name.is_empty() {
        parsed.name = Path::new(process_name)
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| process_name.to_string());
    }
    if parsed.port.is_none() && parsed.engine == "llama.cpp" {
        parsed.port = Some(8080);
    }
    if parsed.port.is_none() && parsed.engine == "ollama" {
        parsed.port = Some(11434);
    }
    parsed
}

fn walk_proc_llms() -> Vec<(u32, String, String)> {
    let mut out = Vec::new();
    let Ok(dir) = std::fs::read_dir("/proc") else {
        return out;
    };
    for ent in dir.flatten() {
        let pid: u32 = match ent.file_name().to_string_lossy().parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let cmdline = match read_cmdline(pid) {
            Some(c) if !c.is_empty() => c,
            _ => continue,
        };
        let name = std::fs::read_to_string(format!("/proc/{pid}/comm"))
            .unwrap_or_default()
            .trim()
            .to_string();
        if looks_like_llm(&name, &cmdline) {
            out.push((pid, name, cmdline));
        }
    }
    out
}

fn read_cmdline(pid: u32) -> Option<String> {
    let raw = std::fs::read(format!("/proc/{pid}/cmdline")).ok()?;
    if raw.is_empty() {
        return None;
    }
    Some(
        raw.split(|b| *b == 0)
            .filter(|p| !p.is_empty())
            .map(|p| String::from_utf8_lossy(p).into_owned())
            .collect::<Vec<_>>()
            .join(" "),
    )
}

struct ComputeApp {
    pid: u32,
    process_name: String,
    gpu_index: u32,
    mem_used_mb: u64,
}

fn gpu_uuid_index_map() -> std::collections::HashMap<String, u32> {
    let output = match std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=index,uuid", "--format=csv,noheader,nounits"])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return std::collections::HashMap::new(),
    };
    let mut map = std::collections::HashMap::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        if let Ok(idx) = parts[0].trim().parse::<u32>() {
            map.insert(parts[1].trim().to_string(), idx);
        }
    }
    map
}

fn nvidia_compute_apps() -> Vec<ComputeApp> {
    let uuid_map = gpu_uuid_index_map();
    let output = match std::process::Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return Vec::new(),
    };

    let mut apps = Vec::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 4 {
            continue;
        }
        let uuid = parts[0].trim();
        let pid = match parts[1].trim().parse::<u32>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let process_name = parts[2].trim().to_string();
        let mem_used = match parts[3].trim().parse::<u64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let gpu_index = uuid_map.get(uuid).copied().unwrap_or(0);
        apps.push(ComputeApp {
            pid,
            process_name,
            gpu_index,
            mem_used_mb: mem_used,
        });
    }
    apps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_llama_server_cmdline() {
        let cmd = "/home/dingo/models/llama.cpp/build/bin/llama-server --model /home/dingo/models/Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL.gguf --alias qwen3.6-35b-a3b --host 0.0.0.0 --port 8080 --n-gpu-layers 99 --tensor-split 63,37 --spec-type draft-mtp --ctx-size 98304";
        let p = parse_cmdline("llama-server", cmd);
        assert_eq!(p.name, "qwen3.6-35b-a3b");
        assert_eq!(p.port, Some(8080));
        assert_eq!(p.ctx_max, Some(98304));
        assert_eq!(p.spec_type.as_deref(), Some("draft-mtp"));
        assert_eq!(p.tensor_split, vec![63.0, 37.0]);
        assert_eq!(p.engine, "llama.cpp");
        assert!(p.path.unwrap().ends_with("Qwen3.6-35B-A3B-MTP-UD-Q3_K_XL.gguf"));
    }

    #[test]
    fn extract_model_equals_form() {
        let p = parse_cmdline("python3", "python3 serve.py --model=gpt2");
        assert_eq!(p.name, "gpt2");
    }
}
