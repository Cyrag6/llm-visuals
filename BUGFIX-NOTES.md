# llm-visuals: layers panel shows "1 across 1 GPU" for vLLM safetensors models

## Symptom
Dashboard monitors a running vLLM server (Qwen3.6-35B-A3B, GPTQ safetensors
in `/model`) but the LAYERS panel reads `1 across 1 GPU` instead of
`40 across 1 GPU`. Same for heads/experts panels.

## Root cause (two stacked gaps)
1. **Topology comes only from GGUF.** `DetectedModel::n_layers()/n_heads()/
   n_experts()/n_experts_used()` read `self.gguf` only, and GGUF is parsed
   only when the model path extension is `.gguf` (llama.cpp). vLLM serves a
   directory of `*.safetensors` + `config.json`, so `gguf` is `None` and the
   accessor chain falls back to 0 → clamped to `max(1)` in
   `fade_sample_from_live` (main.rs) → "1 layer".
2. **Container path re-anchoring is host-centric.** The server reports its
   path as `/model` (its mount namespace). `resolve_container_path` maps
   that to the host path by finding the mount's device number (`252:0`) in
   OUR `/proc/self/mountinfo`. Works host→container (host has the ext4
   device mounted), fails container→host: the dashboard container's root is
   overlay, the host ext4 device never appears in its mountinfo, so
   resolution fails and the config.json read never finds the file. This
   second gap means a containerized llm-visuals gets neither GGUF metadata
   nor HF config for a containerized server.

## Fix (src/model_detect.rs)
- New `hf_config_topology(dir) -> Option<GgufInfo>`: parses the served
  directory's `config.json` (`num_hidden_layers`, `num_attention_heads`,
  `num_key_value_heads`, `num_experts`, `num_experts_per_tok`; descends
  into `text_config` first for multimodal wrappers like
  `Qwen3_5MoeForConditionalGeneration`). Stored in a new `DetectedModel.hf`
  field; the `n_*()` accessors fall back GGUF → HF config.
- Path re-anchoring fallback: when device-based mapping fails, read through
  `/proc/<pid>/root/<path>` — reaches the path exactly as the server sees
  it, in either namespace direction (needs same uid or root).
- Earlier fix in same file (kept in the PR): `is_vllm_phantom` dropped real
  servers launched as `python3 /opt/venv/bin/vllm serve ...` (argv0 is an
  interpreter); comm == "vllm" is now accepted as the real server.

## Verification
- Unit tests: flat config, nested text_config, missing config, plus
  regression for interpreter-launched vLLM. 69 passed.
- Live check (`cargo test -- detects_live_servers --ignored` in a
  `--pid host` container against the real server):
  `pid=5121 engine=vllm port=8010 layers=40 heads=16 experts=8/256`
  (Qwen3.6-35B-A3B config.json says 40/16/256/8 — matches).
