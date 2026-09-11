# llama.cpp patches

## `llama-server-expert-stats.patch`

Adds live Mixture-of-Experts routing statistics to `llama-server`.

Stock llama.cpp computes the router's top-k expert choices in every MoE layer
(the graph tensor `ffn_moe_topk-<layer>`, int32, shape `[n_expert_used, n_tokens]`)
but exposes them nowhere. This patch:

| Change | File |
|---|---|
| `--expert-stats` flag, env `LLAMA_ARG_EXPERT_STATS=1` | `common/arg.cpp`, `common/common.h` |
| Scheduler eval callback that reads each layer's top-k tensor after it is computed | `tools/server/server-context.cpp` |
| Per-layer ring of the newest 256 routings, a hit histogram over that window, lifetime totals | same |
| `GET /experts` endpoint | `tools/server/server-context.cpp`, `server-context.h`, `server.cpp` |

The callback is only installed when the flag is set, so default behaviour is
unchanged. It was written against llama.cpp commit `9a3a0fbbe` (1 Sep 2026) and
verified by loading a Qwen3.6-35B-A3B GGUF (256 experts, 8 active).

### Apply

```sh
cd /path/to/llama.cpp
git apply /path/to/llm-visuals/patches/llama-server-expert-stats.patch
cmake --build build --target llama-server -j
```

If `git apply` reports conflicts on a newer llama.cpp, the four hunks are
small and self-contained; `git apply --3way` or a manual merge takes minutes.

### Run

```sh
llama-server --model model.gguf --expert-stats --metrics ...
# or, for a systemd unit, a drop-in with:
#   Environment=LLAMA_ARG_EXPERT_STATS=1
#   Environment=LLAMA_ARG_ENDPOINT_METRICS=1
```

### Endpoint

`GET /experts` (add `?total=1` for lifetime counts):

```json
{
  "n_expert": 256,
  "n_expert_used": 8,
  "n_tokens": 31,
  "window": 256,
  "tail": 16,
  "layers": [
    {
      "il": 0,
      "n_tokens": 31,
      "tokens": [[181,107,82,251,39,99,68,83], ...],   // newest 16 routings, oldest first
      "recent": [3, 0, 1, ...],                          // hits per expert in the last 256 tokens
      "total":  [12, 0, 4, ...]                          // only with ?total=1
    },
    ...
  ]
}
```

Layers that have never routed (dense layers, or before the first token) are
omitted. `n_tokens` is the largest per-layer count and increases during prefill
as well as decode, so a client can flash exactly the routings that arrived
since its previous poll by comparing counters.

### Cost

With an eval callback installed, ggml's scheduler has to finish computing each
watched tensor before the callback can read it, which adds a device sync per
MoE layer per batch. On a two-GPU Qwen3.6-35B-A3B setup this is a few percent
of decode throughput. Leave the flag off when you are not looking at the map.
