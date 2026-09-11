# Roadmap

See `../README.md` for what the tool shows and `ARCHITECTURE.md` for how it
is put together.

## Done

- Auto-detects the running inference server and reads its GGUF header.
- Throughput, TTFT, tok/J and request log derived from `/slots` counters.
- GPU cards from `nvidia-smi` with peak-hold gauges, VRAM split, histories.
- Context bar, MTP acceptance panel (`/metrics`), layer tiles, expert heat map.
- Real router choices via a llama-server patch (`patches/`, `GET /experts`).
- Truecolor with automatic 256-colour fallback; `--demo` runs without hardware.

## Ideas not yet done

- Per-position acceptance rates for deeper MTP drafts (server logs them but
  does not export them).
- Show the /experts 256-token histogram as a load-balance view per layer.
- Ollama / vLLM metrics endpoints for throughput on non-llama.cpp servers.
