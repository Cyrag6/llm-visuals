# Error Log

### vLLM positional arg parser consumed argv[0] — 2026-09-16

- **Severity:** High
- **Category:** Logic
- **File(s):** `src/model_detect.rs`
- **Pattern:** A positional-argument catch-all in a cmdline parser that accepts "any bare token" also matches argv[0] (always a full binary path in /proc cmdline) and bare values of unhandled flags.
- **Root cause:** The vLLM catch-all only checked "doesn't start with `-` and path not yet set", which argv[0] satisfies before any subcommand token is seen.
- **Fix applied:** Positional capture now requires a preceding `serve`/`api-server` token and a path-shaped value (contains `/` or a weights extension).
- **Prevention rule:** When parsing /proc cmdlines, never let a positional catch-all run before the subcommand token; add a test with the full binary path as argv[0].

### vLLM phantom filter keyed on comm == "vllm" — 2026-09-16

- **Severity:** High
- **Category:** Logic
- **File(s):** `src/model_detect.rs`
- **Pattern:** Classifying a process by requiring an exact comm name, when the same server can appear as `python3`, `pt_main_thread`, or a full executable path depending on launch form and data source.
- **Root cause:** The filter treated "comm != 'vllm' and no --port" as a phantom, dropping real servers launched via the python module form or without an explicit port.
- **Fix applied:** Classify by argv[0] basename (`vllm`) or `vllm.entrypoints` in the cmdline; name prefixes only for known helper/client patterns.
- **Prevention rule:** Never gate process classification on an exact comm string; test with the module-invocation and default-port launch forms.

### Windowed rate cleared on the only poll that carried the delta — 2026-09-16

- **Severity:** High
- **Category:** Logic
- **File(s):** `src/perf.rs`
- **Pattern:** For counters that only move on a terminal event (vLLM tokens move at completion), clearing/resetting the measurement window on that same event reads the value as permanently zero.
- **Root cause:** `decode_win.clear()` on `!processing` ran before the rate was read, and only prefill received the histogram-based clamp.
- **Fix applied:** Symmetric decode clamp from the server ITL sum (`(decoded − 1) / itl_sum`) on the completion poll; test asserts `peak_decode_tps`.
- **Prevention rule:** When adding a rate fallback for completion-only counters, apply it to every rate derived from those counters and assert each in the test.

### Record close gated on an optional measurement — 2026-09-16

- **Severity:** Medium
- **Category:** Logic
- **File(s):** `src/perf.rs`
- **Pattern:** Setting a record's terminal state (`ended`) only inside a branch that also requires an optional measurement (TTFT > 0), leaving the record open forever when the measurement is absent.
- **Root cause:** The backdate logic and the `ended` stamp were fused in one conditional.
- **Fix applied:** `ended` is stamped unconditionally; only the backdate stays gated on a usable TTFT.
- **Prevention rule:** State transitions must never depend on optional telemetry; gate only the enrichment, not the transition.

### Poll loop with no failure path — 2026-09-16

- **Severity:** Medium
- **Category:** Logic
- **File(s):** `src/main.rs`
- **Pattern:** A polling loop whose failure branch is empty freezes downstream consumers on the last accepted sample and keeps polling a deterministically-failing target at full rate.
- **Root cause:** The vLLM loop had `if let Some(..)` with no else; the llama.cpp path's miss-counter policy was not mirrored.
- **Fix applied:** Miss counter: after 3 consecutive failures send an idle sample (UI decays) and back the poll interval off to ≥2 s.
- **Prevention rule:** Every poll loop needs an explicit failure branch: decay the consumer's state and back off; mirror the existing loop's policy when adding a sibling.

### docker-proxy match by container IP only — 2026-09-16

- **Severity:** Medium
- **Category:** Logic
- **File(s):** `src/model_detect.rs`
- **Pattern:** Resolving a container port mapping by matching only the container IP returns an arbitrary mapping when the container publishes several ports (one docker-proxy per mapping, same IP).
- **Root cause:** `-container-port` was not captured or compared against the server's own port.
- **Fix applied:** The match now also requires `-container-port == cmdline_port.unwrap_or(8000)`.
- **Prevention rule:** When resolving via /proc process scans, match on every discriminating field available, not just the first sufficient-looking one.

### Unknown context length rendered as 100% full — 2026-09-16

- **Severity:** Medium
- **Category:** Logic
- **File(s):** `src/main.rs`, `src/model_detect.rs`
- **Pattern:** Substituting `.max(1)` for an unknown denominator turns "unknown" into "completely full" in every ratio-based display.
- **Root cause:** vLLM safetensors servers without `--max-model-len` had no ctx source (the GGUF fallback needs a .gguf file), leaving ctx_max = 0.
- **Fix applied:** Read `max_position_embeddings` from the weights dir's config.json — exactly vLLM's own default for `max-model-len`.
- **Prevention rule:** Every engine-specific detection path needs its own source for each field the renderer divides by; grep for `.max(1)` on the field when adding a new engine.

### Test asserting a duplicated copy of production logic — 2026-09-16

- **Severity:** Low
- **Category:** Convention
- **File(s):** `src/vllm.rs`
- **Pattern:** A unit test exercising a verbatim in-test copy of a production predicate, which stays green when the shipped code drifts.
- **Root cause:** The cross-wiring guard was inlined in an async fn and copy-pasted into the test module for testability.
- **Fix applied:** Extracted `fn cross_wired(..)`, called from `poll_vllm` and tested directly; deleted the duplicate.
- **Prevention rule:** If logic must be copied to be testable, extract it instead — a test may only assert code the binary actually runs.

### nvidia-smi compute-app rows dropped on "[N/A]" memory — 2026-09-17

- **Severity:** High
- **Category:** Logic
- **File(s):** `src/model_detect.rs`
- **Pattern:** Discarding a whole CSV record from an external tool because one informational column fails to parse, when tools print placeholders like `[N/A]` / `[Not Supported]` on some drivers or platforms.
- **Root cause:** `used_gpu_memory` was parsed with `continue` on error; Windows (WDDM) drivers always report `[N/A]`, so every GPU process vanished from detection.
- **Fix applied:** Parse the column with `unwrap_or(0)` so the process is kept with unknown memory.
- **Prevention rule:** When parsing `nvidia-smi` (or similar) CSV, only skip a row when its identity columns (pid, uuid, index) fail to parse; default non-key numeric columns.

### Discover tests assumed an empty host process table — 2026-09-20

- **Severity:** Low
- **Category:** Logic
- **File(s):** `src/main.rs`
- **Pattern:** An integration test of `discover()` asserting `models.is_empty()` or `models.len() == 1` for an explicit `--endpoint`, while production still scans the host process table and can attach extra real servers.
- **Root cause:** Mock-server tests were written on a machine with no llama-server running, so process detection returned nothing and the assertions only covered the mock.
- **Fix applied:** Assert the explicit endpoint is present (and error text for failures) without requiring it to be the only detected model.
- **Prevention rule:** Tests that call `discover()` must select the model under test by port/name; never assert the whole result set is empty or length 1 unless process scanning is stubbed.
