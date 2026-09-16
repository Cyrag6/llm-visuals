# Error Log

### nvidia-smi compute-app rows dropped on "[N/A]" memory — 2026-09-17

- **Severity:** High
- **Category:** Logic
- **File(s):** `src/model_detect.rs`
- **Pattern:** Discarding a whole CSV record from an external tool because one informational column fails to parse, when tools print placeholders like `[N/A]` / `[Not Supported]` on some drivers or platforms.
- **Root cause:** `used_gpu_memory` was parsed with `continue` on error; Windows (WDDM) drivers always report `[N/A]`, so every GPU process vanished from detection.
- **Fix applied:** Parse the column with `unwrap_or(0)` so the process is kept with unknown memory.
- **Prevention rule:** When parsing `nvidia-smi` (or similar) CSV, only skip a row when its identity columns (pid, uuid, index) fail to parse; default non-key numeric columns.
