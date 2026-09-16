//! `--log-db`: append samples and finished requests to a SQLite file so a
//! session can be queried after the dashboard is closed.

use crate::gpu::GpuStats;
use crate::model_detect::DetectedModel;
use crate::perf::PerfTracker;
use rusqlite::{params, Connection};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS model_samples (
    ts REAL NOT NULL, pid INTEGER NOT NULL, model TEXT NOT NULL, engine TEXT NOT NULL,
    processing INTEGER NOT NULL, decode_tps REAL NOT NULL, prefill_tps REAL NOT NULL,
    ctx_used INTEGER NOT NULL, ctx_max INTEGER NOT NULL,
    session_decoded INTEGER NOT NULL, session_prefilled INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS gpu_samples (
    ts REAL NOT NULL, gpu INTEGER NOT NULL, name TEXT NOT NULL,
    util_pct REAL NOT NULL, mem_used_mb INTEGER NOT NULL, mem_total_mb INTEGER NOT NULL,
    power_w REAL NOT NULL, temp_c REAL);
CREATE TABLE IF NOT EXISTS requests (
    ended_ts REAL NOT NULL, pid INTEGER NOT NULL, model TEXT NOT NULL, id_task INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL, cached_tokens INTEGER NOT NULL, decoded INTEGER NOT NULL,
    ttft_s REAL, duration_s REAL NOT NULL,
    avg_prefill_tps REAL NOT NULL, avg_decode_tps REAL NOT NULL, peak_decode_tps REAL NOT NULL);
";

/// One row per model and per GPU each `every`; one row per finished request.
pub struct DbLog {
    conn: Connection,
    last: Option<Instant>,
    every: std::time::Duration,
    /// pid → `PerfTracker::finished` already written.
    logged: HashMap<u32, u64>,
}

fn unix_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

impl DbLog {
    pub fn open(path: &Path, every: std::time::Duration) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        // WAL lets `sqlite3` read the file while the dashboard is writing.
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(SCHEMA)?;
        Ok(Self {
            conn,
            last: None,
            every,
            logged: HashMap::new(),
        })
    }

    /// Call every frame; writes at most once per `every`.
    pub fn tick<'a>(
        &mut self,
        models: impl Iterator<Item = (&'a DetectedModel, &'a PerfTracker, usize, usize)>,
        gpus: &[GpuStats],
        now: Instant,
    ) -> rusqlite::Result<()> {
        if self.last.is_some_and(|t| now - t < self.every) {
            return Ok(());
        }
        self.last = Some(now);
        let ts = unix_now();
        let tx = self.conn.transaction()?;
        for (m, perf, ctx_used, ctx_max) in models {
            tx.execute(
                "INSERT INTO model_samples VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
                params![
                    ts,
                    m.pid,
                    m.name,
                    m.engine,
                    perf.phase != crate::perf::Phase::Idle,
                    perf.decode_tps,
                    perf.prefill_tps,
                    ctx_used as i64,
                    ctx_max as i64,
                    perf.session_decoded as i64,
                    perf.session_prefilled as i64
                ],
            )?;
            let seen = self.logged.entry(m.pid).or_insert(0);
            // A rescan can hand the pid a fresh tracker; restart the count.
            if perf.finished < *seen {
                *seen = 0;
            }
            let new = (perf.finished - *seen) as usize;
            let start = perf.history.len().saturating_sub(new);
            for r in perf.history.range(start..) {
                let ended = r.ended.unwrap_or(now);
                tx.execute(
                    "INSERT INTO requests VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
                    params![
                        ts - (now - ended).as_secs_f64(),
                        m.pid,
                        m.name,
                        r.id_task,
                        r.prompt_tokens as i64,
                        r.cached_tokens as i64,
                        r.decoded as i64,
                        r.ttft().map(|d| d.as_secs_f64()),
                        r.duration(now).as_secs_f64(),
                        r.avg_prefill_tps(),
                        r.avg_decode_tps(),
                        r.peak_decode_tps
                    ],
                )?;
            }
            *seen = perf.finished;
        }
        for g in gpus {
            tx.execute(
                "INSERT INTO gpu_samples VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
                params![
                    ts,
                    g.index,
                    g.name,
                    g.utilization_gpu,
                    g.mem_used_mb as i64,
                    g.mem_total_mb as i64,
                    g.power_watts,
                    g.temperature
                ],
            )?;
        }
        tx.commit()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observe::LiveStats;
    use std::time::Duration;

    #[test]
    fn logs_samples_and_each_finished_request_once() {
        let path = std::env::temp_dir().join(format!("llm-visuals-dblog-{}", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let mut log = DbLog::open(&path, Duration::ZERO).unwrap();
        let model = crate::demo::demo_models(8192, 1).remove(0);
        let mut perf = PerfTracker::new();
        let t0 = Instant::now();
        let s = |processing, done, decoded| LiveStats {
            id_task: 7,
            processing,
            prompt_tokens: 100,
            prompt_processed: done,
            decoded,
            ..Default::default()
        };
        perf.observe(&s(true, 0, 0), t0);
        perf.observe(&s(true, 100, 0), t0 + Duration::from_millis(200));
        perf.observe(&s(true, 100, 10), t0 + Duration::from_millis(400));
        perf.observe(&s(false, 100, 10), t0 + Duration::from_millis(600));
        assert_eq!(perf.finished, 1);

        let gpus = [crate::gpu::DemoGpu::new(0).step(0.5)];
        let now = t0 + Duration::from_millis(700);
        for _ in 0..2 {
            log.tick(std::iter::once((&model, &perf, 110, 8192)), &gpus, now).unwrap();
        }
        let count = |t: &str| -> i64 {
            log.conn
                .query_row(&format!("SELECT COUNT(*) FROM {t}"), [], |r| r.get(0))
                .unwrap()
        };
        assert_eq!(count("model_samples"), 2);
        assert_eq!(count("gpu_samples"), 2);
        assert_eq!(count("requests"), 1, "a request must not be logged twice");
        let decoded: i64 = log
            .conn
            .query_row("SELECT decoded FROM requests", [], |r| r.get(0))
            .unwrap();
        assert_eq!(decoded, 10);
        let _ = std::fs::remove_file(&path);
    }
}
