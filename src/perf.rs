//! Derived performance metrics: tokens/sec, latency, per-request records and
//! history ring buffers for sparklines. Everything here comes from deltas of
//! the server's own counters between polls, not from guessed constants.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::gpu::GpuStats;
use crate::observe::{LiveStats, SpecMetrics};

pub const HISTORY: usize = 240;
const MAX_REQUESTS: usize = 32;

#[derive(Debug, Clone)]
pub struct RequestRecord {
    pub id_task: i64,
    pub started: Instant,
    pub first_token: Option<Instant>,
    pub ended: Option<Instant>,
    pub prompt_tokens: usize,
    pub cached_tokens: usize,
    pub decoded: usize,
    /// Prompt tokens actually pushed through prefill (prompt minus cache hits).
    pub prefill_tokens: usize,
    pub prefill_secs: f32,
    pub decode_secs: f32,
    pub peak_decode_tps: f32,
}

impl RequestRecord {
    fn new(id_task: i64, now: Instant, s: &LiveStats) -> Self {
        Self {
            id_task,
            started: now,
            first_token: None,
            ended: None,
            prompt_tokens: s.prompt_tokens,
            cached_tokens: s.cache_tokens,
            decoded: s.decoded,
            prefill_tokens: 0,
            prefill_secs: 0.0,
            decode_secs: 0.0,
            peak_decode_tps: 0.0,
        }
    }

    pub fn ttft(&self) -> Option<Duration> {
        self.first_token.map(|t| t - self.started)
    }

    pub fn duration(&self, now: Instant) -> Duration {
        self.ended.unwrap_or(now) - self.started
    }

    pub fn avg_decode_tps(&self) -> f32 {
        if self.decode_secs > 0.05 {
            self.decoded as f32 / self.decode_secs
        } else {
            0.0
        }
    }

    pub fn avg_prefill_tps(&self) -> f32 {
        if self.prefill_secs > 0.05 {
            self.prefill_tokens as f32 / self.prefill_secs
        } else {
            0.0
        }
    }

    pub fn is_live(&self) -> bool {
        self.ended.is_none()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Prefill,
    Decode,
}

/// Token deltas over a sliding time window. Speculative decoders (MTP, draft
/// models) land tokens in bursts, so a per-poll rate flickers; a ~1 s window
/// gives an honest, steady tokens/sec.
#[derive(Debug, Clone)]
pub struct RateWindow {
    span: Duration,
    /// (interval start, interval end, tokens landed in that interval)
    samples: VecDeque<(Instant, Instant, usize)>,
}

impl RateWindow {
    pub fn new(span: Duration) -> Self {
        Self {
            span,
            samples: VecDeque::new(),
        }
    }

    pub fn clear(&mut self) {
        self.samples.clear();
    }

    pub fn push(&mut self, from: Instant, to: Instant, tokens: usize) {
        self.samples.push_back((from, to, tokens));
        while let Some((_, end, _)) = self.samples.front() {
            if to.duration_since(*end) > self.span {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Tokens per second over the window. Zero once nothing has landed recently.
    pub fn rate(&self, now: Instant) -> f32 {
        let total: usize = self.samples.iter().map(|(_, _, n)| *n).sum();
        if total == 0 {
            return 0.0;
        }
        let oldest = self.samples.front().map(|(t, _, _)| *t).unwrap_or(now);
        let secs = now.duration_since(oldest).as_secs_f32().max(0.05);
        total as f32 / secs
    }
}

#[derive(Debug)]
pub struct PerfTracker {
    pub started: Instant,
    last_sample: Option<(Instant, LiveStats)>,
    last_gpu: Option<Instant>,
    decode_win: RateWindow,
    prefill_win: RateWindow,
    pub phase: Phase,
    /// Instantaneous (last-interval) rates.
    pub decode_tps: f32,
    pub prefill_tps: f32,
    /// Smoothed rates for the big numerals.
    pub decode_tps_smooth: f32,
    pub prefill_tps_smooth: f32,
    pub peak_decode_tps: f32,
    pub peak_prefill_tps: f32,
    pub decode_hist: VecDeque<f32>,
    pub prefill_hist: VecDeque<f32>,
    pub util_hist: Vec<VecDeque<f32>>,
    pub power_hist: Vec<VecDeque<f32>>,
    pub temp_hist: Vec<VecDeque<f32>>,
    /// Peak-hold markers for VU meters (fraction 0..1, and time set).
    pub util_peak: Vec<(f32, Instant)>,
    pub power_peak: Vec<(f32, Instant)>,
    pub current: Option<RequestRecord>,
    pub history: VecDeque<RequestRecord>,
    pub session_decoded: u64,
    pub session_prefilled: u64,
    pub session_requests: u64,
    pub total_power_w: f32,
    pub samples: u64,
    pub poll_ok: bool,
    /// Speculative decoding (MTP / draft) — only when `/metrics` is served.
    pub spec: SpecStats,
}

#[derive(Debug, Clone)]
pub struct SpecStats {
    pub available: bool,
    pub totals: SpecMetrics,
    last: Option<SpecMetrics>,
    draft_win: RateWindow,
    accept_win: RateWindow,
    steps_win: RateWindow,
    last_time: Option<Instant>,
    /// Windowed acceptance rate (0..1), and its history for the sparkline.
    pub accept_rate: f32,
    pub accept_hist: VecDeque<f32>,
    /// Windowed mean accepted draft tokens per verification step.
    pub mean_accepted: f32,
    pub steps_per_sec: f32,
    pub drafts_per_sec: f32,
}

impl SpecStats {
    fn new() -> Self {
        Self {
            available: false,
            totals: SpecMetrics::default(),
            last: None,
            draft_win: RateWindow::new(Duration::from_millis(1500)),
            accept_win: RateWindow::new(Duration::from_millis(1500)),
            steps_win: RateWindow::new(Duration::from_millis(1500)),
            last_time: None,
            accept_rate: 0.0,
            accept_hist: VecDeque::with_capacity(HISTORY),
            mean_accepted: 0.0,
            steps_per_sec: 0.0,
            drafts_per_sec: 0.0,
        }
    }

    /// Session-wide acceptance rate from the cumulative counters.
    pub fn session_accept_rate(&self) -> f32 {
        if self.totals.draft_tokens == 0 {
            0.0
        } else {
            self.totals.accepted as f32 / self.totals.draft_tokens as f32
        }
    }

    /// Fraction of generated tokens that came from accepted drafts.
    pub fn session_draft_share(&self) -> f32 {
        if self.totals.tokens_predicted == 0 {
            0.0
        } else {
            (self.totals.accepted as f32 / self.totals.tokens_predicted as f32).clamp(0.0, 1.0)
        }
    }

    fn observe(&mut self, m: &SpecMetrics, now: Instant) {
        self.available = true;
        if let (Some(prev), Some(t0)) = (&self.last, self.last_time) {
            let d = |a: u64, b: u64| a.saturating_sub(b) as usize;
            self.draft_win.push(t0, now, d(m.draft_tokens, prev.draft_tokens));
            self.accept_win.push(t0, now, d(m.accepted, prev.accepted));
            self.steps_win.push(t0, now, d(m.verify_steps, prev.verify_steps));
            let drafts = self.draft_win.rate(now);
            let acc = self.accept_win.rate(now);
            let steps = self.steps_win.rate(now);
            self.drafts_per_sec = drafts;
            self.steps_per_sec = steps;
            self.accept_rate = if drafts > 0.0 { (acc / drafts).clamp(0.0, 1.0) } else { 0.0 };
            self.mean_accepted = if steps > 0.0 { acc / steps } else { 0.0 };
            // Record only while drafting so the sparkline is a real trace.
            if drafts > 0.0 {
                push(&mut self.accept_hist, self.accept_rate);
            }
        }
        self.totals = m.clone();
        self.last = Some(m.clone());
        self.last_time = Some(now);
    }
}

impl PerfTracker {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            last_sample: None,
            last_gpu: None,
            decode_win: RateWindow::new(Duration::from_millis(1000)),
            prefill_win: RateWindow::new(Duration::from_millis(600)),
            phase: Phase::Idle,
            decode_tps: 0.0,
            prefill_tps: 0.0,
            decode_tps_smooth: 0.0,
            prefill_tps_smooth: 0.0,
            peak_decode_tps: 0.0,
            peak_prefill_tps: 0.0,
            decode_hist: VecDeque::with_capacity(HISTORY),
            prefill_hist: VecDeque::with_capacity(HISTORY),
            util_hist: Vec::new(),
            power_hist: Vec::new(),
            temp_hist: Vec::new(),
            util_peak: Vec::new(),
            power_peak: Vec::new(),
            current: None,
            history: VecDeque::with_capacity(MAX_REQUESTS),
            session_decoded: 0,
            session_prefilled: 0,
            session_requests: 0,
            total_power_w: 0.0,
            samples: 0,
            poll_ok: false,
            spec: SpecStats::new(),
        }
    }

    pub fn observe_spec(&mut self, m: &SpecMetrics, now: Instant) {
        self.spec.observe(m, now);
    }

    pub fn uptime(&self) -> Duration {
        self.started.elapsed()
    }

    /// Tokens generated per joule across all monitored GPUs.
    pub fn tokens_per_joule(&self) -> f32 {
        if self.total_power_w > 1.0 {
            self.decode_tps_smooth / self.total_power_w
        } else {
            0.0
        }
    }

    pub fn observe(&mut self, s: &LiveStats, now: Instant) {
        self.samples += 1;
        self.poll_ok = true;
        let Some((t0, prev)) = self.last_sample.clone() else {
            self.last_sample = Some((now, s.clone()));
            if s.processing {
                self.current = Some(RequestRecord::new(s.id_task, now, s));
                self.session_requests += 1;
            }
            return;
        };
        let dt = (now - t0).as_secs_f32().max(1e-3);

        let new_request = s.id_task != prev.id_task
            || (s.processing && !prev.processing)
            || (s.processing && s.decoded < prev.decoded);
        if new_request && s.processing {
            if let Some(mut done) = self.current.take() {
                done.ended = Some(now);
                self.finish(done);
            }
            self.current = Some(RequestRecord::new(s.id_task, now, s));
            self.session_requests += 1;
            self.decode_win.clear();
            self.prefill_win.clear();
        }

        // Deltas only make sense within one request.
        let same = !new_request;
        let d_dec = if same {
            s.decoded.saturating_sub(prev.decoded)
        } else {
            s.decoded
        };
        let d_pre = if same {
            s.prompt_processed.saturating_sub(prev.prompt_processed)
        } else {
            s.prompt_processed
        };

        self.phase = if !s.processing {
            Phase::Idle
        } else if d_dec > 0 || (s.decoded > 0 && d_pre == 0) {
            Phase::Decode
        } else {
            Phase::Prefill
        };

        // Windowed rates: steady under bursty speculative decoding, and they
        // fall to zero within a second of the counter stopping.
        self.decode_win.push(t0, now, d_dec);
        self.prefill_win.push(t0, now, d_pre);
        if !s.processing {
            self.decode_win.clear();
            self.prefill_win.clear();
        }
        self.decode_tps = self.decode_win.rate(now);
        self.prefill_tps = self.prefill_win.rate(now);
        let ema = |prev: f32, x: f32, up: f32, down: f32| -> f32 {
            let a = if x > prev { up } else { down };
            prev + (x - prev) * a
        };
        self.decode_tps_smooth = ema(self.decode_tps_smooth, self.decode_tps, 0.5, 0.25);
        self.prefill_tps_smooth = ema(self.prefill_tps_smooth, self.prefill_tps, 0.5, 0.25);
        if self.decode_tps_smooth < 0.05 {
            self.decode_tps_smooth = 0.0;
        }
        if self.prefill_tps_smooth < 0.5 {
            self.prefill_tps_smooth = 0.0;
        }
        self.peak_decode_tps = self.peak_decode_tps.max(self.decode_tps);
        self.peak_prefill_tps = self.peak_prefill_tps.max(self.prefill_tps);
        push(&mut self.decode_hist, self.decode_tps);
        push(&mut self.prefill_hist, self.prefill_tps);
        self.session_decoded += d_dec as u64;
        self.session_prefilled += d_pre as u64;

        if let Some(cur) = self.current.as_mut() {
            // The slot can still carry the previous request's prompt size on
            // the first sample, so always take the latest value.
            cur.prompt_tokens = s.prompt_tokens;
            cur.cached_tokens = s.cache_tokens;
            cur.decoded = cur.decoded.max(s.decoded);
            if d_pre > 0 {
                cur.prefill_tokens += d_pre;
                cur.prefill_secs += dt;
            }
            if d_dec > 0 {
                cur.decode_secs += dt;
                cur.peak_decode_tps = cur.peak_decode_tps.max(self.decode_tps);
                if cur.first_token.is_none() {
                    cur.first_token = Some(now);
                }
            }
            if !s.processing {
                let mut done = self.current.take().unwrap();
                done.ended = Some(now);
                self.finish(done);
            }
        }

        self.last_sample = Some((now, s.clone()));
    }

    fn finish(&mut self, r: RequestRecord) {
        if r.decoded == 0 && r.prefill_tokens == 0 {
            return;
        }
        if self.history.len() >= MAX_REQUESTS {
            self.history.pop_front();
        }
        self.history.push_back(r);
    }

    pub fn observe_gpu(&mut self, gpus: &[GpuStats], now: Instant) {
        let dt = self
            .last_gpu
            .map(|t| (now - t).as_secs_f32())
            .unwrap_or(0.2)
            .clamp(0.0, 2.0);
        self.last_gpu = Some(now);
        let n = gpus.iter().map(|g| g.index as usize + 1).max().unwrap_or(0);
        while self.util_hist.len() < n {
            self.util_hist.push(VecDeque::with_capacity(HISTORY));
            self.power_hist.push(VecDeque::with_capacity(HISTORY));
            self.temp_hist.push(VecDeque::with_capacity(HISTORY));
            self.util_peak.push((0.0, now));
            self.power_peak.push((0.0, now));
        }
        self.total_power_w = gpus.iter().map(|g| g.power_watts).sum();
        for g in gpus {
            let i = g.index as usize;
            push(&mut self.util_hist[i], g.utilization_gpu);
            push(&mut self.power_hist[i], g.power_watts);
            push(&mut self.temp_hist[i], g.temperature.unwrap_or(0.0));
            hold_peak(&mut self.util_peak[i], g.utilization_gpu / 100.0, now, dt);
            hold_peak(&mut self.power_peak[i], g.power_frac(), now, dt);
        }
    }

    /// Rows for the request table, newest first, live request on top.
    pub fn recent_requests(&self, n: usize) -> Vec<&RequestRecord> {
        let mut out: Vec<&RequestRecord> = Vec::with_capacity(n);
        if let Some(c) = &self.current {
            out.push(c);
        }
        out.extend(self.history.iter().rev().take(n.saturating_sub(out.len())));
        out
    }
}

/// VU-style peak hold: sits for 1.2 s then falls at 0.4/s.
fn hold_peak(peak: &mut (f32, Instant), value: f32, now: Instant, dt: f32) {
    if value >= peak.0 {
        *peak = (value, now);
    } else if (now - peak.1).as_secs_f32() > 1.2 {
        peak.0 = (peak.0 - 0.4 * dt).max(value);
    }
}

fn push(h: &mut VecDeque<f32>, v: f32) {
    if h.len() >= HISTORY {
        h.pop_front();
    }
    h.push_back(v);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slot(id: i64, processing: bool, prompt: usize, processed: usize, decoded: usize) -> LiveStats {
        LiveStats {
            ctx_max: 4096,
            prompt_tokens: prompt,
            prompt_processed: processed,
            decoded,
            processing,
            id_task: id,
            ..Default::default()
        }
    }

    #[test]
    fn derives_rates_and_request_record() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let step = Duration::from_millis(200);
        p.observe(&slot(1, false, 0, 0, 0), t0);
        p.observe(&slot(2, true, 1000, 200, 0), t0 + step);
        assert_eq!(p.phase, Phase::Prefill);
        p.observe(&slot(2, true, 1000, 600, 0), t0 + step * 2);
        assert!((p.prefill_tps - 1500.0).abs() < 1.0, "prefill {}", p.prefill_tps);
        p.observe(&slot(2, true, 1000, 1000, 0), t0 + step * 3);
        p.observe(&slot(2, true, 1000, 1000, 10), t0 + step * 4);
        assert_eq!(p.phase, Phase::Decode);
        p.observe(&slot(2, true, 1000, 1000, 20), t0 + step * 5);
        // 20 tokens landed over the last second of samples.
        assert!((p.decode_tps - 20.0).abs() < 0.5, "decode {}", p.decode_tps);
        p.observe(&slot(2, false, 1000, 0, 20), t0 + step * 6);
        assert_eq!(p.phase, Phase::Idle);
        assert!(p.current.is_none());
        let r = p.history.back().expect("finished request");
        assert_eq!(r.decoded, 20);
        assert_eq!(r.prefill_tokens, 1000);
        assert!(r.ttft().is_some());
        assert!(r.avg_decode_tps() > 40.0);
        assert_eq!(p.session_decoded, 20);
        assert_eq!(p.session_requests, 1);
    }

    #[test]
    fn rate_window_smooths_bursts_and_decays() {
        let t0 = Instant::now();
        let mut w = RateWindow::new(Duration::from_millis(1000));
        let ms = |m: u64| t0 + Duration::from_millis(m);
        for (i, n) in [8, 0, 8, 0, 8].iter().enumerate() {
            let a = i as u64 * 200;
            w.push(ms(a), ms(a + 200), *n);
        }
        // 24 tokens over the full second, not 40 during the burst samples.
        assert!((w.rate(ms(1000)) - 24.0).abs() < 0.1, "{}", w.rate(ms(1000)));
        for i in 5..12u64 {
            w.push(ms(i * 200), ms(i * 200 + 200), 0);
        }
        assert_eq!(w.rate(ms(2400)), 0.0);
    }

    #[test]
    fn spec_stats_windowed_acceptance() {
        let mut p = PerfTracker::new();
        let t0 = Instant::now();
        let m = |d: u64, a: u64, s: u64| SpecMetrics {
            draft_tokens: d,
            accepted: a,
            verify_steps: s,
            n_decode: 0,
            tokens_predicted: a + s,
        };
        p.observe_spec(&m(100, 60, 100), t0);
        p.observe_spec(&m(120, 75, 120), t0 + Duration::from_millis(200));
        p.observe_spec(&m(140, 85, 140), t0 + Duration::from_millis(400));
        assert!(p.spec.available);
        // 25 accepted of 40 drafted in the window.
        assert!((p.spec.accept_rate - 0.625).abs() < 1e-3, "{}", p.spec.accept_rate);
        assert!((p.spec.mean_accepted - 0.625).abs() < 1e-3);
        assert!((p.spec.session_accept_rate() - 85.0 / 140.0).abs() < 1e-4);
        assert_eq!(p.spec.accept_hist.len(), 2);
    }

    #[test]
    fn peak_hold_decays_after_delay() {
        let now = Instant::now();
        let mut pk = (0.0f32, now);
        hold_peak(&mut pk, 0.9, now, 0.2);
        assert_eq!(pk.0, 0.9);
        hold_peak(&mut pk, 0.2, now + Duration::from_millis(500), 0.2);
        assert_eq!(pk.0, 0.9, "held");
        hold_peak(&mut pk, 0.2, now + Duration::from_millis(2000), 0.5);
        assert!(pk.0 < 0.9 && pk.0 >= 0.2, "decayed to {}", pk.0);
    }
}
