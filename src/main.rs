mod colors;
mod config;
mod demo;
mod fade;
mod gguf;
mod gpu;
mod llm;
mod model_detect;
mod observe;
mod perf;
mod pipeline;
mod render;

use clap::Parser;
use config::{Args, ViewMode};
use crossterm::{
    event::{self, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen},
};
use fade::{FadeSample, FadeState};
use gguf::layer_device;
use gpu::{GpuMonitor, GpuSample, GpuStats};
use model_detect::DetectedModel;
use observe::{ExpertStats, LiveStats, SpecMetrics};
use perf::PerfTracker;
use pipeline::{ActivityAggregator, GeneratedText, TokenBuffer};
use ratatui::{backend::CrosstermBackend, Terminal};
use render::{Dashboard, Renderer};
use std::io;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

const DEMO_CTX: usize = 32_768;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    colors::init_color_mode(&args.color);
    let mut theme_name = args.theme.clone();
    let theme = colors::get_theme(&theme_name);

    let mut detected_model = if args.demo {
        Some(demo::demo_model(DEMO_CTX))
    } else {
        model_detect::detect_models().into_iter().next()
    };

    let moe_experts = detected_model
        .as_ref()
        .map(|d| d.n_experts_used())
        .filter(|n| *n > 0)
        .unwrap_or(args.moe_experts);
    let mut renderer = Renderer::new(theme, args.max_layers, args.max_heads, moe_experts);

    crossterm::terminal::enable_raw_mode()?;
    execute!(io::stdout(), EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    struct TerminalGuard {
        restored: bool,
    }
    impl Drop for TerminalGuard {
        fn drop(&mut self) {
            if !self.restored {
                let _ = crossterm::terminal::disable_raw_mode();
                let _ = execute!(io::stdout(), LeaveAlternateScreen);
            }
        }
    }
    let mut _guard = TerminalGuard { restored: false };

    // Python bridge path (explicit --model): streams real attention weights.
    let bridge_mode = !args.demo && !args.auto_detect();
    let mut _bridge: Option<llm::PythonBridge> = None;
    let mut events_rx = if bridge_mode {
        let (bridge, rx) = llm::PythonBridge::spawn(
            &args.model,
            &args.prompt,
            args.max_tokens,
            args.bridge_script(),
        )
        .await?;
        _bridge = Some(bridge);
        rx
    } else {
        let (_tx, rx) = mpsc::channel::<llm::LlmEvent>(1);
        rx
    };

    let mut attention = TokenBuffer::new(args.window);
    let mut aggregator = ActivityAggregator::new(0, 0);
    let mut generated = GeneratedText::new();
    let mut num_layers: usize = detected_model.as_ref().map(|d| d.n_layers()).unwrap_or(0);
    let mut num_heads: usize = detected_model.as_ref().map(|d| d.n_heads()).unwrap_or(0);
    let mut ctx_max: usize = detected_model
        .as_ref()
        .and_then(|d| d.ctx_max)
        .or_else(|| detected_model.as_ref().and_then(|d| d.gguf.as_ref().map(|g| g.ctx_train)))
        .unwrap_or(0);

    let (gpu_tx, mut gpu_rx) = mpsc::channel::<GpuSample>(64);
    let (live_tx, mut live_rx) = mpsc::channel::<LiveStats>(64);
    let (spec_tx, mut spec_rx) = mpsc::channel::<SpecMetrics>(64);
    let (experts_tx, mut experts_rx) = mpsc::channel::<ExpertStats>(16);
    let (port_tx, mut port_rx) =
        tokio::sync::watch::channel(detected_model.as_ref().and_then(|d| d.port));
    let gpu_filter = args.gpu_indices();
    let poll = Duration::from_millis(args.poll_ms.max(50));

    if args.demo {
        let n = if gpu_filter.is_empty() { 2 } else { gpu_filter.len().max(1) };
        demo::spawn(live_tx, gpu_tx, spec_tx, experts_tx, n, DEMO_CTX);
    } else {
        tokio::spawn(async move {
            GpuMonitor::new().run(gpu_tx, gpu_filter).await;
        });
        tokio::spawn(async move {
            let mut port = *port_rx.borrow();
            let mut metrics_ok = true;
            let mut metrics_misses = 0u32;
            let mut experts_ok = true;
            let mut experts_misses = 0u32;
            loop {
                if let Some(p) = port {
                    if let Some(stats) = observe::poll_llama(p).await {
                        let _ = live_tx.try_send(stats);
                    }
                    // Draft/MTP counters live on /metrics; skip once we know
                    // the server was started without --metrics.
                    if metrics_ok {
                        match observe::poll_metrics(p).await {
                            Some(m) => {
                                let _ = spec_tx.try_send(m);
                            }
                            None => metrics_misses += 1,
                        }
                        if metrics_misses >= 3 {
                            metrics_ok = false;
                        }
                    }
                    // Real MoE routing needs the patched server (--expert-stats).
                    if experts_ok {
                        match observe::poll_experts(p).await {
                            Some(e) => {
                                let _ = experts_tx.try_send(e);
                            }
                            None => experts_misses += 1,
                        }
                        if experts_misses >= 3 {
                            experts_ok = false;
                        }
                    }
                }
                tokio::select! {
                    changed = port_rx.changed() => {
                        if changed.is_err() { break; }
                        port = *port_rx.borrow();
                        metrics_ok = true;
                        metrics_misses = 0;
                        experts_ok = true;
                        experts_misses = 0;
                    }
                    _ = tokio::time::sleep(poll) => {}
                }
            }
        });
    }

    let mut latest_gpu: Vec<GpuStats> = Vec::new();
    let mut gpu_error: Option<String> = None;
    if !args.demo {
        match GpuMonitor::collect_once() {
            Ok(stats) => latest_gpu = gpu::filter_gpus(stats, &args.gpu_indices()),
            Err(e) => gpu_error = Some(e),
        }
    }

    let mut live = LiveStats {
        ctx_max,
        ..Default::default()
    };
    let mut perf = PerfTracker::new();
    let mut expert_stats: Option<ExpertStats> = None;
    let mut experts_seen: u64 = 0;
    let mut fade = FadeState::new();
    let mut view_mode = ViewMode::All;
    let mut running = true;
    let mut done = false;
    let mut status = if args.demo {
        "demo: synthetic server, two synthetic GPUs".to_string()
    } else if let Some(d) = &detected_model {
        format!("attached to {d}")
    } else {
        "no running LLM found (nvidia-smi / llama-server / ollama) — press r to rescan".into()
    };

    loop {
        tokio::task::yield_now().await;
        if event::poll(Duration::from_millis(0))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => break,
                        KeyCode::Char('a') => view_mode = ViewMode::All,
                        KeyCode::Char('p') => view_mode = ViewMode::Perf,
                        KeyCode::Char('h') => view_mode = ViewMode::Heatmap,
                        KeyCode::Char('m') => view_mode = ViewMode::MoE,
                        KeyCode::Char('r') if !args.demo => {
                            let found = model_detect::detect_models();
                            detected_model = found.first().cloned();
                            let _ = port_tx.send(detected_model.as_ref().and_then(|d| d.port));
                            if let Some(d) = &detected_model {
                                num_layers = d.n_layers();
                                num_heads = d.n_heads();
                                renderer.moe_experts = d.n_experts_used().max(1);
                                if let Some(c) = d.ctx_max {
                                    ctx_max = c;
                                }
                                status = format!("re-scanned: attached to {d}");
                            } else {
                                status = "re-scan: no running LLMs".into();
                            }
                        }
                        KeyCode::Char('t') => {
                            theme_name = colors::next_theme_name(&theme_name).to_string();
                            renderer.theme = colors::get_theme(&theme_name);
                        }
                        _ => {}
                    }
                }
            }
        }

        let now = Instant::now();
        let mut gpu_updated = false;
        while let Ok(sample) = gpu_rx.try_recv() {
            match sample {
                Ok(stats) => {
                    latest_gpu = stats;
                    gpu_error = None;
                    gpu_updated = true;
                }
                Err(e) => gpu_error = Some(e),
            }
        }
        if gpu_updated {
            perf.observe_gpu(&latest_gpu, now);
        }
        let mut routing: Option<Vec<(usize, Vec<Vec<i32>>)>> = None;
        while let Ok(e) = experts_rx.try_recv() {
            // Only the routings that arrived since the previous poll flash.
            let new_tokens = e.n_tokens.saturating_sub(experts_seen) as usize;
            experts_seen = e.n_tokens;
            let mut upd: Vec<(usize, Vec<Vec<i32>>)> = Vec::with_capacity(e.layers.len());
            for l in &e.layers {
                let take = new_tokens.min(l.tokens.len());
                if take > 0 {
                    upd.push((l.il, l.tokens[l.tokens.len() - take..].to_vec()));
                }
            }
            routing = Some(upd);
            expert_stats = Some(e);
        }
        while let Ok(m) = spec_rx.try_recv() {
            perf.observe_spec(&m, now);
        }
        while let Ok(s) = live_rx.try_recv() {
            if s.ctx_max > 0 {
                ctx_max = s.ctx_max;
            }
            perf.observe(&s, now);
            live = s;
        }

        while let Ok(event) = events_rx.try_recv() {
            match event {
                llm::LlmEvent::ModelInfo {
                    num_layers: nl,
                    num_heads: nh,
                    ctx_max: cm,
                    model: ref m,
                } => {
                    num_layers = nl;
                    num_heads = nh;
                    if cm > 0 {
                        ctx_max = cm;
                    }
                    aggregator = ActivityAggregator::new(nl, nh);
                    status = format!("loaded {m} ({nl}L × {nh}H, ctx {ctx_max})");
                }
                llm::LlmEvent::Attention(weight) => aggregator.process(weight),
                llm::LlmEvent::Token { index, text } => {
                    generated.push(index, text);
                    if let Some(col) = aggregator.finalize(index) {
                        attention.push(col);
                    }
                }
                llm::LlmEvent::Status(msg) => status = msg,
                llm::LlmEvent::Done { tokens_generated } => {
                    status = format!("done — generated {tokens_generated} tokens");
                    done = true;
                }
                llm::LlmEvent::Error(msg) => {
                    status = format!("error: {msg}");
                    running = false;
                }
            }
        }

        if live.ctx_max == 0 {
            live.ctx_max = ctx_max;
        }
        let mut sample = fade_sample_from_live(detected_model.as_ref(), &latest_gpu, &live, fade::KV_BUCKETS);
        sample.routing = routing;
        fade.tick(&sample);

        let dash = Dashboard {
            detected: detected_model.as_ref(),
            gpus: &latest_gpu,
            gpu_error: gpu_error.as_deref(),
            fade: &fade,
            perf: &perf,
            live: &live,
            attention: &attention,
            generated: &generated,
            num_layers,
            num_heads,
            view: view_mode,
            status: &status,
            theme_name: &theme_name,
            demo: args.demo,
            experts: expert_stats.as_ref(),
        };
        renderer.render_frame(&mut terminal, &dash);

        if !running && !done {
            break;
        }
        tokio::time::sleep(Duration::from_millis(33)).await;
    }

    crossterm::terminal::disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen)?;
    _guard.restored = true;
    Ok(())
}

fn fade_sample_from_live(
    detected: Option<&DetectedModel>,
    gpu: &[GpuStats],
    live: &LiveStats,
    kv_buckets: usize,
) -> FadeSample {
    let n_layers = detected.map(|d| d.n_layers()).unwrap_or(0);
    let n_exp_used = detected.map(|d| d.n_experts_used()).unwrap_or(1).max(1);
    let n_exp_total = detected
        .map(|d| d.n_experts())
        .unwrap_or(0)
        .max(n_exp_used)
        .max(1);
    let split = detected.map(|d| d.tensor_split.clone()).unwrap_or_default();
    let n_gpus = gpu.iter().map(|g| g.index as usize + 1).max().unwrap_or(1);
    let mut util = vec![0.0f32; n_gpus.max(8)];
    let mut vram = vec![0.0f32; n_gpus.max(8)];
    for g in gpu {
        let i = g.index as usize;
        if i < util.len() {
            util[i] = g.utilization_gpu;
            vram[i] = g.vram_percent();
        }
    }
    let n_layers = n_layers.max(1);
    let layer_gpu: Vec<usize> = (0..n_layers)
        .map(|l| layer_device(l, n_layers, &split))
        .collect();
    let processing = live.processing;
    let layer_target: Vec<f32> = (0..n_layers)
        .map(|l| {
            let dev = layer_gpu[l];
            let u = util.get(dev).copied().unwrap_or(0.0) / 100.0;
            if processing {
                u.clamp(0.08, 1.0)
            } else {
                0.0
            }
        })
        .collect();
    let ctx_max = live.ctx_max.max(1);
    let used = live.ctx_used();
    let kv_filled: Vec<bool> = (0..kv_buckets)
        .map(|i| ((i as f32 + 0.5) / kv_buckets as f32 * ctx_max as f32) as usize <= used)
        .collect();
    let file_mb = detected
        .and_then(|d| d.path.as_ref())
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len() / (1024 * 1024))
        .or_else(|| detected.map(|d| d.mem_used_mb))
        .unwrap_or(0);
    let split_sum: f32 = split.iter().copied().sum::<f32>().max(1.0);
    let mut weight_frac = vec![0.0f32; n_gpus];
    let mut kv_alloc_frac = vec![0.0f32; n_gpus];
    for g in gpu {
        let i = g.index as usize;
        if i >= n_gpus {
            continue;
        }
        let share = if split.is_empty() {
            1.0 / gpu.len().max(1) as f32
        } else {
            split.get(i).copied().unwrap_or(0.0) / split_sum
        };
        let used_f = if g.mem_total_mb == 0 {
            0.0
        } else {
            g.mem_used_mb as f32 / g.mem_total_mb as f32
        };
        let w = if g.mem_total_mb == 0 {
            0.0
        } else {
            (file_mb as f32 * share) / g.mem_total_mb as f32
        };
        weight_frac[i] = w.clamp(0.0, used_f);
        kv_alloc_frac[i] = (used_f - weight_frac[i]).max(0.0);
    }
    FadeSample {
        layer_target,
        layer_gpu,
        kv_filled,
        processing,
        decoded: live.decoded,
        token_step: live.prompt_processed + live.decoded,
        routing: None,
        gpu_util: util.into_iter().take(n_gpus.max(1)).collect(),
        gpu_vram: vram.into_iter().take(n_gpus.max(1)).collect(),
        weight_frac,
        kv_alloc_frac,
        ctx_used: used,
        ctx_max,
        n_experts: n_exp_total,
        n_experts_used: n_exp_used,
        n_heads: detected.map(|d| d.n_heads()).unwrap_or(1).max(1),
    }
}
