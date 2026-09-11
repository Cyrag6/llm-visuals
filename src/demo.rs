//! Synthetic inference server + GPUs so the dashboard runs without hardware.

use std::path::PathBuf;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::gguf::GgufInfo;
use crate::gpu::{DemoGpu, GpuSample, GpuStats};
use crate::model_detect::DetectedModel;
use crate::observe::{ExpertLayer, ExpertStats, LiveStats, SpecMetrics};

pub fn demo_model(ctx_max: usize) -> DetectedModel {
    DetectedModel {
        name: "Qwen3-Demo-35B-A3B".into(),
        path: Some(PathBuf::from("/models/qwen3-demo-35b-a3b-Q4_K_M.gguf")),
        pid: 4242,
        process_name: "llama-server".into(),
        engine: "llama.cpp".into(),
        gpu_indices: vec![0, 1],
        mem_used_mb: 18_900,
        port: Some(8080),
        ctx_max: Some(ctx_max),
        spec_type: Some("draft-mtp".into()),
        n_gpu_layers: Some(99),
        tensor_split: vec![63.0, 37.0],
        cmdline: "llama-server --demo".into(),
        gguf: Some(GgufInfo {
            name: "Qwen3 Demo 35B A3B".into(),
            architecture: "qwen3moe".into(),
            n_layers: 41,
            n_heads: 16,
            n_kv_heads: 4,
            n_experts: 256,
            n_experts_used: 8,
            ctx_train: ctx_max,
            n_embd: 2048,
            n_mtp: 1,
        }),
    }
}

fn next_f(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 32) as u32 as f32) / (u32::MAX as f32)
}

/// One task drives slot counters and GPU walks together so they agree.
pub fn spawn(
    live_tx: mpsc::Sender<LiveStats>,
    gpu_tx: mpsc::Sender<GpuSample>,
    spec_tx: mpsc::Sender<SpecMetrics>,
    experts_tx: mpsc::Sender<ExpertStats>,
    num_gpus: usize,
    ctx_max: usize,
) {
    tokio::spawn(async move {
        let mut gpus: Vec<DemoGpu> = (0..num_gpus.max(1)).map(DemoGpu::new).collect();
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let tick = Duration::from_millis(200);
        let mut id_task: i64 = 1000;
        let mut cache_tokens: usize = 0;
        let mut spec = SpecMetrics::default();
        let model = demo_model(ctx_max);
        let gg = model.gguf.as_ref().unwrap();
        let mut experts = DemoExperts::new(gg.n_layers, gg.n_experts, gg.n_experts_used);
        let mut stats = LiveStats {
            ctx_max,
            spec_types: "none,draft-mtp".into(),
            n_slots: 1,
            ..Default::default()
        };

        loop {
            // ---- idle -------------------------------------------------
            let idle_ticks = 4 + (next_f(&mut seed) * 12.0) as usize;
            stats.processing = false;
            stats.prompt_processed = 0;
            stats.slots_busy = 0;
            for _ in 0..idle_ticks {
                if emit(&live_tx, &gpu_tx, &spec_tx, &experts_tx, &stats, &spec, &mut experts, &mut gpus, 0.02).await.is_err() {
                    return;
                }
                tokio::time::sleep(tick).await;
            }

            // ---- prefill ---------------------------------------------
            id_task += 1;
            let prompt = 400 + (next_f(&mut seed) * 9000.0) as usize;
            let cached = if next_f(&mut seed) < 0.5 {
                cache_tokens.min(prompt / 2)
            } else {
                0
            };
            let prefill_tps = 700.0 + next_f(&mut seed) * 900.0;
            stats.id_task = id_task;
            stats.processing = true;
            stats.slots_busy = 1;
            stats.prompt_tokens = prompt;
            stats.cache_tokens = cached;
            stats.prompt_processed = cached;
            stats.decoded = 0;
            while stats.prompt_processed < prompt {
                let step = (prefill_tps * 0.2 * (0.85 + next_f(&mut seed) * 0.3)) as usize;
                let before = stats.prompt_processed;
                stats.prompt_processed = (stats.prompt_processed + step.max(1)).min(prompt);
                experts.route(stats.prompt_processed - before, &mut seed);
                if emit(&live_tx, &gpu_tx, &spec_tx, &experts_tx, &stats, &spec, &mut experts, &mut gpus, 0.95).await.is_err() {
                    return;
                }
                tokio::time::sleep(tick).await;
            }

            // ---- decode ----------------------------------------------
            let gen = 40 + (next_f(&mut seed) * 500.0) as usize;
            let base_tps = 28.0 + next_f(&mut seed) * 50.0;
            let mut t = 0.0f32;
            while stats.decoded < gen {
                t += 0.2;
                let wobble = 1.0 + 0.18 * (t * 1.7).sin() + 0.08 * (t * 5.3).cos();
                let tps = base_tps * wobble * (1.0 - stats.decoded as f32 / (gen as f32 * 6.0));
                let step = (tps * 0.2).round().max(1.0) as usize;
                let before = stats.decoded;
                stats.decoded = (stats.decoded + step).min(gen);
                experts.route(stats.decoded - before, &mut seed);
                // MTP depth 1: every verification step drafts one token; the
                // acceptance rate drifts so the panel has something to show.
                let acc_rate = (0.62 + 0.25 * (t * 0.9).sin() + 0.08 * (t * 3.1).cos()).clamp(0.15, 0.95);
                let accepted = (step as f32 * acc_rate / (1.0 + acc_rate)).round() as u64;
                let steps = step as u64 - accepted;
                spec.verify_steps += steps;
                spec.draft_tokens += steps;
                spec.accepted += accepted.min(steps);
                spec.n_decode += steps;
                spec.tokens_predicted += step as u64;
                let load = 0.55 + 0.35 * (tps / (base_tps * 1.3)).clamp(0.0, 1.0);
                if emit(&live_tx, &gpu_tx, &spec_tx, &experts_tx, &stats, &spec, &mut experts, &mut gpus, load).await.is_err() {
                    return;
                }
                tokio::time::sleep(tick).await;
            }
            cache_tokens = prompt + gen;
            stats.processing = false;
            stats.prompt_processed = 0;
            stats.slots_busy = 0;
            if emit(&live_tx, &gpu_tx, &spec_tx, &experts_tx, &stats, &spec, &mut experts, &mut gpus, 0.2).await.is_err() {
                return;
            }
            tokio::time::sleep(tick).await;
        }
    });
}

/// Stand-in for the server's router: a skewed but stable preference per layer
/// (real MoEs are lopsided), refreshed one token at a time.
struct DemoExperts {
    n_expert: usize,
    k: usize,
    stats: ExpertStats,
    ring: Vec<std::collections::VecDeque<Vec<i32>>>,
}

impl DemoExperts {
    fn new(n_layers: usize, n_expert: usize, k: usize) -> Self {
        let layers = (0..n_layers)
            .map(|il| ExpertLayer {
                il,
                n_tokens: 0,
                tokens: Vec::new(),
                recent: vec![0; n_expert],
            })
            .collect();
        Self {
            n_expert,
            k,
            stats: ExpertStats {
                n_expert,
                n_expert_used: k,
                n_tokens: 0,
                window: 256,
                layers,
            },
            ring: vec![std::collections::VecDeque::new(); n_layers],
        }
    }

    fn route(&mut self, n_tokens: usize, seed: &mut u64) {
        for _ in 0..n_tokens {
            self.stats.n_tokens += 1;
            for (il, layer) in self.stats.layers.iter_mut().enumerate() {
                let mut ids: Vec<i32> = Vec::with_capacity(self.k);
                while ids.len() < self.k {
                    // Skewed draw: a third of picks land in a per-layer "favourite" band.
                    let r = next_f(seed);
                    let e = if next_f(seed) < 0.35 {
                        ((il * 37) % self.n_expert + (r * 24.0) as usize) % self.n_expert
                    } else {
                        (r * self.n_expert as f32) as usize % self.n_expert
                    } as i32;
                    if !ids.contains(&e) {
                        ids.push(e);
                    }
                }
                for &e in &ids {
                    layer.recent[e as usize] += 1;
                }
                let ring = &mut self.ring[il];
                ring.push_back(ids.clone());
                if ring.len() > 256 {
                    for e in ring.pop_front().unwrap() {
                        layer.recent[e as usize] = layer.recent[e as usize].saturating_sub(1);
                    }
                }
                layer.n_tokens += 1;
                layer.tokens.push(ids);
                if layer.tokens.len() > 16 {
                    layer.tokens.remove(0);
                }
            }
        }
    }
}

async fn emit(
    live_tx: &mpsc::Sender<LiveStats>,
    gpu_tx: &mpsc::Sender<GpuSample>,
    spec_tx: &mpsc::Sender<SpecMetrics>,
    experts_tx: &mpsc::Sender<ExpertStats>,
    stats: &LiveStats,
    spec: &SpecMetrics,
    experts: &mut DemoExperts,
    gpus: &mut [DemoGpu],
    load: f32,
) -> Result<(), ()> {
    let g: Vec<GpuStats> = gpus.iter_mut().map(|d| d.step(load)).collect();
    gpu_tx.send(Ok(g)).await.map_err(|_| ())?;
    spec_tx.send(spec.clone()).await.map_err(|_| ())?;
    experts_tx.send(experts.stats.clone()).await.map_err(|_| ())?;
    live_tx.send(stats.clone()).await.map_err(|_| ())
}
