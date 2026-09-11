use clap::Parser;
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Copy, Default)]
pub enum ViewMode {
    /// Everything on one screen.
    #[default]
    All,
    /// Throughput, GPUs and request log, enlarged.
    Perf,
    /// Layer tiles zoom.
    Heatmap,
    /// MoE expert map zoom.
    MoE,
}

impl std::fmt::Display for ViewMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ViewMode::All => write!(f, "dashboard"),
            ViewMode::Perf => write!(f, "perf"),
            ViewMode::Heatmap => write!(f, "layers"),
            ViewMode::MoE => write!(f, "moe"),
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "llm-visuals",
    about = "Real-time terminal dashboard for a locally running LLM"
)]
pub struct Args {
    /// HuggingFace model id or local path. Default `auto` observes the running LLM.
    #[arg(long, default_value = "auto")]
    pub model: String,

    /// Prompt to generate from
    #[arg(long, default_value = "Once upon a time")]
    pub prompt: String,

    /// Max tokens to generate
    #[arg(long, default_value_t = 64)]
    pub max_tokens: usize,

    /// Color theme: defrag, neon, fire, ocean, monochrome
    #[arg(long, default_value = "defrag")]
    pub theme: String,

    /// Colour depth: auto, truecolor, 256
    #[arg(long, default_value = "auto")]
    pub color: String,

    /// Sliding window of token columns to keep on screen
    #[arg(long, default_value_t = 80)]
    pub window: usize,

    /// Max layers to display (0 = all)
    #[arg(long, default_value_t = 0)]
    pub max_layers: usize,

    /// Max heads per layer to display (0 = all)
    #[arg(long, default_value_t = 0)]
    pub max_heads: usize,

    /// Path to python_bridge.py (defaults to beside the binary / src/llm/)
    #[arg(long)]
    pub bridge: Option<PathBuf>,

    /// Run a synthetic demo without a model or GPU
    #[arg(long)]
    pub demo: bool,

    /// Demo: number of layers
    #[arg(long, default_value_t = 12)]
    pub demo_layers: usize,

    /// Demo: number of heads
    #[arg(long, default_value_t = 8)]
    pub demo_heads: usize,

    /// Auto-detect the currently running LLM model on the machine
    #[arg(long)]
    pub detect_auto: bool,

    /// GPU index to monitor (comma-separated, e.g. "0,1")
    #[arg(long, default_value = "all")]
    pub gpu: String,

    /// Number of MoE experts per layer (for demo)
    #[arg(long, default_value_t = 4)]
    pub moe_experts: usize,

    /// Poll interval for the inference server and nvidia-smi, in ms
    #[arg(long, default_value_t = 200)]
    pub poll_ms: u64,
}

impl Args {
    pub fn bridge_script(&self) -> PathBuf {
        if let Some(path) = &self.bridge {
            return path.clone();
        }
        let candidates = [
            PathBuf::from("src/llm/python_bridge.py"),
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/llm/python_bridge.py"),
        ];
        for c in candidates {
            if c.exists() {
                return c;
            }
        }
        PathBuf::from("src/llm/python_bridge.py")
    }

    /// Whether to auto-detect the running model (explicit flag or "auto" value)
    pub fn auto_detect(&self) -> bool {
        self.detect_auto || self.model == "auto"
    }

    /// GPU indices to monitor. Empty means all devices.
    pub fn gpu_indices(&self) -> Vec<usize> {
        if self.gpu == "all" {
            return vec![];
        }
        self.gpu
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect()
    }
}
