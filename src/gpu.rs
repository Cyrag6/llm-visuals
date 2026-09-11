use std::time::Duration;
use tokio::time;

/// Statistics for a single GPU
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct GpuStats {
    pub index: u32,
    pub name: String,
    pub utilization_gpu: f32, // %
    pub utilization_mem: f32, // %
    pub mem_total_mb: u64,
    pub mem_used_mb: u64,
    pub mem_free_mb: u64,
    pub power_watts: f32,
    pub power_max_watts: f32,
    pub temperature: Option<f32>,
    pub clock_sm_mhz: u32,
    pub clock_sm_max_mhz: u32,
    pub clock_mem_mhz: u32,
    pub fan_pct: Option<f32>,
    pub pcie_gen: u32,
    pub pcie_width: u32,
}

impl GpuStats {
    pub fn vram_percent(&self) -> f32 {
        if self.mem_total_mb == 0 {
            0.0
        } else {
            (self.mem_used_mb as f32 / self.mem_total_mb as f32) * 100.0
        }
    }

    pub fn vram_gb(&self) -> f32 {
        self.mem_used_mb as f32 / 1024.0
    }

    pub fn vram_total_gb(&self) -> f32 {
        self.mem_total_mb as f32 / 1024.0
    }

    pub fn power_frac(&self) -> f32 {
        if self.power_max_watts <= 0.0 {
            0.0
        } else {
            (self.power_watts / self.power_max_watts).clamp(0.0, 1.0)
        }
    }

    pub fn clock_frac(&self) -> f32 {
        if self.clock_sm_max_mhz == 0 {
            0.0
        } else {
            (self.clock_sm_mhz as f32 / self.clock_sm_max_mhz as f32).clamp(0.0, 1.0)
        }
    }

    /// Short marketing name: "GeForce GTX 1070" → "GTX 1070", "Tesla P100-PCIE-12GB" → "P100".
    pub fn short_name(&self) -> String {
        let n = self
            .name
            .trim_start_matches("NVIDIA ")
            .trim_start_matches("GeForce ")
            .trim_start_matches("Tesla ");
        let n = n.split("-PCIE").next().unwrap_or(n);
        let n = n.split("-SXM").next().unwrap_or(n);
        n.trim().to_string()
    }
}

/// One poll: the stats, or the reason nvidia-smi gave nothing.
pub type GpuSample = Result<Vec<GpuStats>, String>;

/// Collects GPU stats from nvidia-smi (CSV format) every 200ms.
pub struct GpuMonitor {
    interval: Duration,
}

const QUERY: &str = "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,power.draw,power.limit,temperature.gpu,clocks.sm,clocks.max.sm,clocks.mem,fan.speed,pcie.link.gen.current,pcie.link.width.current";

impl GpuMonitor {
    pub fn new() -> Self {
        Self {
            interval: Duration::from_millis(200),
        }
    }

    /// Run the monitor loop, sending updated stats to the channel.
    /// `filter` empty means all GPUs; otherwise only matching `index` values.
    pub async fn run(self, tx: tokio::sync::mpsc::Sender<GpuSample>, filter: Vec<usize>) {
        let mut interval = time::interval(self.interval);
        loop {
            interval.tick().await;
            let sample = match tokio::task::spawn_blocking(Self::collect).await {
                Ok(Ok(stats)) => Ok(filter_gpus(stats, &filter)),
                Ok(Err(e)) => Err(e),
                Err(e) => Err(format!("nvidia-smi task failed: {e}")),
            };
            if tx.send(sample).await.is_err() {
                break;
            }
        }
    }

    /// Parse nvidia-smi CSV output into GpuStats.
    pub fn collect() -> Result<Vec<GpuStats>, String> {
        let output = std::process::Command::new("nvidia-smi")
            .args([QUERY, "--format=csv,noheader,nounits"])
            .output()
            .map_err(|e| format!("Failed to run nvidia-smi: {e}"))?;

        if !output.status.success() {
            // First non-empty line of either stream is the human-readable reason
            // ("Failed to initialize NVML: Driver/library version mismatch", ...).
            let msg = [output.stderr.as_slice(), output.stdout.as_slice()]
                .iter()
                .flat_map(|b| String::from_utf8_lossy(b).lines().map(str::to_string).collect::<Vec<_>>())
                .find(|l| !l.trim().is_empty())
                .unwrap_or_else(|| format!("nvidia-smi exited with {}", output.status));
            return Err(msg);
        }

        Ok(parse_csv(&String::from_utf8_lossy(&output.stdout)))
    }

    /// Single-shot collect for initial stats.
    pub fn collect_once() -> Result<Vec<GpuStats>, String> {
        Self::collect()
    }
}

pub fn parse_csv(stdout: &str) -> Vec<GpuStats> {
    let mut stats = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if parts.len() < 10 {
            continue;
        }
        let num = |i: usize| -> f32 { parts.get(i).and_then(|s| s.parse().ok()).unwrap_or(0.0) };
        let int = |i: usize| -> u64 { num(i).max(0.0) as u64 };
        let opt = |i: usize| -> Option<f32> { parts.get(i).and_then(|s| s.parse().ok()) };

        stats.push(GpuStats {
            index: int(0) as u32,
            name: parts[1].to_string(),
            mem_total_mb: int(2),
            mem_used_mb: int(3),
            mem_free_mb: int(4),
            utilization_gpu: num(5),
            utilization_mem: num(6),
            power_watts: num(7),
            power_max_watts: num(8).max(1.0),
            temperature: opt(9),
            clock_sm_mhz: int(10) as u32,
            clock_sm_max_mhz: int(11) as u32,
            clock_mem_mhz: int(12) as u32,
            fan_pct: opt(13),
            pcie_gen: int(14) as u32,
            pcie_width: int(15) as u32,
        });
    }
    stats
}

pub fn filter_gpus(stats: Vec<GpuStats>, filter: &[usize]) -> Vec<GpuStats> {
    if filter.is_empty() {
        stats
    } else {
        stats
            .into_iter()
            .filter(|g| filter.contains(&(g.index as usize)))
            .collect()
    }
}

fn next_f(seed: &mut u64) -> f32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*seed >> 32) as u32 as f32) / (u32::MAX as f32)
}

/// Demo GPU: a smooth random walk so meters breathe instead of flicker.
pub struct DemoGpu {
    index: usize,
    seed: u64,
    util: f32,
    power: f32,
    temp: f32,
    clock: f32,
    mem_used: f32,
}

impl DemoGpu {
    pub fn new(index: usize) -> Self {
        Self {
            index,
            seed: (index as u64 + 1).wrapping_mul(6364136223846793005),
            util: 20.0,
            power: 0.3,
            temp: 45.0,
            clock: 0.5,
            mem_used: 0.0,
        }
    }

    /// `load` in 0..1 is the inference activity the walk is pulled toward.
    pub fn step(&mut self, load: f32) -> GpuStats {
        let names = [
            "NVIDIA GeForce GTX 1070",
            "Tesla P100-PCIE-12GB",
            "NVIDIA A100-SXM4-80GB",
        ];
        let (mem_total, pmax, cmax): (u64, f32, u32) = match self.index {
            0 => (8192, 151.0, 1911),
            1 => (12288, 250.0, 1328),
            _ => (81920, 400.0, 1410),
        };
        let jitter = |s: &mut u64, k: f32| (next_f(s) - 0.5) * k;
        let target_util = (load * 92.0 + 3.0).clamp(0.0, 100.0);
        self.util += (target_util - self.util) * 0.35 + jitter(&mut self.seed, 14.0);
        self.util = self.util.clamp(0.0, 100.0);
        let target_power = 0.18 + 0.8 * (self.util / 100.0);
        self.power += (target_power - self.power) * 0.25 + jitter(&mut self.seed, 0.04);
        self.power = self.power.clamp(0.05, 1.0);
        let target_temp = 42.0 + 38.0 * self.power;
        self.temp += (target_temp - self.temp) * 0.03 + jitter(&mut self.seed, 0.3);
        let target_clock = if self.util > 8.0 { 0.97 } else { 0.35 };
        self.clock += (target_clock - self.clock) * 0.4 + jitter(&mut self.seed, 0.02);
        self.clock = self.clock.clamp(0.1, 1.0);
        let weights = mem_total as f32 * 0.68;
        let target_mem = weights + mem_total as f32 * 0.22 * load.max(0.15);
        self.mem_used += (target_mem - self.mem_used) * 0.2;
        let mem_used_mb = self.mem_used.clamp(0.0, mem_total as f32) as u64;
        GpuStats {
            index: self.index as u32,
            name: names
                .get(self.index)
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("GPU {}", self.index)),
            utilization_gpu: self.util,
            utilization_mem: self.util * 0.6,
            mem_total_mb: mem_total,
            mem_used_mb,
            mem_free_mb: mem_total.saturating_sub(mem_used_mb),
            power_watts: self.power * pmax,
            power_max_watts: pmax,
            temperature: Some(self.temp),
            clock_sm_mhz: (self.clock * cmax as f32) as u32,
            clock_sm_max_mhz: cmax,
            clock_mem_mhz: if self.index == 1 { 715 } else { 3802 },
            fan_pct: if self.index == 1 {
                None
            } else {
                Some((20.0 + 60.0 * self.power).clamp(0.0, 100.0))
            },
            pcie_gen: 3,
            pcie_width: if self.index == 0 { 8 } else { 16 },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_real_nvidia_smi_line() {
        let csv = "0, NVIDIA GeForce GTX 1070, 8192, 7885, 307, 26, 12, 148.04, 151.00, 61, 1873, 1911, 3802, 29, 3, 8\n\
                   1, Tesla P100-PCIE-12GB, 12288, 11799, 489, 32, 5, 46.44, 250.00, 53, 1189, 1328, 715, [N/A], 3, 16\n";
        let s = parse_csv(csv);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].clock_sm_mhz, 1873);
        assert_eq!(s[0].clock_sm_max_mhz, 1911);
        assert_eq!(s[0].fan_pct, Some(29.0));
        assert_eq!(s[0].pcie_width, 8);
        assert_eq!(s[1].fan_pct, None);
        assert_eq!(s[1].short_name(), "P100");
        assert_eq!(s[0].short_name(), "GTX 1070");
    }

    #[test]
    fn demo_walk_is_bounded() {
        let mut g = DemoGpu::new(1);
        for _ in 0..200 {
            let s = g.step(0.9);
            assert!((0.0..=100.0).contains(&s.utilization_gpu));
            assert!(s.mem_used_mb <= s.mem_total_mb);
            assert!(s.power_watts <= s.power_max_watts);
        }
        assert!(g.step(0.9).utilization_gpu > 50.0);
    }

    #[test]
    fn filter_gpus_empty_keeps_all() {
        let stats = vec![DemoGpu::new(0).step(0.5), DemoGpu::new(1).step(0.5)];
        assert_eq!(filter_gpus(stats.clone(), &[]).len(), 2);
        assert_eq!(filter_gpus(stats, &[1]).len(), 1);
    }
}
