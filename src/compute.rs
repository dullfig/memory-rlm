//! GPU detection via WGPU and compute capability assessment.
//!
//! Uses WGPU to enumerate GPU adapters and determine if local model inference
//! is viable. Assessment results are cached to disk so we only benchmark once.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    /// Max buffer size in bytes (proxy for VRAM)
    pub max_buffer_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeAssessment {
    pub gpu: Option<GpuInfo>,
    pub tokens_per_second: Option<f64>,
    pub use_local: bool,
    pub reason: String,
}

/// Detect the best available GPU using WGPU.
///
/// Returns None if no GPU adapter is found or WGPU fails to initialize.
pub fn detect_gpu() -> Option<GpuInfo> {
    std::panic::catch_unwind(|| {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))?;

        let info = adapter.get_info();
        let limits = adapter.limits();
        Some(GpuInfo {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            max_buffer_size: limits.max_buffer_size,
        })
    })
    .ok()
    .flatten()
}

/// Initialize a WGPU device + queue for GPU compute.
///
/// Returns the device, queue, and GPU info, or None if no suitable GPU found.
pub fn init_gpu_device() -> Option<(wgpu::Device, wgpu::Queue, GpuInfo)> {
    std::panic::catch_unwind(|| {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            },
        ))?;

        let info = adapter.get_info();
        let limits = adapter.limits();
        let gpu_info = GpuInfo {
            name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            device_type: format!("{:?}", info.device_type),
            max_buffer_size: limits.max_buffer_size,
        };

        // Only use discrete or integrated GPUs, not software renderers
        if info.device_type == wgpu::DeviceType::Cpu {
            return None;
        }

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("inference"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits {
                    max_buffer_size: 1024 * 1024 * 1024,              // 1 GB
                    max_storage_buffer_binding_size: 1024 * 1024 * 1024, // 1 GB
                    max_compute_workgroups_per_dimension: 65535,
                    ..Default::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .ok()?;

        Some((device, queue, gpu_info))
    })
    .ok()
    .flatten()
}

/// Check cached assessment only (fast path for inference routing).
///
/// Returns a "not assessed" result if no cache exists. The user must run
/// `memory-rlm model benchmark` to populate the cache.
pub fn assess_compute(speed_threshold: f64) -> ComputeAssessment {
    if let Some(cached) = load_cached_assessment() {
        // Re-evaluate threshold in case it changed
        if let Some(tps) = cached.tokens_per_second {
            let use_local = tps >= speed_threshold;
            return ComputeAssessment {
                use_local,
                reason: if use_local {
                    format!("{:.1} tok/s >= {:.1} threshold", tps, speed_threshold)
                } else {
                    format!("{:.1} tok/s < {:.1} threshold", tps, speed_threshold)
                },
                ..cached
            };
        }
        return cached;
    }

    ComputeAssessment {
        gpu: detect_gpu(),
        tokens_per_second: None,
        use_local: false,
        reason: "Not benchmarked. Run 'memory-rlm model benchmark' to assess.".to_string(),
    }
}

/// Full assessment: detect GPU, benchmark local model, cache result.
///
/// This is the slow path — used by the `memory-rlm model benchmark` CLI command.
pub fn run_full_assessment(speed_threshold: f64) -> anyhow::Result<ComputeAssessment> {
    let gpu = detect_gpu();

    let tps = crate::local_model::benchmark_default()?;
    let use_local = tps >= speed_threshold;

    let assessment = ComputeAssessment {
        gpu,
        tokens_per_second: Some(tps),
        use_local,
        reason: if use_local {
            format!("{:.1} tok/s >= {:.1} threshold", tps, speed_threshold)
        } else {
            format!("{:.1} tok/s < {:.1} threshold", tps, speed_threshold)
        },
    };

    save_assessment(&assessment);
    Ok(assessment)
}

/// Run automatic setup if not already done:
/// 1. Detect GPU
/// 2. Pick the right model for the VRAM
/// 3. Download (GitHub releases first, HuggingFace fallback)
/// 4. Benchmark and cache result
///
/// Runs in the background during MCP server startup.
/// Returns true if setup completed (or was already done).
pub fn auto_setup() -> bool {
    // Already benchmarked? Nothing to do.
    if load_cached_assessment().is_some() {
        return true;
    }

    let gpu = match detect_gpu() {
        Some(g) => g,
        None => {
            eprintln!("[memory-rlm] No GPU detected, skipping local inference setup");
            return false;
        }
    };

    if gpu.device_type == "Cpu" {
        eprintln!("[memory-rlm] Software GPU only, skipping local inference setup");
        return false;
    }

    let vram_gb = gpu.max_buffer_size as f64 / (1024.0 * 1024.0 * 1024.0);
    eprintln!("[memory-rlm] Auto-setup: {} ({}, {:.1} GB)", gpu.name, gpu.backend, vram_gb);

    let spec = crate::local_model::pick_model(vram_gb);
    eprintln!("[memory-rlm] Auto-setup: selected model '{}'", spec.name);

    let (model_path, tokenizer_path) = match crate::local_model::download_model(spec) {
        Ok(paths) => paths,
        Err(e) => {
            eprintln!("[memory-rlm] Auto-setup: download failed: {}", e);
            return false;
        }
    };

    eprintln!("[memory-rlm] Auto-setup: benchmarking...");

    match crate::local_model::benchmark_with_paths(&model_path, &tokenizer_path) {
        Ok(tps) => {
            let threshold = 10.0;
            let assessment = ComputeAssessment {
                gpu: Some(gpu),
                tokens_per_second: Some(tps),
                use_local: tps >= threshold,
                reason: format!("{:.1} tok/s (auto-setup)", tps),
            };
            save_assessment(&assessment);
            eprintln!("[memory-rlm] Auto-setup complete: {:.1} tok/s", tps);
            true
        }
        Err(e) => {
            eprintln!("[memory-rlm] Auto-setup: benchmark failed: {}", e);
            false
        }
    }
}

/// Clear the cached assessment (forces re-benchmark on next check).
pub fn clear_cache() {
    if let Some(path) = cache_path() {
        let _ = std::fs::remove_file(path);
    }
}

// --- Cache management ---

fn cache_path() -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var("APPDATA")
            .ok()
            .map(|d| PathBuf::from(d).join("memory-rlm").join("benchmark.json"))
    } else {
        std::env::var("HOME")
            .ok()
            .map(|d| PathBuf::from(d).join(".config").join("memory-rlm").join("benchmark.json"))
    }
}

fn load_cached_assessment() -> Option<ComputeAssessment> {
    let path = cache_path()?;
    let contents = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&contents).ok()
}

fn save_assessment(assessment: &ComputeAssessment) {
    if let Some(path) = cache_path() {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_string_pretty(assessment) {
            let _ = std::fs::write(&path, json);
        }
    }
}
