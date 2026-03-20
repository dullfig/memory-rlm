//! Local model inference via Candle.
//!
//! Loads quantized GGUF models and runs inference locally. Uses HuggingFace Hub
//! for model downloading and caching.

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor, D};
use candle_core::quantized::gguf_file;
use candle_transformers::models::quantized_qwen2 as qlm;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// GitHub release asset names for bundled models.
/// Attach these files to your GitHub releases for self-hosted distribution.
const MODELS: &[ModelSpec] = &[
    ModelSpec {
        name: "qwen2.5-3b",
        gguf_asset: "model-qwen2.5-3b-q4km.gguf",
        tokenizer_asset: "tokenizer-qwen2.5-3b.json",
        min_vram_gb: 4.0,
        hf_repo: "Qwen/Qwen2.5-3B-Instruct-GGUF",
        hf_file: "qwen2.5-3b-instruct-q4_k_m.gguf",
        hf_tokenizer_repo: "Qwen/Qwen2.5-3B-Instruct",
    },
    ModelSpec {
        name: "qwen2.5-0.5b",
        gguf_asset: "model-qwen2.5-0.5b-q4km.gguf",
        tokenizer_asset: "tokenizer-qwen2.5-0.5b.json",
        min_vram_gb: 0.0,
        hf_repo: "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        hf_file: "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        hf_tokenizer_repo: "Qwen/Qwen2.5-0.5B-Instruct",
    },
];

pub struct ModelSpec {
    pub name: &'static str,
    pub gguf_asset: &'static str,
    pub tokenizer_asset: &'static str,
    pub min_vram_gb: f64,
    pub hf_repo: &'static str,
    pub hf_file: &'static str,
    pub hf_tokenizer_repo: &'static str,
}

const GITHUB_REPO: &str = "dullfig/memory-rlm";

/// A locally-loaded quantized model for inference.
pub struct LocalModel {
    model: qlm::ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    eos_token_id: u32,
}

impl LocalModel {
    /// Load a GGUF model and its tokenizer from disk.
    pub fn load(model_path: &Path, tokenizer_path: &Path) -> Result<Self> {
        let device = Device::Cpu;

        let mut file = std::fs::File::open(model_path)
            .map_err(|e| anyhow!("Cannot open model file {}: {}", model_path.display(), e))?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow!("Failed to read GGUF: {}", e))?;
        let model = qlm::ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| anyhow!("Failed to load model weights: {}", e))?;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;

        // Find EOS token — try ChatML end token first, then common alternatives
        let eos_token_id = tokenizer
            .token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(2);

        Ok(Self {
            model,
            tokenizer,
            device,
            eos_token_id,
        })
    }

    /// Run inference with a system + user prompt in ChatML format.
    pub fn complete(
        &mut self,
        system: &str,
        user_message: &str,
        max_tokens: usize,
    ) -> Result<String> {
        let prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system, user_message
        );
        self.generate(&prompt, max_tokens)
    }

    /// Generate text from a raw prompt string.
    fn generate(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let prompt_tokens = encoding.get_ids().to_vec();

        if prompt_tokens.is_empty() {
            return Ok(String::new());
        }

        // Process prompt (prefill)
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self
            .model
            .forward(&input, 0)
            .map_err(|e| anyhow!("Prefill forward pass failed: {}", e))?;
        let logits = logits.squeeze(0)?;
        let mut next_token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;

        let mut generated_tokens = Vec::new();
        if next_token != self.eos_token_id {
            generated_tokens.push(next_token);
        }

        // Generate tokens one at a time (decode)
        let mut pos = prompt_tokens.len();
        for _ in 1..max_tokens {
            if next_token == self.eos_token_id {
                break;
            }

            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, pos)
                .map_err(|e| anyhow!("Decode forward pass failed: {}", e))?;
            pos += 1;

            let logits = logits.squeeze(0)?;
            next_token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;

            if next_token == self.eos_token_id {
                break;
            }
            generated_tokens.push(next_token);
        }

        let output = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| anyhow!("Decoding failed: {}", e))?;

        Ok(output)
    }

    /// Run a quick benchmark: generate `num_tokens` tokens and return tok/s.
    pub fn benchmark(&mut self, num_tokens: usize) -> Result<f64> {
        let prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nList 5 colors.<|im_end|>\n<|im_start|>assistant\n";

        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let prompt_tokens = encoding.get_ids().to_vec();

        // Prefill
        let input = Tensor::new(prompt_tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        let mut next_token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;

        // Timed decode phase
        let start = Instant::now();
        let mut generated = 0usize;
        let mut pos = prompt_tokens.len();

        for _ in 0..num_tokens {
            if next_token == self.eos_token_id {
                break;
            }

            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?;
            pos += 1;

            let logits = logits.squeeze(0)?;
            next_token = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
            generated += 1;
        }

        let elapsed = start.elapsed().as_secs_f64();
        if elapsed < 0.001 || generated == 0 {
            return Err(anyhow!("Benchmark produced no tokens"));
        }
        Ok(generated as f64 / elapsed)
    }
}

// --- Model management ---

/// Local models directory.
pub fn models_dir() -> PathBuf {
    if cfg!(windows) {
        let base = std::env::var("APPDATA").unwrap_or_else(|_| ".".into());
        PathBuf::from(base).join("memory-rlm").join("models")
    } else {
        let base = std::env::var("HOME").unwrap_or_else(|_| ".".into());
        PathBuf::from(base).join(".config").join("memory-rlm").join("models")
    }
}

/// Pick the best model for the given VRAM.
pub fn pick_model(vram_gb: f64) -> &'static ModelSpec {
    // Pick the largest model that fits
    for spec in MODELS {
        if vram_gb >= spec.min_vram_gb {
            return spec;
        }
    }
    &MODELS[MODELS.len() - 1] // smallest
}

/// Get model paths if already downloaded locally.
pub fn ensure_default_model() -> Result<(PathBuf, PathBuf)> {
    let dir = models_dir();

    // Check for any downloaded model (prefer 3B)
    for spec in MODELS {
        let model_path = dir.join(spec.gguf_asset);
        let tokenizer_path = dir.join(spec.tokenizer_asset);
        if model_path.exists() && tokenizer_path.exists() {
            return Ok((model_path, tokenizer_path));
        }
    }

    // Fallback: check HuggingFace cache (for backward compat)
    if let Ok(api) = hf_hub::api::sync::Api::new() {
        for spec in MODELS {
            if let Ok(model_path) = api.model(spec.hf_repo.to_string()).get(spec.hf_file) {
                if let Ok(tok_path) = api.model(spec.hf_tokenizer_repo.to_string()).get("tokenizer.json") {
                    return Ok((model_path, tok_path));
                }
            }
        }
    }

    Err(anyhow!("No model found. Run 'memory-rlm model download' or wait for auto-setup."))
}

/// Download model — tries GitHub releases first, falls back to HuggingFace.
pub fn download_default_model() -> Result<(PathBuf, PathBuf)> {
    // Detect GPU to pick the right model
    #[cfg(feature = "local-inference")]
    let vram_gb = crate::compute::detect_gpu()
        .map(|g| g.max_buffer_size as f64 / (1024.0 * 1024.0 * 1024.0))
        .unwrap_or(0.0);
    #[cfg(not(feature = "local-inference"))]
    let vram_gb = 0.0;

    let spec = pick_model(vram_gb);
    eprintln!("[memory-rlm] Selected model: {} (VRAM: {:.1} GB)", spec.name, vram_gb);

    download_model(spec)
}

/// Download a specific model spec.
pub fn download_model(spec: &ModelSpec) -> Result<(PathBuf, PathBuf)> {
    let dir = models_dir();
    std::fs::create_dir_all(&dir)?;

    let model_path = dir.join(spec.gguf_asset);
    let tokenizer_path = dir.join(spec.tokenizer_asset);

    // Already have it?
    if model_path.exists() && tokenizer_path.exists() {
        eprintln!("[memory-rlm] Model already downloaded: {}", model_path.display());
        return Ok((model_path, tokenizer_path));
    }

    // Try GitHub releases first
    match download_from_github(spec, &model_path, &tokenizer_path) {
        Ok(()) => {
            eprintln!("[memory-rlm] Downloaded from GitHub releases");
            return Ok((model_path, tokenizer_path));
        }
        Err(e) => {
            eprintln!("[memory-rlm] GitHub download failed ({}), trying HuggingFace...", e);
        }
    }

    // Fallback: HuggingFace
    download_from_huggingface(spec, &model_path, &tokenizer_path)?;
    Ok((model_path, tokenizer_path))
}

/// Download model assets from the latest GitHub release.
fn download_from_github(spec: &ModelSpec, model_dst: &Path, tokenizer_dst: &Path) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent("memory-rlm")
        .timeout(std::time::Duration::from_secs(600))
        .build()?;

    let url = format!("https://api.github.com/repos/{}/releases/latest", GITHUB_REPO);
    let resp: serde_json::Value = client.get(&url).send()?.json()?;

    let assets = resp["assets"].as_array()
        .ok_or_else(|| anyhow!("No assets in release"))?;

    // Find model asset
    let model_asset = assets.iter()
        .find(|a| a["name"].as_str() == Some(spec.gguf_asset))
        .ok_or_else(|| anyhow!("Model asset '{}' not found in release", spec.gguf_asset))?;

    let tokenizer_asset = assets.iter()
        .find(|a| a["name"].as_str() == Some(spec.tokenizer_asset))
        .ok_or_else(|| anyhow!("Tokenizer asset '{}' not found in release", spec.tokenizer_asset))?;

    // Download model
    let model_url = model_asset["browser_download_url"].as_str()
        .ok_or_else(|| anyhow!("No download URL for model asset"))?;
    eprintln!("[memory-rlm] Downloading {} (~2 GB)...", spec.gguf_asset);
    let data = client.get(model_url).send()?.bytes()?;
    std::fs::write(model_dst, &data)?;

    // Download tokenizer
    let tok_url = tokenizer_asset["browser_download_url"].as_str()
        .ok_or_else(|| anyhow!("No download URL for tokenizer asset"))?;
    eprintln!("[memory-rlm] Downloading {}...", spec.tokenizer_asset);
    let data = client.get(tok_url).send()?.bytes()?;
    std::fs::write(tokenizer_dst, &data)?;

    Ok(())
}

/// Fallback: download from HuggingFace Hub.
fn download_from_huggingface(spec: &ModelSpec, model_dst: &Path, tokenizer_dst: &Path) -> Result<()> {
    eprintln!("[memory-rlm] Downloading from HuggingFace: {}/{}", spec.hf_repo, spec.hf_file);

    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(true)
        .build()
        .map_err(|e| anyhow!("Failed to init HuggingFace API: {}", e))?;

    let hf_model = api.model(spec.hf_repo.to_string())
        .get(spec.hf_file)
        .map_err(|e| anyhow!("Failed to download model: {}", e))?;

    let hf_tokenizer = api.model(spec.hf_tokenizer_repo.to_string())
        .get("tokenizer.json")
        .map_err(|e| anyhow!("Failed to download tokenizer: {}", e))?;

    // Copy to our models dir for future use
    if !model_dst.exists() {
        std::fs::copy(&hf_model, model_dst)?;
    }
    if !tokenizer_dst.exists() {
        std::fs::copy(&hf_tokenizer, tokenizer_dst)?;
    }

    eprintln!("[memory-rlm] Model:     {}", model_dst.display());
    eprintln!("[memory-rlm] Tokenizer: {}", tokenizer_dst.display());
    Ok(())
}

/// Run a benchmark with the default model. Returns tokens per second.
/// Tries GPU first, falls back to CPU.
pub fn benchmark_default() -> Result<f64> {
    let (model_path, tokenizer_path) = ensure_default_model()?;

    // Try GPU benchmark
    if let Some((device, queue, gpu_info)) = crate::compute::init_gpu_device() {
        eprintln!("[memory-rlm] Loading model to GPU ({})...", gpu_info.name);
        match crate::wgpu_inference::WgpuInference::load(
            &model_path, &tokenizer_path, device, queue,
        ) {
            Ok(mut gpu) => {
                eprintln!("[memory-rlm] Running GPU benchmark (20 tokens)...");
                return gpu.benchmark(20);
            }
            Err(e) => eprintln!("[memory-rlm] GPU init failed, CPU fallback: {}", e),
        }
    }

    // CPU fallback
    eprintln!("[memory-rlm] Loading model for CPU benchmark...");
    let mut model = LocalModel::load(&model_path, &tokenizer_path)?;
    eprintln!("[memory-rlm] Running CPU benchmark (20 tokens)...");
    model.benchmark(20)
}

/// Benchmark with explicit model paths (used by auto-setup).
pub fn benchmark_with_paths(model_path: &std::path::Path, tokenizer_path: &std::path::Path) -> Result<f64> {
    if let Some((device, queue, gpu_info)) = crate::compute::init_gpu_device() {
        eprintln!("[memory-rlm] Benchmarking on GPU ({})...", gpu_info.name);
        match crate::wgpu_inference::WgpuInference::load(model_path, tokenizer_path, device, queue) {
            Ok(mut gpu) => return gpu.benchmark(20),
            Err(e) => eprintln!("[memory-rlm] GPU init failed, CPU fallback: {}", e),
        }
    }
    let mut model = LocalModel::load(model_path, tokenizer_path)?;
    model.benchmark(20)
}

/// Run inference with the default model, using GPU if available.
pub fn complete_local(system: &str, user_message: &str, max_tokens: usize) -> Result<String> {
    let (model_path, tokenizer_path) = ensure_default_model()?;

    // Try GPU inference first
    if let Some((device, queue, gpu_info)) = crate::compute::init_gpu_device() {
        eprintln!("[memory-rlm] Using GPU: {} ({})", gpu_info.name, gpu_info.backend);
        match crate::wgpu_inference::WgpuInference::load(
            &model_path, &tokenizer_path, device, queue,
        ) {
            Ok(mut gpu) => return gpu.complete(system, user_message, max_tokens),
            Err(e) => eprintln!("[memory-rlm] GPU init failed, CPU fallback: {}", e),
        }
    }

    // CPU fallback
    let mut model = LocalModel::load(&model_path, &tokenizer_path)?;
    model.complete(system, user_message, max_tokens)
}
