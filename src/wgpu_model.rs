//! Load GGUF model weights onto GPU as dequantized f32 buffers.
//!
//! Uses candle's GGUF parser and dequantization, then uploads f32 data to
//! wgpu storage buffers. This trades VRAM for shader simplicity — the matvec
//! shader is a trivial f32 dot product instead of on-the-fly Q4_K decode.

use anyhow::{anyhow, Result};
use candle_core::quantized::gguf_file;
use candle_core::Device;
use std::path::Path;
use wgpu::util::DeviceExt;

/// Model hyperparameters extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub n_layers: usize,
    pub ffn_intermediate: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_freq_base: f32,
}

/// Weight format on GPU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightFormat {
    F16Packed, // f16 pairs as u32 (dequantized on CPU, read as unpack2x16float)
    Q4K,       // raw Q4_K blocks (dequantized on GPU in shader)
}

/// A weight matrix stored on GPU.
pub struct GpuWeight {
    pub buffer: wgpu::Buffer,
    pub rows: usize,
    pub cols: usize,
    pub format: WeightFormat,
}

/// Optional bias vector on GPU.
pub struct GpuBias {
    pub buffer: wgpu::Buffer,
    pub len: usize,
}

/// One transformer layer's weights on GPU.
pub struct GpuLayerWeights {
    pub attn_q: GpuWeight,
    pub attn_k: GpuWeight,
    pub attn_v: GpuWeight,
    pub attn_o: GpuWeight,
    pub attn_q_bias: Option<GpuBias>,
    pub attn_k_bias: Option<GpuBias>,
    pub attn_v_bias: Option<GpuBias>,
    pub attn_norm: Vec<f32>,       // CPU copy for reference
    pub attn_norm_buf: wgpu::Buffer, // GPU buffer for all-GPU forward
    pub ffn_gate: GpuWeight,
    pub ffn_down: GpuWeight,
    pub ffn_up: GpuWeight,
    pub ffn_norm: Vec<f32>,
    pub ffn_norm_buf: wgpu::Buffer,
}

/// Complete model with weights on GPU and small tensors on CPU.
pub struct GpuModel {
    pub config: ModelConfig,
    pub embedding: Vec<f32>,       // [vocab_size * d_model] — CPU (lookup only)
    pub layers: Vec<GpuLayerWeights>,
    pub output_norm: Vec<f32>,
    pub output_norm_buf: wgpu::Buffer,
    pub output_weight: GpuWeight,
}

impl GpuModel {
    /// Load a GGUF model, dequantize weights, and upload to GPU.
    pub fn from_gguf(
        model_path: &Path,
        device: &wgpu::Device,
    ) -> Result<Self> {
        let mut file = std::fs::File::open(model_path)?;
        let content = gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow!("Failed to read GGUF: {}", e))?;

        let config = extract_config(&content)?;
        eprintln!(
            "[claude-rlm] Model: {}d, {}h, {}kv, {}layers, {}ffn",
            config.d_model, config.n_heads, config.n_kv_heads,
            config.n_layers, config.ffn_intermediate
        );

        // Load embedding table (CPU only — just a table lookup)
        let embedding = load_f32_tensor(&content, &mut file, "token_embd.weight")?;

        // Load output norm
        let output_norm = load_f32_tensor(&content, &mut file, "output_norm.weight")?;
        let output_norm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("output_norm"),
            contents: bytemuck::cast_slice(&output_norm),
            usage: wgpu::BufferUsages::STORAGE,
        });

        // Load output projection (GPU)
        let output_weight = load_and_upload(&content, &mut file, device, "output.weight",
            config.vocab_size, config.d_model)?;

        // Load each transformer layer
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            eprintln!("[claude-rlm] Loading layer {}/{}...", i + 1, config.n_layers);
            let layer = load_layer(&content, &mut file, &config, device, i)?;
            layers.push(layer);
        }

        eprintln!("[claude-rlm] All weights loaded to GPU");
        Ok(Self {
            config,
            embedding,
            layers,
            output_norm,
            output_norm_buf,
            output_weight,
        })
    }
}

/// Extract model config from GGUF metadata.
fn extract_config(content: &gguf_file::Content) -> Result<ModelConfig> {
    let arch = get_meta_str(content, "general.architecture")
        .unwrap_or_else(|| "qwen2".to_string());

    let d_model = get_meta_u32(content, &format!("{}.embedding_length", arch))
        .ok_or_else(|| anyhow!("Missing {}.embedding_length", arch))? as usize;
    let n_heads = get_meta_u32(content, &format!("{}.attention.head_count", arch))
        .ok_or_else(|| anyhow!("Missing {}.attention.head_count", arch))? as usize;
    let n_kv_heads = get_meta_u32(content, &format!("{}.attention.head_count_kv", arch))
        .unwrap_or(n_heads as u32) as usize;
    let n_layers = get_meta_u32(content, &format!("{}.block_count", arch))
        .ok_or_else(|| anyhow!("Missing {}.block_count", arch))? as usize;
    let ffn_intermediate = get_meta_u32(content, &format!("{}.feed_forward_length", arch))
        .ok_or_else(|| anyhow!("Missing {}.feed_forward_length", arch))? as usize;
    let vocab_size = get_meta_u32(content, &format!("{}.vocab_size", arch))
        .or_else(|| {
            // Fall back to embedding tensor shape
            content.tensor_infos.get("token_embd.weight")
                .map(|t| t.shape.dims()[0] as u32)
        })
        .unwrap_or(151936) as usize;
    let rms_norm_eps = get_meta_f32(content, &format!("{}.attention.layer_norm_rms_epsilon", arch))
        .unwrap_or(1e-6);
    let rope_freq_base = get_meta_f32(content, &format!("{}.rope.freq_base", arch))
        .unwrap_or(1_000_000.0);
    let head_dim = d_model / n_heads;

    Ok(ModelConfig {
        d_model, n_heads, n_kv_heads, head_dim, n_layers,
        ffn_intermediate, vocab_size, rms_norm_eps, rope_freq_base,
    })
}

/// Load one transformer layer's weights.
fn load_layer(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    config: &ModelConfig,
    device: &wgpu::Device,
    layer_idx: usize,
) -> Result<GpuLayerWeights> {
    let prefix = format!("blk.{}", layer_idx);
    let kv_dim = config.n_kv_heads * config.head_dim;

    // Weight matrices → dequantize → GPU
    let attn_q = load_and_upload(content, file, device, &format!("{}.attn_q.weight", prefix),
        config.d_model, config.d_model)?;
    let attn_k = load_and_upload(content, file, device, &format!("{}.attn_k.weight", prefix),
        kv_dim, config.d_model)?;
    let attn_v = load_and_upload(content, file, device, &format!("{}.attn_v.weight", prefix),
        kv_dim, config.d_model)?;
    let attn_o = load_and_upload(content, file, device, &format!("{}.attn_output.weight", prefix),
        config.d_model, config.d_model)?;

    // Biases (optional, small → GPU)
    let attn_q_bias = try_load_bias(content, file, device, &format!("{}.attn_q.bias", prefix))?;
    let attn_k_bias = try_load_bias(content, file, device, &format!("{}.attn_k.bias", prefix))?;
    let attn_v_bias = try_load_bias(content, file, device, &format!("{}.attn_v.bias", prefix))?;

    // Norms (CPU + GPU buffers)
    let attn_norm = load_f32_tensor(content, file, &format!("{}.attn_norm.weight", prefix))?;
    let attn_norm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&attn_norm),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let ffn_norm = load_f32_tensor(content, file, &format!("{}.ffn_norm.weight", prefix))?;
    let ffn_norm_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None, contents: bytemuck::cast_slice(&ffn_norm),
        usage: wgpu::BufferUsages::STORAGE,
    });

    // FFN weights → GPU
    let ffn_gate = load_and_upload(content, file, device, &format!("{}.ffn_gate.weight", prefix),
        config.ffn_intermediate, config.d_model)?;
    let ffn_down = load_and_upload(content, file, device, &format!("{}.ffn_down.weight", prefix),
        config.d_model, config.ffn_intermediate)?;
    let ffn_up = load_and_upload(content, file, device, &format!("{}.ffn_up.weight", prefix),
        config.ffn_intermediate, config.d_model)?;

    Ok(GpuLayerWeights {
        attn_q, attn_k, attn_v, attn_o,
        attn_q_bias, attn_k_bias, attn_v_bias,
        attn_norm, attn_norm_buf,
        ffn_gate, ffn_down, ffn_up,
        ffn_norm, ffn_norm_buf,
    })
}

// --- Helpers ---

fn load_dequantized(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    name: &str,
) -> Result<Vec<f32>> {
    let qtensor = content.tensor(file, name, &Device::Cpu)
        .map_err(|e| anyhow!("Failed to load tensor '{}': {}", name, e))?;
    let tensor = qtensor.dequantize(&Device::Cpu)
        .map_err(|e| anyhow!("Failed to dequantize '{}': {}", name, e))?;
    let data = tensor.flatten_all()
        .map_err(|e| anyhow!("Failed to flatten '{}': {}", name, e))?
        .to_vec1::<f32>()
        .map_err(|e| anyhow!("Failed to read f32 data for '{}': {}", name, e))?;
    Ok(data)
}

fn load_f32_tensor(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    name: &str,
) -> Result<Vec<f32>> {
    load_dequantized(content, file, name)
}

fn load_and_upload(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    device: &wgpu::Device,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<GpuWeight> {
    // Try raw Q4_K upload (3x less VRAM, GPU dequant on-the-fly)
    if let Some(info) = content.tensor_infos.get(name) {
        let gguf_dims = info.shape.dims();
        eprintln!("[claude-rlm]   {} dtype={:?} gguf_shape={:?} expected={}x{}",
            name, info.ggml_dtype, gguf_dims, rows, cols);
        if info.ggml_dtype == candle_core::quantized::GgmlDType::Q4K && cols % 256 == 0 {
            // Q4K shader disabled pending debugging. Using f16 dequant fallback.
            // The on-the-fly dequant approach is correct (validated in Rust) but
            // the WGSL shader needs further investigation for GPU-side correctness.
        }
    }

    // Fallback: dequantize to f32, pack as f16
    let data = load_dequantized(content, file, name)?;
    if data.len() != rows * cols {
        return Err(anyhow!(
            "Shape mismatch for '{}': expected {}x{}={}, got {}",
            name, rows, cols, rows * cols, data.len()
        ));
    }
    Ok(upload_weight_f16(device, &data, rows, cols))
}

/// Read raw quantized bytes directly from the GGUF file.
fn load_raw_bytes(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    name: &str,
    rows: usize,
    cols: usize,
) -> Result<Vec<u8>> {
    use std::io::{Read, Seek, SeekFrom};

    let info = content.tensor_infos.get(name)
        .ok_or_else(|| anyhow!("Tensor '{}' not found", name))?;

    let n_blocks = (rows * cols) / 256;
    let byte_size = n_blocks * 144; // 144 bytes per Q4_K block

    let offset = content.tensor_data_offset + info.offset;
    file.seek(SeekFrom::Start(offset))?;
    let mut data = vec![0u8; byte_size];
    file.read_exact(&mut data)?;

    Ok(data)
}

fn try_load_bias(
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    device: &wgpu::Device,
    name: &str,
) -> Result<Option<GpuBias>> {
    if !content.tensor_infos.contains_key(name) {
        return Ok(None);
    }
    let data = load_dequantized(content, file, name)?;
    let len = data.len();
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(name),
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    Ok(Some(GpuBias { buffer, len }))
}

fn upload_weight_f16(device: &wgpu::Device, data: &[f32], rows: usize, cols: usize) -> GpuWeight {
    assert!(cols % 2 == 0, "cols must be even for f16 packing");
    let half_cols = cols / 2;
    let mut packed = vec![0u32; rows * half_cols];
    for r in 0..rows {
        for c in 0..half_cols {
            let idx = r * cols + c * 2;
            let lo = half::f16::from_f32(data[idx]);
            let hi = half::f16::from_f32(data[idx + 1]);
            packed[r * half_cols + c] = lo.to_bits() as u32 | ((hi.to_bits() as u32) << 16);
        }
    }
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&packed),
        usage: wgpu::BufferUsages::STORAGE,
    });
    GpuWeight { buffer, rows, cols, format: WeightFormat::F16Packed }
}

fn upload_weight_raw(device: &wgpu::Device, data: &[u8], rows: usize, cols: usize, format: WeightFormat) -> GpuWeight {
    let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: data,
        usage: wgpu::BufferUsages::STORAGE,
    });
    GpuWeight { buffer, rows, cols, format }
}

/// Validate Q4K raw bytes by dequantizing first block and comparing with candle's output.
fn validate_q4k_block(raw: &[u8], candle_vals: &[f32], name: &str) {
    if raw.len() < 144 || candle_vals.len() < 256 { return; }

    // Read block header
    let d = half::f16::from_le_bytes([raw[0], raw[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([raw[2], raw[3]]).to_f32();
    let scales = &raw[4..16];
    let qs = &raw[16..144];

    // Dequantize first 8 values (group 0, low nibbles, first 8 bytes)
    let sc = (scales[0] & 63) as f32;
    let mn = (scales[4] & 63) as f32;
    let d1 = d * sc;
    let m1 = dmin * mn;

    eprintln!("[claude-rlm]   Q4K validate '{}': d={:.6} dmin={:.6} sc={} mn={}", name, d, dmin, sc, mn);
    for i in 0..8usize {
        let nibble = (qs[i] & 0x0F) as f32;
        let my_val = d1 * nibble - m1;
        let candle_val = candle_vals[i];
        let diff = (my_val - candle_val).abs();
        if i < 4 || diff > 0.001 {
            eprintln!("[claude-rlm]     [{}] mine={:.6} candle={:.6} diff={:.6}{}",
                i, my_val, candle_val, diff, if diff > 0.001 { " MISMATCH" } else { "" });
        }
    }
}

fn get_meta_u32(content: &gguf_file::Content, key: &str) -> Option<u32> {
    content.metadata.get(key).and_then(|v| v.to_u32().ok())
}

fn get_meta_f32(content: &gguf_file::Content, key: &str) -> Option<f32> {
    content.metadata.get(key).and_then(|v| v.to_f32().ok())
}

fn get_meta_str(content: &gguf_file::Content, key: &str) -> Option<String> {
    content.metadata.get(key).and_then(|v| match v {
        gguf_file::Value::String(s) => Some(s.clone()),
        _ => None,
    })
}
