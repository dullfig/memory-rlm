//! All-GPU transformer inference engine.
//!
//! Every operation runs on GPU. Per token: one command submit, one 4-byte readback.
//! The GPU processes all 24 layers + output projection as a continuous stream.

use anyhow::{anyhow, Result};
use std::path::Path;
use wgpu::util::DeviceExt;

use crate::wgpu_model::{GpuModel, GpuWeight, GpuBias};

// --- Params structs (bytemuck-compatible for GPU uniforms) ---

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatvecParams { rows: u32, cols: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsNormParams { n: u32, eps: f32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeParams { n_heads: u32, head_dim: u32, position: u32, half_dim: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarParams { n: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct KvWriteParams { kv_dim: u32, position: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnScoreParams {
    n_heads: u32, n_kv_heads: u32, head_dim: u32, seq_len: u32,
    max_seq: u32, heads_per_kv: u32, kv_dim: u32, scale: f32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct SoftmaxParams { n_heads: u32, seq_len: u32, max_seq: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnValueParams {
    n_heads: u32, n_kv_heads: u32, head_dim: u32, seq_len: u32,
    max_seq: u32, heads_per_kv: u32, kv_dim: u32, _pad: u32,
}

// --- Pipelines ---

struct Pipelines {
    // Single-token (decode)
    matvec: wgpu::ComputePipeline,
    matvec_bias: wgpu::ComputePipeline,
    matvec_q4k: wgpu::ComputePipeline,
    rmsnorm: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    silu_mul: wgpu::ComputePipeline,
    add_inplace: wgpu::ComputePipeline,
    kv_write: wgpu::ComputePipeline,
    attn_score: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    attn_value: wgpu::ComputePipeline,
    argmax: wgpu::ComputePipeline,
    // Batch (prefill)
    matmul: wgpu::ComputePipeline,
    matmul_bias: wgpu::ComputePipeline,
    rmsnorm_batch: wgpu::ComputePipeline,
    rope_batch: wgpu::ComputePipeline,
    silu_mul_batch: wgpu::ComputePipeline,
    add_inplace_batch: wgpu::ComputePipeline,
    kv_write_batch: wgpu::ComputePipeline,
    attn_score_batch: wgpu::ComputePipeline,
    softmax_batch: wgpu::ComputePipeline,
    attn_value_batch: wgpu::ComputePipeline,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchParams { rows: u32, cols: u32, n_tokens: u32, _pad: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchScalarParams { n: u32, n_tokens: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchRmsNormParams { n: u32, eps: f32, n_tokens: u32, _pad: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchRopeParams { n_heads: u32, head_dim: u32, start_pos: u32, half_dim: u32, n_tokens: u32, _p1: u32, _p2: u32, _p3: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchKvWriteParams { kv_dim: u32, start_pos: u32, n_tokens: u32, _pad: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchAttnScoreParams {
    n_heads: u32, n_kv_heads: u32, head_dim: u32, start_pos: u32,
    max_seq: u32, heads_per_kv: u32, kv_dim: u32, scale: f32,
    n_tokens: u32, _p1: u32, _p2: u32, _p3: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchSoftmaxParams { n_heads: u32, max_seq: u32, start_pos: u32, n_tokens: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchAttnValueParams {
    n_heads: u32, n_kv_heads: u32, head_dim: u32, start_pos: u32,
    max_seq: u32, heads_per_kv: u32, kv_dim: u32, n_tokens: u32,
}

// --- GPU scratch buffers ---

struct Scratch {
    hidden: wgpu::Buffer,
    normed: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    attn_out: wgpu::Buffer,
    temp: wgpu::Buffer,       // for projection results before residual add
    ffn_gate: wgpu::Buffer,
    ffn_up: wgpu::Buffer,
    ffn_mid: wgpu::Buffer,
    attn_scores: wgpu::Buffer,
    logits: wgpu::Buffer,
    token_id: wgpu::Buffer,   // u32[1]
    staging: wgpu::Buffer,    // MAP_READ for token_id readback (4 bytes)
    // KV caches on GPU
    k_caches: Vec<wgpu::Buffer>,
    v_caches: Vec<wgpu::Buffer>,
    // Precomputed cos/sin on GPU
    cos: wgpu::Buffer,
    sin: wgpu::Buffer,
    // Batch buffers (for prefill) — sized for max_batch tokens
    batch_hidden: wgpu::Buffer,
    batch_normed: wgpu::Buffer,
    batch_q: wgpu::Buffer,
    batch_k: wgpu::Buffer,
    batch_v: wgpu::Buffer,
    batch_attn_out: wgpu::Buffer,
    batch_temp: wgpu::Buffer,
    batch_ffn_gate: wgpu::Buffer,
    batch_ffn_up: wgpu::Buffer,
    batch_ffn_mid: wgpu::Buffer,
    batch_attn_scores: wgpu::Buffer,
    max_batch: usize,
}

/// All-GPU inference engine.
pub struct WgpuInference {
    device: wgpu::Device,
    queue: wgpu::Queue,
    model: GpuModel,
    tokenizer: tokenizers::Tokenizer,
    eos_token_id: u32,
    pipelines: Pipelines,
    scratch: Scratch,
    max_seq: usize,
}

impl WgpuInference {
    pub fn load(
        model_path: &Path,
        tokenizer_path: &Path,
        device: wgpu::Device,
        queue: wgpu::Queue,
    ) -> Result<Self> {
        let model = GpuModel::from_gguf(model_path, &device)?;
        let cfg = &model.config;

        let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer: {}", e))?;
        let eos_token_id = tokenizer.token_to_id("<|im_end|>")
            .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
            .or_else(|| tokenizer.token_to_id("</s>"))
            .unwrap_or(2);

        let pipelines = create_all_pipelines(&device);

        let max_seq = 2048usize;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let scratch = create_scratch(&device, cfg, max_seq, kv_dim);

        // Upload precomputed cos/sin tables
        let (cos_data, sin_data) = precompute_rope(cfg.head_dim, max_seq, cfg.rope_freq_base);
        queue.write_buffer(&scratch.cos, 0, bytemuck::cast_slice(&cos_data));
        queue.write_buffer(&scratch.sin, 0, bytemuck::cast_slice(&sin_data));

        Ok(Self { device, queue, model, tokenizer, eos_token_id, pipelines, scratch, max_seq })
    }

    pub fn complete(&mut self, system: &str, user_message: &str, max_tokens: usize) -> Result<String> {
        let prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            system, user_message
        );
        let mut result = String::new();
        self.complete_streaming(&prompt, max_tokens, |token| {
            result.push_str(&token);
        })?;
        Ok(result)
    }

    /// Generate with a per-token callback. Calls `on_token` with each decoded piece.
    /// `prompt` should be a fully formatted ChatML string.
    pub fn complete_streaming(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        mut on_token: impl FnMut(String),
    ) -> Result<()> {
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let prompt_tokens = encoding.get_ids().to_vec();
        if prompt_tokens.is_empty() { return Ok(()); }

        // Reset KV cache for new conversation
        self.reset_kv_cache();

        // Batch prefill: process all prompt tokens at once
        let mut next_token = self.forward_prefill(&prompt_tokens);

        let mut pos = prompt_tokens.len();
        for _ in 0..max_tokens {
            if next_token == self.eos_token_id { break; }
            // Decode single token and emit
            if let Ok(text) = self.tokenizer.decode(&[next_token], true) {
                on_token(text);
            }
            next_token = self.forward_token(next_token, pos);
            pos += 1;
        }

        Ok(())
    }

    /// Clear KV cache for a fresh conversation.
    fn reset_kv_cache(&self) {
        // Zero out all KV cache buffers on GPU
        let kv_dim = self.model.config.n_kv_heads * self.model.config.head_dim;
        let zeros = vec![0u8; self.max_seq * kv_dim * 4];
        for i in 0..self.model.config.n_layers {
            self.queue.write_buffer(&self.scratch.k_caches[i], 0, &zeros);
            self.queue.write_buffer(&self.scratch.v_caches[i], 0, &zeros);
        }
    }

    pub fn benchmark(&mut self, num_tokens: usize) -> Result<f64> {
        let prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nList 5 colors.<|im_end|>\n<|im_start|>assistant\n";
        let encoding = self.tokenizer.encode(prompt, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        let prompt_tokens = encoding.get_ids().to_vec();

        self.reset_kv_cache();
        let mut next_token = self.forward_prefill(&prompt_tokens);

        let start = std::time::Instant::now();
        let mut generated = 0usize;
        let mut pos = prompt_tokens.len();
        for _ in 0..num_tokens {
            if next_token == self.eos_token_id { break; }
            next_token = self.forward_token(next_token, pos);
            pos += 1;
            generated += 1;
        }
        let elapsed = start.elapsed().as_secs_f64();
        if elapsed < 0.001 || generated == 0 { return Err(anyhow!("No tokens generated")); }
        Ok(generated as f64 / elapsed)
    }

    /// Batch prefill: process all prompt tokens at once. Returns next token ID.
    /// Much faster than sequential — reads weight matrices once, not N times.
    fn forward_prefill(&self, tokens: &[u32]) -> u32 {
        let cfg = &self.model.config;
        let d = cfg.d_model;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let n = tokens.len();

        // Process in chunks of max_batch
        let mut start_pos = 0usize;
        for chunk_start in (0..n).step_by(self.scratch.max_batch) {
            let chunk_end = (chunk_start + self.scratch.max_batch).min(n);
            let chunk = &tokens[chunk_start..chunk_end];
            let nt = chunk.len();

            // Upload embeddings for all tokens in chunk
            let mut embed_data = vec![0.0f32; nt * d];
            for (i, &tok) in chunk.iter().enumerate() {
                let offset = tok as usize * d;
                embed_data[i * d..(i + 1) * d].copy_from_slice(&self.model.embedding[offset..offset + d]);
            }
            self.queue.write_buffer(&self.scratch.batch_hidden, 0, bytemuck::cast_slice(&embed_data));

            let mut encoder = self.device.create_command_encoder(&Default::default());

            for layer_idx in 0..cfg.n_layers {
                let layer = &self.model.layers[layer_idx];

                // RMSNorm
                self.dispatch_batch_rmsnorm(&mut encoder, &self.scratch.batch_hidden, &layer.attn_norm_buf, &self.scratch.batch_normed, d, nt);
                // Q/K/V
                self.dispatch_batch_matmul_maybe_bias(&mut encoder, &layer.attn_q, layer.attn_q_bias.as_ref(), &self.scratch.batch_normed, &self.scratch.batch_q, nt);
                self.dispatch_batch_matmul_maybe_bias(&mut encoder, &layer.attn_k, layer.attn_k_bias.as_ref(), &self.scratch.batch_normed, &self.scratch.batch_k, nt);
                self.dispatch_batch_matmul_maybe_bias(&mut encoder, &layer.attn_v, layer.attn_v_bias.as_ref(), &self.scratch.batch_normed, &self.scratch.batch_v, nt);
                // RoPE
                self.dispatch_batch_rope(&mut encoder, &self.scratch.batch_q, cfg.n_heads, cfg.head_dim, start_pos, nt);
                self.dispatch_batch_rope(&mut encoder, &self.scratch.batch_k, cfg.n_kv_heads, cfg.head_dim, start_pos, nt);
                // KV write
                self.dispatch_batch_kv_write(&mut encoder, layer_idx, kv_dim, start_pos, nt);
                // Attention
                self.dispatch_batch_attn_score(&mut encoder, layer_idx, cfg, start_pos, nt);
                self.dispatch_batch_softmax(&mut encoder, cfg, start_pos, nt);
                self.dispatch_batch_attn_value(&mut encoder, layer_idx, cfg, start_pos, nt);
                // O projection
                self.dispatch_batch_matmul(&mut encoder, &layer.attn_o, &self.scratch.batch_attn_out, &self.scratch.batch_temp, nt);
                // Residual
                self.dispatch_batch_add(&mut encoder, &self.scratch.batch_hidden, &self.scratch.batch_temp, d, nt);
                // FFN
                let layer = &self.model.layers[layer_idx];
                self.dispatch_batch_rmsnorm(&mut encoder, &self.scratch.batch_hidden, &layer.ffn_norm_buf, &self.scratch.batch_normed, d, nt);
                self.dispatch_batch_matmul(&mut encoder, &layer.ffn_gate, &self.scratch.batch_normed, &self.scratch.batch_ffn_gate, nt);
                self.dispatch_batch_matmul(&mut encoder, &layer.ffn_up, &self.scratch.batch_normed, &self.scratch.batch_ffn_up, nt);
                self.dispatch_batch_silu_mul(&mut encoder, cfg.ffn_intermediate, nt);
                self.dispatch_batch_matmul(&mut encoder, &layer.ffn_down, &self.scratch.batch_ffn_mid, &self.scratch.batch_temp, nt);
                self.dispatch_batch_add(&mut encoder, &self.scratch.batch_hidden, &self.scratch.batch_temp, d, nt);
            }

            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);

            start_pos += nt;
        }

        // Extract last token's hidden state → single-token buffers for output projection
        let last_offset = ((n - 1) % self.scratch.max_batch) * d;
        let copy_size = (d * 4) as u64;
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.scratch.batch_hidden, (last_offset * 4) as u64, &self.scratch.hidden, 0, copy_size);
        self.queue.submit(Some(encoder.finish()));
        self.device.poll(wgpu::Maintain::Wait);

        // Output projection + argmax on last token
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.dispatch_rmsnorm(&mut encoder, &self.scratch.hidden, &self.model.output_norm_buf, &self.scratch.normed, d);
        self.dispatch_matvec(&mut encoder, &self.model.output_weight, &self.scratch.normed, &self.scratch.logits);
        self.dispatch_argmax(&mut encoder, cfg.vocab_size);
        encoder.copy_buffer_to_buffer(&self.scratch.token_id, 0, &self.scratch.staging, 0, 4);
        self.queue.submit(Some(encoder.finish()));

        let slice = self.scratch.staging.slice(..4);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let token_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        self.scratch.staging.unmap();

        token_id
    }

    /// Run one full forward pass for a single token. Returns next token ID.
    /// Everything runs on GPU in ONE submit. Only 4 bytes read back.
    fn forward_token(&self, token: u32, pos: usize) -> u32 {
        let cfg = &self.model.config;
        let d = cfg.d_model;
        let kv_dim = cfg.n_kv_heads * cfg.head_dim;
        let seq_len = (pos + 1) as u32;

        // Upload embedding (CPU lookup → GPU)
        let offset = token as usize * d;
        let embed = &self.model.embedding[offset..offset + d];
        self.queue.write_buffer(&self.scratch.hidden, 0, bytemuck::cast_slice(embed));

        let mut encoder = self.device.create_command_encoder(&Default::default());

        for layer_idx in 0..cfg.n_layers {
            let layer = &self.model.layers[layer_idx];

            // Pass 1: RMSNorm(hidden → normed)
            self.dispatch_rmsnorm(&mut encoder, &self.scratch.hidden, &layer.attn_norm_buf, &self.scratch.normed, d);

            // Pass 2: Q + K + V projections (independent: all read normed, write different bufs)
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                self.encode_matvec_maybe_bias(&mut pass, &layer.attn_q, layer.attn_q_bias.as_ref(), &self.scratch.normed, &self.scratch.q);
                self.encode_matvec_maybe_bias(&mut pass, &layer.attn_k, layer.attn_k_bias.as_ref(), &self.scratch.normed, &self.scratch.k);
                self.encode_matvec_maybe_bias(&mut pass, &layer.attn_v, layer.attn_v_bias.as_ref(), &self.scratch.normed, &self.scratch.v);
            }
            // If Q4K, bias was skipped in combined pass — add it now
            if layer.attn_q.format == crate::wgpu_model::WeightFormat::Q4K {
                if let Some(b) = layer.attn_q_bias.as_ref() {
                    self.dispatch_add_inplace(&mut encoder, &self.scratch.q, &b.buffer, layer.attn_q.rows);
                }
                if let Some(b) = layer.attn_k_bias.as_ref() {
                    self.dispatch_add_inplace(&mut encoder, &self.scratch.k, &b.buffer, layer.attn_k.rows);
                }
                if let Some(b) = layer.attn_v_bias.as_ref() {
                    self.dispatch_add_inplace(&mut encoder, &self.scratch.v, &b.buffer, layer.attn_v.rows);
                }
            }

            // Pass 3: RoPE on Q + K (independent: different buffers)
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                self.encode_rope(&mut pass, &self.scratch.q, cfg.n_heads, cfg.head_dim, pos);
                self.encode_rope(&mut pass, &self.scratch.k, cfg.n_kv_heads, cfg.head_dim, pos);
            }

            // Pass 4: KV cache write
            self.dispatch_kv_write(&mut encoder, layer_idx, kv_dim, pos);
            // Pass 5: Attention scores
            self.dispatch_attn_score(&mut encoder, layer_idx, cfg, seq_len);
            // Pass 6: Softmax
            self.dispatch_softmax(&mut encoder, cfg.n_heads as u32, seq_len);
            // Pass 7: Attention value
            self.dispatch_attn_value(&mut encoder, layer_idx, cfg, seq_len);
            // Pass 8: O projection
            self.dispatch_matvec(&mut encoder, &layer.attn_o, &self.scratch.attn_out, &self.scratch.temp);
            // Pass 9: Residual
            self.dispatch_add_inplace(&mut encoder, &self.scratch.hidden, &self.scratch.temp, d);
            // Pass 10: FFN RMSNorm
            self.dispatch_rmsnorm(&mut encoder, &self.scratch.hidden, &layer.ffn_norm_buf, &self.scratch.normed, d);

            // Pass 11: Gate + Up (independent: both read normed, write different bufs)
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                self.encode_matvec(&mut pass, &layer.ffn_gate, &self.scratch.normed, &self.scratch.ffn_gate);
                self.encode_matvec(&mut pass, &layer.ffn_up, &self.scratch.normed, &self.scratch.ffn_up);
            }

            // Pass 12: SiLU * mul
            self.dispatch_silu_mul(&mut encoder, cfg.ffn_intermediate);
            // Pass 13: Down projection
            self.dispatch_matvec(&mut encoder, &layer.ffn_down, &self.scratch.ffn_mid, &self.scratch.temp);
            // Pass 14: Residual
            self.dispatch_add_inplace(&mut encoder, &self.scratch.hidden, &self.scratch.temp, d);
        }

        // Final: norm + output + argmax (3 passes)
        self.dispatch_rmsnorm(&mut encoder, &self.scratch.hidden, &self.model.output_norm_buf, &self.scratch.normed, d);
        self.dispatch_matvec(&mut encoder, &self.model.output_weight, &self.scratch.normed, &self.scratch.logits);
        self.dispatch_argmax(&mut encoder, cfg.vocab_size);

        // Copy token_id to staging for readback
        encoder.copy_buffer_to_buffer(&self.scratch.token_id, 0, &self.scratch.staging, 0, 4);

        // ONE submit for the entire forward pass
        self.queue.submit(Some(encoder.finish()));

        // Read back 4 bytes
        let slice = self.scratch.staging.slice(..4);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.device.poll(wgpu::Maintain::Wait);
        let data = slice.get_mapped_range();
        let token_id = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        drop(data);
        self.scratch.staging.unmap();

        token_id
    }

    // --- Dispatch helpers (each adds one compute pass to the encoder) ---

    fn dispatch_rmsnorm(&self, enc: &mut wgpu::CommandEncoder, input: &wgpu::Buffer, weight: &wgpu::Buffer, output: &wgpu::Buffer, n: usize) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&RmsNormParams { n: n as u32, eps: self.model.config.rms_norm_eps }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.rmsnorm, &[input, weight, output, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.rmsnorm);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    fn dispatch_matvec(&self, enc: &mut wgpu::CommandEncoder, w: &GpuWeight, input: &wgpu::Buffer, output: &wgpu::Buffer) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&MatvecParams { rows: w.rows as u32, cols: w.cols as u32 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let pipeline = match w.format {
            crate::wgpu_model::WeightFormat::Q4K => &self.pipelines.matvec_q4k,
            crate::wgpu_model::WeightFormat::F16Packed => &self.pipelines.matvec,
        };
        let bg = self.make_bg(pipeline, &[&w.buffer, input, output, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        let rows = w.rows as u32;
        pass.dispatch_workgroups(rows.min(65535), (rows + 65534) / 65535, 1);
    }

    fn dispatch_matvec_maybe_bias(&self, enc: &mut wgpu::CommandEncoder, w: &GpuWeight, bias: Option<&GpuBias>, input: &wgpu::Buffer, output: &wgpu::Buffer) {
        match (bias, w.format) {
            (None, _) => self.dispatch_matvec(enc, w, input, output),
            (Some(b), crate::wgpu_model::WeightFormat::F16Packed) => {
                // f16 matvec_bias shader handles bias inline
                let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None, contents: bytemuck::bytes_of(&MatvecParams { rows: w.rows as u32, cols: w.cols as u32 }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = self.make_bg(&self.pipelines.matvec_bias, &[&w.buffer, input, output, &params, &b.buffer]);
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines.matvec_bias);
                pass.set_bind_group(0, Some(&bg), &[]);
                let rows = w.rows as u32;
                pass.dispatch_workgroups(rows.min(65535), (rows + 65534) / 65535, 1);
            }
            (Some(b), crate::wgpu_model::WeightFormat::Q4K) => {
                // Q4K matvec + separate bias add
                self.dispatch_matvec(enc, w, input, output);
                self.dispatch_add_inplace(enc, output, &b.buffer, w.rows);
            }
        }
    }

    fn dispatch_rope(&self, enc: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, n_heads: usize, head_dim: usize, pos: usize) {
        let half = head_dim / 2;
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&RopeParams { n_heads: n_heads as u32, head_dim: head_dim as u32, position: pos as u32, half_dim: half as u32 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.rope, &[x, &self.scratch.cos, &self.scratch.sin, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.rope);
        pass.set_bind_group(0, Some(&bg), &[]);
        let n = (n_heads * half) as u32;
        pass.dispatch_workgroups((n + 63) / 64, 1, 1);
    }

    fn dispatch_kv_write(&self, enc: &mut wgpu::CommandEncoder, layer: usize, kv_dim: usize, pos: usize) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&KvWriteParams { kv_dim: kv_dim as u32, position: pos as u32 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.kv_write, &[&self.scratch.k, &self.scratch.v, &self.scratch.k_caches[layer], &self.scratch.v_caches[layer], &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.kv_write);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(((kv_dim + 127) / 128) as u32, 1, 1);
    }

    fn dispatch_attn_score(&self, enc: &mut wgpu::CommandEncoder, layer: usize, cfg: &crate::wgpu_model::ModelConfig, seq_len: u32) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&AttnScoreParams {
                n_heads: cfg.n_heads as u32, n_kv_heads: cfg.n_kv_heads as u32,
                head_dim: cfg.head_dim as u32, seq_len,
                max_seq: self.max_seq as u32, heads_per_kv: (cfg.n_heads / cfg.n_kv_heads) as u32,
                kv_dim: (cfg.n_kv_heads * cfg.head_dim) as u32,
                scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.attn_score, &[&self.scratch.q, &self.scratch.k_caches[layer], &self.scratch.attn_scores, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.attn_score);
        pass.set_bind_group(0, Some(&bg), &[]);
        let n = cfg.n_heads as u32 * seq_len;
        pass.dispatch_workgroups((n + 255) / 256, 1, 1);
    }

    fn dispatch_softmax(&self, enc: &mut wgpu::CommandEncoder, n_heads: u32, seq_len: u32) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&SoftmaxParams { n_heads, seq_len, max_seq: self.max_seq as u32 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.softmax, &[&self.scratch.attn_scores, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.softmax);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(n_heads, 1, 1);
    }

    fn dispatch_attn_value(&self, enc: &mut wgpu::CommandEncoder, layer: usize, cfg: &crate::wgpu_model::ModelConfig, seq_len: u32) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&AttnValueParams {
                n_heads: cfg.n_heads as u32, n_kv_heads: cfg.n_kv_heads as u32,
                head_dim: cfg.head_dim as u32, seq_len,
                max_seq: self.max_seq as u32, heads_per_kv: (cfg.n_heads / cfg.n_kv_heads) as u32,
                kv_dim: (cfg.n_kv_heads * cfg.head_dim) as u32, _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.attn_value, &[&self.scratch.attn_scores, &self.scratch.v_caches[layer], &self.scratch.attn_out, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.attn_value);
        pass.set_bind_group(0, Some(&bg), &[]);
        let n = cfg.n_heads as u32 * cfg.head_dim as u32;
        pass.dispatch_workgroups((n + 255) / 256, 1, 1);
    }

    fn dispatch_silu_mul(&self, enc: &mut wgpu::CommandEncoder, n: usize) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&ScalarParams { n: n as u32 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.silu_mul, &[&self.scratch.ffn_gate, &self.scratch.ffn_up, &self.scratch.ffn_mid, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.silu_mul);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(((n + 255) / 256) as u32, 1, 1);
    }

    fn dispatch_add_inplace(&self, enc: &mut wgpu::CommandEncoder, a: &wgpu::Buffer, b: &wgpu::Buffer, n: usize) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&ScalarParams { n: n as u32 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.add_inplace, &[a, b, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.add_inplace);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(((n + 255) / 256) as u32, 1, 1);
    }

    fn dispatch_argmax(&self, enc: &mut wgpu::CommandEncoder, n: usize) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&ScalarParams { n: n as u32 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.argmax, &[&self.scratch.logits, &self.scratch.token_id, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.argmax);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    // --- Encode helpers (add dispatch to existing pass, no pass creation) ---

    fn encode_matvec<'a>(&self, pass: &mut wgpu::ComputePass<'a>, w: &GpuWeight, input: &wgpu::Buffer, output: &wgpu::Buffer) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&MatvecParams { rows: w.rows as u32, cols: w.cols as u32 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let pipeline = match w.format {
            crate::wgpu_model::WeightFormat::Q4K => &self.pipelines.matvec_q4k,
            crate::wgpu_model::WeightFormat::F16Packed => &self.pipelines.matvec,
        };
        let bg = self.make_bg(pipeline, &[&w.buffer, input, output, &params]);
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(&bg), &[]);
        let rows = w.rows as u32;
        pass.dispatch_workgroups(rows.min(65535), (rows + 65534) / 65535, 1);
    }

    fn encode_matvec_maybe_bias<'a>(&self, pass: &mut wgpu::ComputePass<'a>, w: &GpuWeight, bias: Option<&GpuBias>, input: &wgpu::Buffer, output: &wgpu::Buffer) {
        // Note: for Q4K with bias, we can't do inline bias in the combined pass.
        // The bias add needs a separate pass (done in dispatch_matvec_maybe_bias).
        // For combined passes, we only batch projections without bias or with f16.
        match (bias, w.format) {
            (None, _) => self.encode_matvec(pass, w, input, output),
            (Some(b), crate::wgpu_model::WeightFormat::F16Packed) => {
                let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None, contents: bytemuck::bytes_of(&MatvecParams { rows: w.rows as u32, cols: w.cols as u32 }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = self.make_bg(&self.pipelines.matvec_bias, &[&w.buffer, input, output, &params, &b.buffer]);
                pass.set_pipeline(&self.pipelines.matvec_bias);
                pass.set_bind_group(0, Some(&bg), &[]);
                let rows = w.rows as u32;
                pass.dispatch_workgroups(rows.min(65535), (rows + 65534) / 65535, 1);
            }
            (Some(_), crate::wgpu_model::WeightFormat::Q4K) => {
                // Can't do inline bias with Q4K in a combined pass.
                // Just do the matvec; bias will be handled separately.
                self.encode_matvec(pass, w, input, output);
            }
        }
    }

    fn encode_rope<'a>(&self, pass: &mut wgpu::ComputePass<'a>, x: &wgpu::Buffer, n_heads: usize, head_dim: usize, pos: usize) {
        let half = head_dim / 2;
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&RopeParams { n_heads: n_heads as u32, head_dim: head_dim as u32, position: pos as u32, half_dim: half as u32 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.rope, &[x, &self.scratch.cos, &self.scratch.sin, &params]);
        pass.set_pipeline(&self.pipelines.rope);
        pass.set_bind_group(0, Some(&bg), &[]);
        let n = (n_heads * half) as u32;
        pass.dispatch_workgroups((n + 63) / 64, 1, 1);
    }

    // --- Batch dispatch helpers ---

    fn dispatch_batch_rmsnorm(&self, enc: &mut wgpu::CommandEncoder, input: &wgpu::Buffer, weight: &wgpu::Buffer, output: &wgpu::Buffer, n: usize, nt: usize) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&BatchRmsNormParams { n: n as u32, eps: self.model.config.rms_norm_eps, n_tokens: nt as u32, _pad: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.rmsnorm_batch, &[input, weight, output, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.rmsnorm_batch);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups(nt as u32, 1, 1);
    }

    fn dispatch_batch_matmul(&self, enc: &mut wgpu::CommandEncoder, w: &GpuWeight, input: &wgpu::Buffer, output: &wgpu::Buffer, nt: usize) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&BatchParams { rows: w.rows as u32, cols: w.cols as u32, n_tokens: nt as u32, _pad: 0 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.matmul, &[&w.buffer, input, output, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.matmul);
        pass.set_bind_group(0, Some(&bg), &[]);
        let rows = w.rows as u32;
        pass.dispatch_workgroups(rows.min(65535), (rows + 65534) / 65535, nt as u32);
    }

    fn dispatch_batch_matmul_maybe_bias(&self, enc: &mut wgpu::CommandEncoder, w: &GpuWeight, bias: Option<&GpuBias>, input: &wgpu::Buffer, output: &wgpu::Buffer, nt: usize) {
        match bias {
            None => self.dispatch_batch_matmul(enc, w, input, output, nt),
            Some(b) => {
                let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None, contents: bytemuck::bytes_of(&BatchParams { rows: w.rows as u32, cols: w.cols as u32, n_tokens: nt as u32, _pad: 0 }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = self.make_bg(&self.pipelines.matmul_bias, &[&w.buffer, input, output, &params, &b.buffer]);
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipelines.matmul_bias);
                pass.set_bind_group(0, Some(&bg), &[]);
                let rows = w.rows as u32;
                pass.dispatch_workgroups(rows.min(65535), (rows + 65534) / 65535, nt as u32);
            }
        }
    }

    fn dispatch_batch_rope(&self, enc: &mut wgpu::CommandEncoder, x: &wgpu::Buffer, n_heads: usize, head_dim: usize, start_pos: usize, nt: usize) {
        let half = head_dim / 2;
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&BatchRopeParams {
                n_heads: n_heads as u32, head_dim: head_dim as u32, start_pos: start_pos as u32,
                half_dim: half as u32, n_tokens: nt as u32, _p1: 0, _p2: 0, _p3: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.rope_batch, &[x, &self.scratch.cos, &self.scratch.sin, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.rope_batch);
        pass.set_bind_group(0, Some(&bg), &[]);
        let total = (nt * n_heads * half) as u32;
        pass.dispatch_workgroups((total + 63) / 64, 1, 1);
    }

    fn dispatch_batch_kv_write(&self, enc: &mut wgpu::CommandEncoder, layer: usize, kv_dim: usize, start_pos: usize, nt: usize) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&BatchKvWriteParams {
                kv_dim: kv_dim as u32, start_pos: start_pos as u32, n_tokens: nt as u32, _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.kv_write_batch, &[&self.scratch.batch_k, &self.scratch.batch_v, &self.scratch.k_caches[layer], &self.scratch.v_caches[layer], &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.kv_write_batch);
        pass.set_bind_group(0, Some(&bg), &[]);
        let total = (nt * kv_dim) as u32;
        pass.dispatch_workgroups((total + 127) / 128, 1, 1);
    }

    fn dispatch_batch_attn_score(&self, enc: &mut wgpu::CommandEncoder, layer: usize, cfg: &crate::wgpu_model::ModelConfig, start_pos: usize, nt: usize) {
        let max_total_seq = start_pos + nt;
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&BatchAttnScoreParams {
                n_heads: cfg.n_heads as u32, n_kv_heads: cfg.n_kv_heads as u32,
                head_dim: cfg.head_dim as u32, start_pos: start_pos as u32,
                max_seq: self.max_seq as u32, heads_per_kv: (cfg.n_heads / cfg.n_kv_heads) as u32,
                kv_dim: (cfg.n_kv_heads * cfg.head_dim) as u32,
                scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                n_tokens: nt as u32, _p1: 0, _p2: 0, _p3: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.attn_score_batch, &[&self.scratch.batch_q, &self.scratch.k_caches[layer], &self.scratch.batch_attn_scores, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.attn_score_batch);
        pass.set_bind_group(0, Some(&bg), &[]);
        let total = (nt * cfg.n_heads * max_total_seq) as u32;
        pass.dispatch_workgroups((total + 255) / 256, 1, 1);
    }

    fn dispatch_batch_softmax(&self, enc: &mut wgpu::CommandEncoder, cfg: &crate::wgpu_model::ModelConfig, start_pos: usize, nt: usize) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&BatchSoftmaxParams {
                n_heads: cfg.n_heads as u32, max_seq: self.max_seq as u32,
                start_pos: start_pos as u32, n_tokens: nt as u32,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.softmax_batch, &[&self.scratch.batch_attn_scores, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.softmax_batch);
        pass.set_bind_group(0, Some(&bg), &[]);
        pass.dispatch_workgroups((nt * cfg.n_heads) as u32, 1, 1);
    }

    fn dispatch_batch_attn_value(&self, enc: &mut wgpu::CommandEncoder, layer: usize, cfg: &crate::wgpu_model::ModelConfig, start_pos: usize, nt: usize) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&BatchAttnValueParams {
                n_heads: cfg.n_heads as u32, n_kv_heads: cfg.n_kv_heads as u32,
                head_dim: cfg.head_dim as u32, start_pos: start_pos as u32,
                max_seq: self.max_seq as u32, heads_per_kv: (cfg.n_heads / cfg.n_kv_heads) as u32,
                kv_dim: (cfg.n_kv_heads * cfg.head_dim) as u32, n_tokens: nt as u32,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.attn_value_batch, &[&self.scratch.batch_attn_scores, &self.scratch.v_caches[layer], &self.scratch.batch_attn_out, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.attn_value_batch);
        pass.set_bind_group(0, Some(&bg), &[]);
        let total = (nt * cfg.n_heads * cfg.head_dim) as u32;
        pass.dispatch_workgroups((total + 255) / 256, 1, 1);
    }

    fn dispatch_batch_silu_mul(&self, enc: &mut wgpu::CommandEncoder, n: usize, nt: usize) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&BatchScalarParams { n: n as u32, n_tokens: nt as u32 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.silu_mul_batch, &[&self.scratch.batch_ffn_gate, &self.scratch.batch_ffn_up, &self.scratch.batch_ffn_mid, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.silu_mul_batch);
        pass.set_bind_group(0, Some(&bg), &[]);
        let total = (n * nt) as u32;
        pass.dispatch_workgroups((total + 255) / 256, 1, 1);
    }

    fn dispatch_batch_add(&self, enc: &mut wgpu::CommandEncoder, a: &wgpu::Buffer, b: &wgpu::Buffer, n: usize, nt: usize) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, contents: bytemuck::bytes_of(&BatchScalarParams { n: n as u32, n_tokens: nt as u32 }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.make_bg(&self.pipelines.add_inplace_batch, &[a, b, &params]);
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines.add_inplace_batch);
        pass.set_bind_group(0, Some(&bg), &[]);
        let total = (n * nt) as u32;
        pass.dispatch_workgroups((total + 255) / 256, 1, 1);
    }

    /// Create a bind group for a pipeline from a list of buffers.
    fn make_bg(&self, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let layout = pipeline.get_bind_group_layout(0);
        let entries: Vec<wgpu::BindGroupEntry> = buffers.iter().enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.as_entire_binding(),
            })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &layout, entries: &entries,
        })
    }
}

// --- Initialization helpers ---

fn precompute_rope(head_dim: usize, max_seq: usize, freq_base: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0.0f32; max_seq * half];
    let mut sin = vec![0.0f32; max_seq * half];
    for pos in 0..max_seq {
        for i in 0..half {
            let freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * freq;
            cos[pos * half + i] = angle.cos();
            sin[pos * half + i] = angle.sin();
        }
    }
    (cos, sin)
}

fn create_scratch(device: &wgpu::Device, cfg: &crate::wgpu_model::ModelConfig, max_seq: usize, kv_dim: usize) -> Scratch {
    let rw = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
    let buf = |size: usize| -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None, size: (size * 4).max(4) as u64, usage: rw, mapped_at_creation: false,
        })
    };

    let half_dim = cfg.head_dim / 2;
    let max_batch = 64; // Max tokens per prefill batch (balances parallelism vs GPU load)
    Scratch {
        hidden: buf(cfg.d_model),
        normed: buf(cfg.d_model),
        q: buf(cfg.d_model),
        k: buf(kv_dim),
        v: buf(kv_dim),
        attn_out: buf(cfg.d_model),
        temp: buf(cfg.d_model),
        ffn_gate: buf(cfg.ffn_intermediate),
        ffn_up: buf(cfg.ffn_intermediate),
        ffn_mid: buf(cfg.ffn_intermediate),
        attn_scores: buf(cfg.n_heads * max_seq),
        logits: buf(cfg.vocab_size),
        token_id: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("token_id"), size: 4, usage: rw, mapped_at_creation: false,
        }),
        staging: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"), size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        k_caches: (0..cfg.n_layers).map(|_| buf(max_seq * kv_dim)).collect(),
        v_caches: (0..cfg.n_layers).map(|_| buf(max_seq * kv_dim)).collect(),
        cos: buf(max_seq * half_dim),
        sin: buf(max_seq * half_dim),
        batch_hidden: buf(max_batch * cfg.d_model),
        batch_normed: buf(max_batch * cfg.d_model),
        batch_q: buf(max_batch * cfg.d_model),
        batch_k: buf(max_batch * cfg.n_kv_heads * cfg.head_dim),
        batch_v: buf(max_batch * cfg.n_kv_heads * cfg.head_dim),
        batch_attn_out: buf(max_batch * cfg.d_model),
        batch_temp: buf(max_batch * cfg.d_model),
        batch_ffn_gate: buf(max_batch * cfg.ffn_intermediate),
        batch_ffn_up: buf(max_batch * cfg.ffn_intermediate),
        batch_ffn_mid: buf(max_batch * cfg.ffn_intermediate),
        batch_attn_scores: buf(max_batch * cfg.n_heads * max_seq),
        max_batch,
    }
}

fn create_all_pipelines(device: &wgpu::Device) -> Pipelines {
    let make = |src, label| {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label), source: wgpu::ShaderSource::Wgsl(src),
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label), layout: None, module: &module,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        })
    };

    Pipelines {
        matvec: make(include_str!("shaders/matvec.wgsl").into(), "matvec"),
        matvec_bias: make(include_str!("shaders/matvec_bias.wgsl").into(), "matvec_bias"),
        matvec_q4k: make(include_str!("shaders/matvec_q4k.wgsl").into(), "matvec_q4k"),
        rmsnorm: make(include_str!("shaders/rmsnorm.wgsl").into(), "rmsnorm"),
        rope: make(include_str!("shaders/rope.wgsl").into(), "rope"),
        silu_mul: make(include_str!("shaders/silu_mul.wgsl").into(), "silu_mul"),
        add_inplace: make(include_str!("shaders/add_inplace.wgsl").into(), "add_inplace"),
        kv_write: make(include_str!("shaders/kv_write.wgsl").into(), "kv_write"),
        attn_score: make(include_str!("shaders/attn_score.wgsl").into(), "attn_score"),
        softmax: make(include_str!("shaders/softmax.wgsl").into(), "softmax"),
        attn_value: make(include_str!("shaders/attn_value.wgsl").into(), "attn_value"),
        argmax: make(include_str!("shaders/argmax.wgsl").into(), "argmax"),
        matmul: make(include_str!("shaders/matmul.wgsl").into(), "matmul"),
        matmul_bias: make(include_str!("shaders/matmul_bias.wgsl").into(), "matmul_bias"),
        rmsnorm_batch: make(include_str!("shaders/rmsnorm_batch.wgsl").into(), "rmsnorm_batch"),
        rope_batch: make(include_str!("shaders/rope_batch.wgsl").into(), "rope_batch"),
        silu_mul_batch: make(include_str!("shaders/silu_mul_batch.wgsl").into(), "silu_mul_batch"),
        add_inplace_batch: make(include_str!("shaders/add_inplace_batch.wgsl").into(), "add_inplace_batch"),
        kv_write_batch: make(include_str!("shaders/kv_write_batch.wgsl").into(), "kv_write_batch"),
        attn_score_batch: make(include_str!("shaders/attn_score_batch.wgsl").into(), "attn_score_batch"),
        softmax_batch: make(include_str!("shaders/softmax_batch.wgsl").into(), "softmax_batch"),
        attn_value_batch: make(include_str!("shaders/attn_value_batch.wgsl").into(), "attn_value_batch"),
    }
}
