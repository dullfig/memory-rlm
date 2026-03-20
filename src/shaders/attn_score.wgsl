// Attention scores: scores[h * max_seq + t] = dot(Q[h], K_cache[t, kv_h]) * scale
// One thread per (head, time_step) pair. Dot product computed serially (head_dim=64).

struct Params {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq: u32,
    heads_per_kv: u32,
    kv_dim: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> scores: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let head = idx / params.seq_len;
    let t = idx % params.seq_len;
    if (head >= params.n_heads) { return; }

    let kv_h = head / params.heads_per_kv;
    let q_base = head * params.head_dim;
    let k_base = t * params.kv_dim + kv_h * params.head_dim;

    var dot: f32 = 0.0;
    for (var d: u32 = 0u; d < params.head_dim; d++) {
        dot += q[q_base + d] * k_cache[k_base + d];
    }

    scores[head * params.max_seq + t] = dot * params.scale;
}
