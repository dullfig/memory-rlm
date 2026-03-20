// Attention value: output[h*head_dim + d] = sum_t(scores[h, t] * V_cache[t, kv_h, d])
// One thread per (head, dim) pair.

struct Params {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq: u32,
    heads_per_kv: u32,
    kv_dim: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> v_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let head = idx / params.head_dim;
    let d = idx % params.head_dim;
    if (head >= params.n_heads) { return; }

    let kv_h = head / params.heads_per_kv;

    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < params.seq_len; t++) {
        let w = scores[head * params.max_seq + t];
        let v_idx = t * params.kv_dim + kv_h * params.head_dim + d;
        acc += w * v_cache[v_idx];
    }

    output[head * params.head_dim + d] = acc;
}
