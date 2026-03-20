// Batch attention value: output[tok, head*head_dim + d] = sum_t(scores[tok, head, t] * V[t, kv_h, d])
// One thread per (tok, head, d) triple.

struct Params {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    start_pos: u32,
    max_seq: u32,
    heads_per_kv: u32,
    kv_dim: u32,
    n_tokens: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> v_cache: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let out_dim = params.n_heads * params.head_dim;
    let tok = idx / out_dim;
    let rem = idx % out_dim;
    let head = rem / params.head_dim;
    let d = rem % params.head_dim;

    if (tok >= params.n_tokens || head >= params.n_heads) { return; }

    let kv_h = head / params.heads_per_kv;
    let seq_len = params.start_pos + tok + 1u;
    let score_base = tok * params.n_heads * params.max_seq + head * params.max_seq;

    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < seq_len; t++) {
        let w = scores[score_base + t];
        let v_idx = t * params.kv_dim + kv_h * params.head_dim + d;
        acc += w * v_cache[v_idx];
    }

    output[tok * out_dim + head * params.head_dim + d] = acc;
}
