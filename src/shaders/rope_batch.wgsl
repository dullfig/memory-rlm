// Batch RoPE: apply rotary embeddings to x[tok, head, dim] for multiple tokens.
// Each token gets a different position (start_pos + tok_idx).

struct Params { n_heads: u32, head_dim: u32, start_pos: u32, half_dim: u32, n_tokens: u32, _p1: u32, _p2: u32, _p3: u32 }

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat_idx = gid.x;
    let total_pairs = params.n_heads * params.half_dim;
    let tok = flat_idx / total_pairs;
    let pair_in_tok = flat_idx % total_pairs;
    let head = pair_in_tok / params.half_dim;
    let pair = pair_in_tok % params.half_dim;

    if (tok >= params.n_tokens || head >= params.n_heads) { return; }

    let pos = params.start_pos + tok;
    let cos_idx = pos * params.half_dim + pair;
    let c = cos_table[cos_idx];
    let s = sin_table[cos_idx];

    let tok_base = tok * params.n_heads * params.head_dim;
    let base = tok_base + head * params.head_dim;
    let x0 = x[base + pair];
    let x1 = x[base + pair + params.half_dim];

    x[base + pair] = x0 * c - x1 * s;
    x[base + pair + params.half_dim] = x0 * s + x1 * c;
}
