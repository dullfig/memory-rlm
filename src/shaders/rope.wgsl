// RoPE: Rotary position embeddings applied in-place.
// x has shape [n_heads, head_dim]. Rotates pairs (i, i+half) using cos/sin at position.

struct Params { n_heads: u32, head_dim: u32, position: u32, half_dim: u32 }

@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> cos_table: array<f32>;
@group(0) @binding(2) var<storage, read> sin_table: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let head = idx / params.half_dim;
    let pair = idx % params.half_dim;
    if (head >= params.n_heads) { return; }

    let cos_idx = params.position * params.half_dim + pair;
    let c = cos_table[cos_idx];
    let s = sin_table[cos_idx];

    let base = head * params.head_dim;
    let x0 = x[base + pair];
    let x1 = x[base + pair + params.half_dim];

    x[base + pair] = x0 * c - x1 * s;
    x[base + pair + params.half_dim] = x0 * s + x1 * c;
}
