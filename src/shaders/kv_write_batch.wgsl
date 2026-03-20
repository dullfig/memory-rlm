// Write K and V vectors for multiple tokens into KV cache.
// k_src/v_src: [n_tokens, kv_dim], written to positions [start_pos .. start_pos + n_tokens]

struct Params { kv_dim: u32, start_pos: u32, n_tokens: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> k_src: array<f32>;
@group(0) @binding(1) var<storage, read> v_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let total = params.kv_dim * params.n_tokens;
    if (flat >= total) { return; }

    let tok = flat / params.kv_dim;
    let dim = flat % params.kv_dim;
    let cache_pos = (params.start_pos + tok) * params.kv_dim + dim;

    k_cache[cache_pos] = k_src[flat];
    v_cache[cache_pos] = v_src[flat];
}
