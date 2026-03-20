// Write K and V vectors into the KV cache at a given position.

struct Params { kv_dim: u32, position: u32 }

@group(0) @binding(0) var<storage, read> k_src: array<f32>;
@group(0) @binding(1) var<storage, read> v_src: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.kv_dim) { return; }
    let offset = params.position * params.kv_dim + i;
    k_cache[offset] = k_src[i];
    v_cache[offset] = v_src[i];
}
