// Batch in-place add: a[i] += b[i] for n * n_tokens elements.

struct Params { n: u32, n_tokens: u32 }

@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.n * params.n_tokens;
    if (i >= total) { return; }
    a[i] += b[i];
}
