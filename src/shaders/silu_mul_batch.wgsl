// Batch SiLU(gate) * up for multiple tokens.

struct Params { n: u32, n_tokens: u32 }

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.n * params.n_tokens;
    if (i >= total) { return; }
    let g = gate[i];
    output[i] = (g / (1.0 + exp(-g))) * up[i];
}
