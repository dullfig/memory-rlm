// Batch RMSNorm: output[tok, i] = input[tok, i] * weight[i] / sqrt(mean(input[tok]²) + eps)
// One workgroup per token.

struct Params { n: u32, eps: f32, n_tokens: u32, _pad: u32 }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> wg_sum: array<f32, WG>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tok = wid.x;
    if (tok >= params.n_tokens) { return; }
    let tid = lid.x;
    let base = tok * params.n;

    var sq: f32 = 0.0;
    var i = tid;
    while (i < params.n) {
        let v = input[base + i];
        sq += v * v;
        i += WG;
    }
    wg_sum[tid] = sq;
    workgroupBarrier();
    var s = WG / 2u;
    while (s > 0u) {
        if (tid < s) { wg_sum[tid] += wg_sum[tid + s]; }
        workgroupBarrier();
        s >>= 1u;
    }
    let rms = 1.0 / sqrt(wg_sum[0u] / f32(params.n) + params.eps);

    i = tid;
    while (i < params.n) {
        output[base + i] = input[base + i] * rms * weight[i];
        i += WG;
    }
}
