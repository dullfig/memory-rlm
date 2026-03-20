// Matrix-vector multiply with f16-packed weights.
// Weights stored as array<u32> where each u32 = two f16 values.
// output[row] = dot(dequant(weights[row]), input)

struct Params {
    rows: u32,
    cols: u32,  // logical columns (must be even)
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input_vec: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_vec: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> wg_sum: array<f32, WG>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x + wid.y * 65535u;
    if (row >= params.rows) { return; }
    let tid = lid.x;

    let half_cols = params.cols / 2u;
    let base = row * half_cols;

    var sum: f32 = 0.0;
    var i = tid;
    while (i < half_cols) {
        let w = unpack2x16float(weights[base + i]);
        let col = i * 2u;
        sum += w.x * input_vec[col] + w.y * input_vec[col + 1u];
        i += WG;
    }

    wg_sum[tid] = sum;
    workgroupBarrier();

    var s = WG / 2u;
    while (s > 0u) {
        if (tid < s) { wg_sum[tid] += wg_sum[tid + s]; }
        workgroupBarrier();
        s >>= 1u;
    }

    if (tid == 0u) {
        output_vec[row] = wg_sum[0u];
    }
}
