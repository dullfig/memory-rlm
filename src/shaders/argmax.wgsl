// Argmax: find index of maximum value. Single workgroup, 256 threads.

struct Params { n: u32 }

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> wg_val: array<f32, WG>;
var<workgroup> wg_idx: array<u32, WG>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;

    var max_val: f32 = -1e30;
    var max_idx: u32 = 0u;
    var i = tid;
    while (i < params.n) {
        if (input[i] > max_val) {
            max_val = input[i];
            max_idx = i;
        }
        i += WG;
    }

    wg_val[tid] = max_val;
    wg_idx[tid] = max_idx;
    workgroupBarrier();

    var s = WG / 2u;
    while (s > 0u) {
        if (tid < s && wg_val[tid + s] > wg_val[tid]) {
            wg_val[tid] = wg_val[tid + s];
            wg_idx[tid] = wg_idx[tid + s];
        }
        workgroupBarrier();
        s >>= 1u;
    }

    if (tid == 0u) {
        result[0u] = wg_idx[0u];
    }
}
