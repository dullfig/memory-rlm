// Softmax over scores[head * max_seq .. head * max_seq + seq_len] for each head.
// One workgroup per head.

struct Params { n_heads: u32, seq_len: u32, max_seq: u32 }

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> wg_data: array<f32, WG>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let head = wid.x;
    if (head >= params.n_heads) { return; }
    let tid = lid.x;
    let base = head * params.max_seq;

    // Phase 1: find max
    var max_val: f32 = -1e30;
    var i = tid;
    while (i < params.seq_len) {
        max_val = max(max_val, scores[base + i]);
        i += WG;
    }
    wg_data[tid] = max_val;
    workgroupBarrier();
    var s = WG / 2u;
    while (s > 0u) {
        if (tid < s) { wg_data[tid] = max(wg_data[tid], wg_data[tid + s]); }
        workgroupBarrier();
        s >>= 1u;
    }
    let max_v = wg_data[0u];
    workgroupBarrier();

    // Phase 2: exp and sum
    var exp_sum: f32 = 0.0;
    i = tid;
    while (i < params.seq_len) {
        let e = exp(scores[base + i] - max_v);
        scores[base + i] = e;
        exp_sum += e;
        i += WG;
    }
    wg_data[tid] = exp_sum;
    workgroupBarrier();
    s = WG / 2u;
    while (s > 0u) {
        if (tid < s) { wg_data[tid] += wg_data[tid + s]; }
        workgroupBarrier();
        s >>= 1u;
    }
    let total = wg_data[0u];
    workgroupBarrier();

    // Phase 3: normalize
    i = tid;
    while (i < params.seq_len) {
        scores[base + i] /= total;
        i += WG;
    }
}
