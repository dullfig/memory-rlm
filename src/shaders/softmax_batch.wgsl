// Batch softmax: one workgroup per (token, head) pair.
// scores layout: [n_tokens, n_heads, max_seq]
// Softmax over [0, start_pos + tok] for each (tok, head).

struct Params { n_heads: u32, max_seq: u32, start_pos: u32, n_tokens: u32 }

@group(0) @binding(0) var<storage, read_write> scores: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

const WG: u32 = 256u;
var<workgroup> wg_data: array<f32, WG>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    // wid.x encodes (tok * n_heads + head)
    let flat = wid.x;
    let tok = flat / params.n_heads;
    let head = flat % params.n_heads;
    if (tok >= params.n_tokens) { return; }
    let tid = lid.x;
    let seq_len = params.start_pos + tok + 1u;
    let base = tok * params.n_heads * params.max_seq + head * params.max_seq;

    // Max reduction
    var max_val: f32 = -1e30;
    var i = tid;
    while (i < seq_len) {
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

    // Exp + sum
    var exp_sum: f32 = 0.0;
    i = tid;
    while (i < seq_len) {
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

    // Normalize
    i = tid;
    while (i < seq_len) {
        scores[base + i] /= total;
        i += WG;
    }
}
