// Q4_K matrix-vector multiply: dequantize on-the-fly during dot product.
// Weights stored as raw Q4_K blocks: 36 u32s (144 bytes) per block of 256 values.
//
// Block layout (in u32 terms):
//   [0]    : d (f16 low) | dmin (f16 high)
//   [1..3] : 12 bytes of packed 6-bit scales/mins
//   [4..35]: 128 bytes of 4-bit quantized values (2 nibbles per byte)
//
// Each block has 4 groups of 64 values. Each group uses 32 bytes of qs:
//   - First 32 values: low nibbles (& 0xF) with scale[2*group]
//   - Next 32 values: high nibbles (>> 4) with scale[2*group+1]

struct Params {
    rows: u32,
    cols: u32,  // must be multiple of 256
}

@group(0) @binding(0) var<storage, read> weights: array<u32>;
@group(0) @binding(1) var<storage, read> input_vec: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_vec: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

const WG: u32 = 256u;
const BLOCK_QS: u32 = 256u;
const BLOCK_U32S: u32 = 36u;

var<workgroup> wg_sum: array<f32, WG>;

fn scales_byte(s0: u32, s1: u32, s2: u32, idx: u32) -> u32 {
    var word: u32;
    var shift: u32;
    if (idx < 4u) {
        word = s0; shift = idx * 8u;
    } else if (idx < 8u) {
        word = s1; shift = (idx - 4u) * 8u;
    } else {
        word = s2; shift = (idx - 8u) * 8u;
    }
    return (word >> shift) & 0xFFu;
}

fn get_scale_min(s0: u32, s1: u32, s2: u32, j: u32) -> vec2<f32> {
    var sc: u32;
    var mn: u32;
    if (j < 4u) {
        sc = scales_byte(s0, s1, s2, j) & 63u;
        mn = scales_byte(s0, s1, s2, j + 4u) & 63u;
    } else {
        sc = (scales_byte(s0, s1, s2, j + 4u) & 0xFu) | ((scales_byte(s0, s1, s2, j - 4u) >> 6u) << 4u);
        mn = (scales_byte(s0, s1, s2, j + 4u) >> 4u) | ((scales_byte(s0, s1, s2, j) >> 6u) << 4u);
    }
    return vec2<f32>(f32(sc), f32(mn));
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x + wid.y * 65535u;
    if (row >= params.rows) { return; }
    let tid = lid.x;

    let blocks_per_row = params.cols / BLOCK_QS;
    var sum: f32 = 0.0;

    var block_idx = tid;
    while (block_idx < blocks_per_row) {
        let w_off = (row * blocks_per_row + block_idx) * BLOCK_U32S;
        let col_base = block_idx * BLOCK_QS;

        let hdr = weights[w_off];
        let d = unpack2x16float(hdr).x;
        let dmin = unpack2x16float(hdr).y;

        let s0 = weights[w_off + 1u];
        let s1 = weights[w_off + 2u];
        let s2 = weights[w_off + 3u];

        var is: u32 = 0u;
        for (var g: u32 = 0u; g < 4u; g++) {
            let qs_base = w_off + 4u + g * 8u;
            let sm1 = get_scale_min(s0, s1, s2, is);
            let d1 = d * sm1.x;
            let m1 = dmin * sm1.y;
            let sm2 = get_scale_min(s0, s1, s2, is + 1u);
            let d2 = d * sm2.x;
            let m2 = dmin * sm2.y;

            for (var l: u32 = 0u; l < 8u; l++) {
                let qw = weights[qs_base + l];
                let c_lo = col_base + g * 64u + l * 4u;
                let c_hi = c_lo + 32u;

                sum += (d1 * f32((qw >>  0u) & 0xFu) - m1) * input_vec[c_lo + 0u];
                sum += (d1 * f32((qw >>  8u) & 0xFu) - m1) * input_vec[c_lo + 1u];
                sum += (d1 * f32((qw >> 16u) & 0xFu) - m1) * input_vec[c_lo + 2u];
                sum += (d1 * f32((qw >> 24u) & 0xFu) - m1) * input_vec[c_lo + 3u];

                sum += (d2 * f32((qw >>  4u) & 0xFu) - m2) * input_vec[c_hi + 0u];
                sum += (d2 * f32((qw >> 12u) & 0xFu) - m2) * input_vec[c_hi + 1u];
                sum += (d2 * f32((qw >> 20u) & 0xFu) - m2) * input_vec[c_hi + 2u];
                sum += (d2 * f32((qw >> 28u) & 0xFu) - m2) * input_vec[c_hi + 3u];
            }
            is += 2u;
        }
        block_idx += WG;
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
