// Exact pixel coordinates for fractals defined by the digits of a point's coordinates
// (Sierpinski triangle, Sierpinski carpet).
//
// The CPU turns the view centre into integers: the point's two lattice coordinates
// scaled by base^levels, where `levels` is chosen so one unit is a fraction of a
// pixel (see fractals/lattice.rs). A pixel adds its own small integer offset. From
// there on everything is integer arithmetic, which is exact at any zoom and which no
// fast-math optimisation can touch.
//
// Uniform slots this module reads (see common.wgsl), filled by `lattice::apply_view`:
//   extra.x, extra.y              first coordinate of the view centre (low, high word)
//   extra.z, color_scheme         second coordinate (low, high word)
//   ref_escape_iter               number of digit levels in the scaled coordinates
//   c_real, c_imag                first row of the pixel -> lattice-unit matrix
//   center_lo_x, center_lo_y      second row

/// Signed 64-bit integer (two's complement) as two words
struct I64 {
    lo: u32,
    hi: u32,
}

fn i64_add(a: I64, b: I64) -> I64 {
    let lo = a.lo + b.lo;
    return I64(lo, a.hi + b.hi + u32(lo < a.lo));
}

fn i64_from_i32(v: i32) -> I64 {
    return I64(bitcast<u32>(v), bitcast<u32>(v >> 31u));
}

fn i64_is_negative(a: I64) -> bool {
    return (a.hi >> 31u) == 1u;
}

/// Bit `k` (0..63) of a
fn i64_bit(a: I64, k: u32) -> u32 {
    if k >= 32u { return (a.hi >> (k - 32u)) & 1u; }
    return (a.lo >> k) & 1u;
}

fn i64_clear_bit(a: I64, k: u32) -> I64 {
    if k >= 32u { return I64(a.lo, a.hi & ~(1u << (k - 32u))); }
    return I64(a.lo & ~(1u << k), a.hi);
}

/// Whether 0 <= a < 2^k, for a >= 0 and k in 1..63
fn i64_below_pow2(a: I64, k: u32) -> bool {
    if k >= 32u { return (a.hi >> (k - 32u)) == 0u; }
    return a.hi == 0u && (a.lo >> k) == 0u;
}

/// Quotient and remainder of a division by 3
struct Div3 {
    quotient: I64,
    remainder: u32,
}

/// Divide a (>= 0) by 3
fn i64_div3(a: I64) -> Div3 {
    // The high word divides directly; its remainder is carried into the low word,
    // which is then divided 16 bits at a time so nothing exceeds 32 bits.
    // (Dividing the high word 16 bits at a time as well gave wrong quotients on
    // Metal; `digit_fractals_match_f64_reference` in export.rs guards this.)
    let d1 = ((a.hi % 3u) << 16u) | (a.lo >> 16u);
    let d0 = ((d1 % 3u) << 16u) | (a.lo & 0xFFFFu);
    return Div3(I64(((d1 / 3u) << 16u) | (d0 / 3u), a.hi / 3u), d0 % 3u);
}

/// A pixel's two scaled lattice coordinates
struct LatticePoint {
    x: I64,
    y: I64,
}

fn lattice_point(id: vec2<u32>, dims: vec2<u32>) -> LatticePoint {
    let p = rotated_pixel(id, dims);
    let offset = vec2<i32>(round(vec2<f32>(
        uniforms.c_real * p.x + uniforms.c_imag * p.y,
        uniforms.center_lo_x * p.x + uniforms.center_lo_y * p.y,
    )));
    return LatticePoint(
        i64_add(I64(uniforms.extra.x, uniforms.extra.y), i64_from_i32(offset.x)),
        i64_add(I64(uniforms.extra.z, uniforms.color_scheme), i64_from_i32(offset.y)),
    );
}

/// Palette value for a hole of the given depth (0 = the biggest hole)
fn hole_value(depth: u32) -> f32 {
    return 1.0 + 2.9 * f32(depth);
}
