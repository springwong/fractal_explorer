// Emulated double precision from two f32 (Dekker/Knuth algorithms, no fma needed).
// A value is carried as (hi, lo) with value = hi + lo.
// ── Portable double-single arithmetic ──
const SPLIT: f32 = 4097.0;

fn veltkamp_split(a: f32) -> vec2<f32> {
    let c = SPLIT * a;
    let a_hi = c - (c - a);
    let a_lo = a - a_hi;
    return vec2<f32>(a_hi, a_lo);
}

fn two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    let err = (a - (s - v)) + (b - v);
    return vec2<f32>(s, err);
}

fn two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let a_s = veltkamp_split(a);
    let b_s = veltkamp_split(b);
    let err = ((a_s.x * b_s.x - p) + a_s.x * b_s.y + a_s.y * b_s.x) + a_s.y * b_s.y;
    return vec2<f32>(p, err);
}

fn ds_add(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> vec2<f32> {
    let s = two_sum(a_hi, b_hi);
    let lo = s.y + a_lo + b_lo;
    return two_sum(s.x, lo);
}

fn ds_mul(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> vec2<f32> {
    let p = two_prod(a_hi, b_hi);
    let lo = a_hi * b_lo + a_lo * b_hi + p.y;
    return two_sum(p.x, lo);
}

fn ds_div(a_hi: f32, a_lo: f32, b_hi: f32, b_lo: f32) -> vec2<f32> {
    let q1 = a_hi / b_hi;
    let p = two_prod(q1, b_hi);
    let r = ds_add(a_hi, a_lo, -p.x, -(p.y + q1 * b_lo));
    let q2 = r.x / b_hi;
    return two_sum(q1, q2);
}

fn ds_abs(hi: f32, lo: f32) -> vec2<f32> {
    if hi < 0.0 || (hi == 0.0 && lo < 0.0) {
        return vec2<f32>(-hi, -lo);
    }
    return vec2<f32>(hi, lo);
}

/// Plane coordinate of a pixel in double-single: (x_hi, x_lo, y_hi, y_lo).
fn pixel_to_plane_ds(id: vec2<u32>, dims: vec2<u32>) -> vec4<f32> {
    let rotated = rotated_pixel(id, dims) / f32(dims.y);
    let offset_x = ds_div(rotated.x, 0.0, uniforms.zoom, uniforms.zoom_lo);
    let offset_y = ds_div(-rotated.y, 0.0, uniforms.zoom, uniforms.zoom_lo);
    let x = ds_add(uniforms.center.x, uniforms.center_lo_x, offset_x.x, offset_x.y);
    let y = ds_add(uniforms.center.y, uniforms.center_lo_y, offset_y.x, offset_y.y);
    return vec4<f32>(x.x, x.y, y.x, y.y);
}
