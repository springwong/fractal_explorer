// Golden-Silver Gasket: the Sierpinski triangle rebuilt in the golden-ratio base.
//
// Write |x| and |y| greedily as sums of distinct powers of phi with no two
// consecutive powers (Zeckendorf-style). A point belongs to the gasket iff x and y
// never use the same power of phi. Level counts grow by the silver ratio 1+sqrt(2)
// while cells shrink by phi, so the dimension is ln(1+sqrt(2)) / ln(phi) ~ 1.8316.
//
// Rendering: a pixel falls into a hole the first time both expansions use the same
// power; it is coloured by that level. Pixels that survive down to pixel size are
// the gasket itself (black, like the inside of the other sets).
//
// Always runs in emulated double precision (2x f32): the pattern spans |x|, |y| < 199,
// and out there plain f32 runs out of bits at quite modest zoom.
// Uses Dekker/Knuth algorithms — no fma dependency.
struct Uniforms {
    center: vec2<f32>,
    zoom: f32,
    aspect_ratio: f32,
    max_iter: u32,
    fractal_type: u32,
    color_scheme: u32,
    c_real: f32,
    c_imag: f32,
    center_lo_x: f32,
    center_lo_y: f32,
    zoom_lo: f32,
    pixel_step_x: f32,
    pixel_step_y: f32,
    ref_escape_iter: u32,
    rotation: f32,
    _pad2: vec3<u32>,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> palette_lut: array<u32>;

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

const LN_PHI: f32 = 0.4812118251;
const TOP_POW: i32 = 10;
// 1/phi and phi^10 as double-single (hi, lo) pairs
const INV_PHI_HI: f32 = 0.6180340051651001;
const INV_PHI_LO: f32 = -1.641520519513051e-08;
const W_TOP_HI: f32 = 122.99186706542969;
const W_TOP_LO: f32 = 2.315814526809845e-06;
const EXTENT: f32 = 199.0;
// Double-single carries ~48 bits: good to roughly phi^-62.
const MAX_FRAC_LEVELS: i32 = 62;

fn ds_ge(a: vec2<f32>, b: vec2<f32>) -> bool {
    return a.x > b.x || (a.x == b.x && a.y >= b.y);
}

/// Returns the level at which the point is removed (>= 1), or 0 if it survives `levels` levels.
fn gasket_ds(px: vec2<f32>, py: vec2<f32>, levels: u32) -> f32 {
    var x = px;
    var y = py;
    if x.x < 0.0 { x = -x; }
    if y.x < 0.0 { y = -y; }
    if max(x.x, y.x) >= EXTENT { return 0.0; }

    var w = vec2<f32>(W_TOP_HI, W_TOP_LO);
    for (var l: u32 = 1u; l <= levels; l += 1u) {
        let dx = ds_ge(x, w);
        let dy = ds_ge(y, w);
        if dx && dy { return f32(l); }
        if dx { x = ds_add(x.x, x.y, -w.x, -w.y); }
        if dy { y = ds_add(y.x, y.y, -w.x, -w.y); }
        w = ds_mul(w.x, w.y, INV_PHI_HI, INV_PHI_LO);
    }
    return 0.0;
}

/// Number of levels to resolve: stop once cells are about one pixel wide.
fn level_count(dims_y: u32) -> u32 {
    let frac = i32(round(log(max(uniforms.zoom * f32(dims_y), 1e-6)) / LN_PHI));
    // Far from the origin the coordinates spend bits on their integer part
    let reach = max(max(abs(uniforms.center.x), abs(uniforms.center.y)), 1.0);
    let max_frac = MAX_FRAC_LEVELS - i32(ceil(log(reach) / LN_PHI));
    let total = clamp(TOP_POW + 1 + min(frac, max_frac), 1, 96);
    return min(u32(total), uniforms.max_iter);
}

fn sample_palette(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let index = u32(fract(t * 0.05) * 255.0);
    let packed = palette_lut[index];
    let r = f32((packed >> 0u) & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, 1.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let px = (f32(id.x) - f32(dims.x) / 2.0) / f32(dims.y);
    let py = (f32(id.y) - f32(dims.y) / 2.0) / f32(dims.y);
    let cos_r = cos(uniforms.rotation);
    let sin_r = sin(uniforms.rotation);
    let pixel_offset_x = px * cos_r - py * sin_r;
    let pixel_offset_y = px * sin_r + py * cos_r;

    let offset_x_ds = ds_div(pixel_offset_x, 0.0, uniforms.zoom, uniforms.zoom_lo);
    let offset_y_ds = ds_div(-pixel_offset_y, 0.0, uniforms.zoom, uniforms.zoom_lo);

    let cx_ds = ds_add(uniforms.center.x, uniforms.center_lo_x, offset_x_ds.x, offset_x_ds.y);
    let cy_ds = ds_add(uniforms.center.y, uniforms.center_lo_y, offset_y_ds.x, offset_y_ds.y);

    let smooth_val = gasket_ds(cx_ds, cy_ds, level_count(dims.y)) * 2.5;

    let color = sample_palette(smooth_val);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
