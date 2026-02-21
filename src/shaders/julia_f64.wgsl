// Julia set with emulated double precision (2x f32) for deep zoom
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
    _pad: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

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

/// Julia iteration with double-single z, f32 c parameter
fn julia_ds(zx_hi: f32, zx_lo: f32, zy_hi: f32, zy_lo: f32, cx: f32, cy: f32) -> f32 {
    var zrh: f32 = zx_hi; var zrl: f32 = zx_lo;
    var zih: f32 = zy_hi; var zil: f32 = zy_lo;
    var iter: u32 = 0u;

    while (iter < uniforms.max_iter) {
        let mag2 = zrh * zrh + zih * zih;
        if mag2 > 4.0 { break; }

        let zr_sq = ds_mul(zrh, zrl, zrh, zrl);
        let zi_sq = ds_mul(zih, zil, zih, zil);
        let zr_zi = ds_mul(zrh, zrl, zih, zil);
        let two_zr_zi_hi = zr_zi.x + zr_zi.x;
        let two_zr_zi_lo = zr_zi.y + zr_zi.y;

        let diff = ds_add(zr_sq.x, zr_sq.y, -zi_sq.x, -zi_sq.y);
        let new_zr = ds_add(diff.x, diff.y, cx, 0.0);
        let new_zi = ds_add(two_zr_zi_hi, two_zr_zi_lo, cy, 0.0);

        zrh = new_zr.x; zrl = new_zr.y;
        zih = new_zi.x; zil = new_zi.y;
        iter += 1u;
    }

    if iter == uniforms.max_iter { return 0.0; }
    let log_zn = log2(zrh * zrh + zih * zih) / 2.0;
    let nu = log2(log_zn / log2(2.0));
    return f32(iter) + 1.0 - nu;
}

fn colorize_smooth(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let hue = fract(t * 0.05); let sat = 0.8; let val = 0.9;
    let h = hue * 6.0; let i = floor(h); let f = h - i;
    let p = val * (1.0 - sat); let q = val * (1.0 - sat * f);
    let t_val = val * (1.0 - sat * (1.0 - f));
    var rgb: vec3<f32>;
    if i == 0.0 { rgb = vec3<f32>(val, t_val, p); }
    else if i == 1.0 { rgb = vec3<f32>(q, val, p); }
    else if i == 2.0 { rgb = vec3<f32>(p, val, t_val); }
    else if i == 3.0 { rgb = vec3<f32>(p, q, val); }
    else if i == 4.0 { rgb = vec3<f32>(t_val, p, val); }
    else { rgb = vec3<f32>(val, p, q); }
    return vec4<f32>(rgb, 1.0);
}

fn colorize_fire(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let n = fract(t * 0.03);
    return vec4<f32>(min(1.0, n * 2.0), max(0.0, min(1.0, (n - 0.3) * 2.5)), max(0.0, min(1.0, (n - 0.7) * 3.3)), 1.0);
}

fn colorize_ocean(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.1, 1.0); }
    let n = fract(t * 0.04);
    return vec4<f32>(max(0.0, min(1.0, (n - 0.6) * 2.5)), max(0.0, min(1.0, (n - 0.2) * 1.8)), min(1.0, 0.3 + n * 0.7), 1.0);
}

fn colorize_grayscale(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let i = fract(t * 0.05);
    return vec4<f32>(i, i, i, 1.0);
}

fn colorize(t: f32, scheme: u32) -> vec4<f32> {
    switch scheme {
        case 0u: { return colorize_smooth(t); }
        case 1u: { return colorize_fire(t); }
        case 2u: { return colorize_ocean(t); }
        case 3u: { return colorize_grayscale(t); }
        default: { return colorize_smooth(t); }
    }
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let pixel_offset_x = (f32(id.x) - f32(dims.x) / 2.0) / f32(dims.y);
    let pixel_offset_y = (f32(id.y) - f32(dims.y) / 2.0) / f32(dims.y);

    let offset_x_ds = ds_div(pixel_offset_x, 0.0, uniforms.zoom, uniforms.zoom_lo);
    let offset_y_ds = ds_div(-pixel_offset_y, 0.0, uniforms.zoom, uniforms.zoom_lo);

    let zx_ds = ds_add(uniforms.center.x, uniforms.center_lo_x, offset_x_ds.x, offset_x_ds.y);
    let zy_ds = ds_add(uniforms.center.y, uniforms.center_lo_y, offset_y_ds.x, offset_y_ds.y);

    let smooth_val = julia_ds(zx_ds.x, zx_ds.y, zy_ds.x, zy_ds.y, uniforms.c_real, uniforms.c_imag);

    let color = colorize(smooth_val, uniforms.color_scheme);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
