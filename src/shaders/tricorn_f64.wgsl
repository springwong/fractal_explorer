// Tricorn (Mandelbar) fractal with emulated double precision (2x f32) for deep zoom
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

/// Tricorn iteration with full double-single precision
fn tricorn_ds(cx_hi: f32, cx_lo: f32, cy_hi: f32, cy_lo: f32) -> f32 {
    var zrh: f32 = 0.0; var zrl: f32 = 0.0;
    var zih: f32 = 0.0; var zil: f32 = 0.0;
    var iter: u32 = 0u;

    while (iter < uniforms.max_iter) {
        let mag2 = zrh * zrh + zih * zih;
        if mag2 > 4.0 { break; }

        let zr_sq = ds_mul(zrh, zrl, zrh, zrl);
        let zi_sq = ds_mul(zih, zil, zih, zil);
        // Conjugate: -2 * zr * zi
        let zr_zi = ds_mul(zrh, zrl, zih, zil);
        let neg_two_zr_zi_hi = -(zr_zi.x + zr_zi.x);
        let neg_two_zr_zi_lo = -(zr_zi.y + zr_zi.y);

        let diff = ds_add(zr_sq.x, zr_sq.y, -zi_sq.x, -zi_sq.y);
        let new_zr = ds_add(diff.x, diff.y, cx_hi, cx_lo);
        let new_zi = ds_add(neg_two_zr_zi_hi, neg_two_zr_zi_lo, cy_hi, cy_lo);

        zrh = new_zr.x; zrl = new_zr.y;
        zih = new_zi.x; zil = new_zi.y;
        iter += 1u;
    }

    if iter == uniforms.max_iter { return 0.0; }
    let log_zn = log2(zrh * zrh + zih * zih) / 2.0;
    let nu = log2(log_zn / log2(2.0));
    return f32(iter) + 1.0 - nu;
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

    let smooth_val = tricorn_ds(cx_ds.x, cx_ds.y, cy_ds.x, cy_ds.y);

    let color = sample_palette(smooth_val);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
