// Julia set with perturbation theory for deep zoom
// Reference orbit computed in f64 on CPU, per-pixel deltas in f32 on GPU.
// Julia perturbation: w_{n+1} = 2 * Z*_n * w_n + w_n^2  (no +delta_c since c is same for all pixels)
// w_0 = pixel_offset (initial z differs per pixel)
struct Uniforms {
    center: vec2<f32>,       // offset 0, 8 bytes (hi part)
    zoom: f32,               // offset 8, 4 bytes (hi part)
    aspect_ratio: f32,       // offset 12, 4 bytes
    max_iter: u32,           // offset 16, 4 bytes
    fractal_type: u32,       // offset 20, 4 bytes
    color_scheme: u32,       // offset 24, 4 bytes
    c_real: f32,             // offset 28, 4 bytes (Julia c parameter)
    c_imag: f32,             // offset 32, 4 bytes (Julia c parameter)
    center_lo_x: f32,        // offset 36, 4 bytes
    center_lo_y: f32,        // offset 40, 4 bytes
    zoom_lo: f32,            // offset 44, 4 bytes
    pixel_step_x: f32,       // offset 48, 4 bytes
    pixel_step_y: f32,       // offset 52, 4 bytes
    ref_escape_iter: u32,    // offset 56, 4 bytes
    rotation: f32,           // offset 60, 4 bytes
    _pad2: vec3<u32>,        // offset 64, 12 bytes
    _pad3: u32,              // offset 76, 4 bytes
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
// Reference orbit: flat array of f32 pairs [zx, zy] for iterations 0..max_iter
@group(0) @binding(2) var<storage, read> ref_orbit: array<f32>;

const BAILOUT2: f32 = 256.0;
const GLITCH_TOLERANCE: f32 = 1e-3;

/// Perturbation-based Julia iteration with glitch detection.
///
/// Julia: z_{n+1} = z_n^2 + c  (c is fixed for all pixels)
/// Reference: Z*_0 = center, Z*_{n+1} = (Z*_n)^2 + c
/// Perturbation: w_n = z_n - Z*_n
///   w_0 = z_0 - Z*_0 = pixel_coord - center = delta
///   w_{n+1} = 2 * Z*_n * w_n + w_n^2  (no delta_c term!)
fn julia_perturbation(delta_x: f32, delta_y: f32) -> f32 {
    var wx: f32 = delta_x;
    var wy: f32 = delta_y;
    var iter: u32 = 0u;
    let ref_max = uniforms.ref_escape_iter;

    while (iter < uniforms.max_iter) {
        // Reference orbit at iteration n
        let idx = iter * 2u;
        let zx_ref = ref_orbit[idx];
        let zy_ref = ref_orbit[idx + 1u];

        // Full Z = Z* + w
        let full_zx = zx_ref + wx;
        let full_zy = zy_ref + wy;
        let full_mag2 = full_zx * full_zx + full_zy * full_zy;

        // Escape check
        if full_mag2 > BAILOUT2 {
            let log_zn = log2(full_mag2) / 2.0;
            let nu = log2(log_zn / log2(2.0));
            return f32(iter) + 1.0 - nu;
        }

        // If reference orbit has escaped, switch to direct iteration
        if iter >= ref_max {
            return direct_iterate(full_zx, full_zy, iter);
        }

        // Glitch detection
        let w_mag2 = wx * wx + wy * wy;
        if w_mag2 > GLITCH_TOLERANCE * full_mag2 {
            return direct_iterate(full_zx, full_zy, iter);
        }

        // Perturbation step: w_{n+1} = 2 * Z*_n * w_n + w_n^2
        // (no +delta_c because c is the same for reference and pixel)
        let new_wx = 2.0 * (zx_ref * wx - zy_ref * wy) + (wx * wx - wy * wy);
        let new_wy = 2.0 * (zx_ref * wy + zy_ref * wx) + 2.0 * wx * wy;

        wx = new_wx;
        wy = new_wy;
        iter += 1u;
    }

    return 0.0;
}

/// Direct f32 iteration fallback for Julia set.
fn direct_iterate(zx_start: f32, zy_start: f32, start_iter: u32) -> f32 {
    let cx = uniforms.c_real;
    let cy = uniforms.c_imag;
    var zx = zx_start;
    var zy = zy_start;
    var j = start_iter;

    while (j < uniforms.max_iter) {
        let mag2 = zx * zx + zy * zy;
        if mag2 > BAILOUT2 {
            let log_zn = log2(mag2) / 2.0;
            let nu = log2(log_zn / log2(2.0));
            return f32(j) + 1.0 - nu;
        }
        let temp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = temp;
        j += 1u;
    }

    return 0.0;
}

fn colorize_smooth(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let hue = fract(t * 0.05); let sat = 0.8; let val = 0.9;
    let h = hue * 6.0; let i = floor(h); let f = h - i;
    let p = val * (1.0 - sat);
    let q = val * (1.0 - sat * f);
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
    let intensity = fract(t * 0.05);
    return vec4<f32>(intensity, intensity, intensity, 1.0);
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

    // Per-pixel delta from center (pixel_step computed on CPU in f64)
    let px = f32(id.x) - f32(dims.x) / 2.0;
    let py = f32(id.y) - f32(dims.y) / 2.0;
    let cos_r = cos(uniforms.rotation);
    let sin_r = sin(uniforms.rotation);
    let rpx = px * cos_r - py * sin_r;
    let rpy = px * sin_r + py * cos_r;
    let delta_x = rpx * uniforms.pixel_step_x;
    let delta_y = rpy * uniforms.pixel_step_y;

    let smooth_val = julia_perturbation(delta_x, delta_y);

    let color = colorize(smooth_val, uniforms.color_scheme);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
