// Mandelbrot set with perturbation theory for deep zoom
// Reference orbit computed in f64 on CPU, per-pixel deltas in f32 on GPU.
// Includes glitch detection with rebasing to handle boundary artifacts.
struct Uniforms {
    center: vec2<f32>,       // offset 0, 8 bytes (hi part)
    zoom: f32,               // offset 8, 4 bytes (hi part)
    aspect_ratio: f32,       // offset 12, 4 bytes
    max_iter: u32,           // offset 16, 4 bytes
    fractal_type: u32,       // offset 20, 4 bytes
    color_scheme: u32,       // offset 24, 4 bytes
    c_real: f32,             // offset 28, 4 bytes
    c_imag: f32,             // offset 32, 4 bytes
    center_lo_x: f32,        // offset 36, 4 bytes
    center_lo_y: f32,        // offset 40, 4 bytes
    zoom_lo: f32,            // offset 44, 4 bytes
    pixel_step_x: f32,       // offset 48, 4 bytes (1.0 / (zoom * height), computed on CPU in f64)
    pixel_step_y: f32,       // offset 52, 4 bytes (-1.0 / (zoom * height), computed on CPU in f64)
    ref_escape_iter: u32,    // offset 56, 4 bytes
    rotation: f32,           // offset 60, 4 bytes
    _pad2: vec3<u32>,        // offset 64, 12 bytes
    _pad3: u32,              // offset 76, 4 bytes
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
// Reference orbit: flat array of f32 pairs [zx, zy] for iterations 0..max_iter
@group(0) @binding(2) var<storage, read> ref_orbit: array<f32>;
@group(0) @binding(3) var<storage, read> palette_lut: array<u32>;

// Bailout radius squared (using large value for smooth coloring)
const BAILOUT2: f32 = 256.0;

// Glitch detection threshold: if |w|^2 > GLITCH_TOLERANCE * |Z|^2,
// the perturbation has lost accuracy and we rebase.
const GLITCH_TOLERANCE: f32 = 1e-3;

/// Perturbation-based Mandelbrot iteration with glitch detection and rebasing.
///
/// Reference orbit Z*_n is precomputed on CPU in f64.
/// Per-pixel perturbation: w_{n+1} = 2 * Z*_n * w_n + w_n^2 + delta_c
/// Full orbit: Z_n = Z*_n + w_n
///
/// When |w_n| grows too large relative to |Z*_n|, precision is lost (glitch).
/// Fix: rebase by computing c from full Z, then restart with w=0 from next ref step.
fn mandelbrot_perturbation(delta_x: f32, delta_y: f32) -> f32 {
    var wx: f32 = 0.0;
    var wy: f32 = 0.0;
    var iter: u32 = 0u;
    let ref_max = uniforms.ref_escape_iter;

    // The pixel's c offset from center (stays constant)
    let dc_x = delta_x;
    let dc_y = delta_y;

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
            return direct_iterate(full_zx, full_zy, dc_x, dc_y, iter);
        }

        // Glitch detection: if |w|^2 > tolerance * |Z|^2, perturbation is unreliable
        let w_mag2 = wx * wx + wy * wy;
        if w_mag2 > GLITCH_TOLERANCE * full_mag2 {
            // Rebase: treat full Z as if it were the new reference at this iteration
            // and restart perturbation from w=0 by switching to direct iteration.
            return direct_iterate(full_zx, full_zy, dc_x, dc_y, iter);
        }

        // Perturbation step: w_{n+1} = 2 * Z*_n * w_n + w_n^2 + delta_c
        let new_wx = 2.0 * (zx_ref * wx - zy_ref * wy) + (wx * wx - wy * wy) + dc_x;
        let new_wy = 2.0 * (zx_ref * wy + zy_ref * wx) + 2.0 * wx * wy + dc_y;

        wx = new_wx;
        wy = new_wy;
        iter += 1u;
    }

    return 0.0; // Inside the set
}

/// Direct f32 iteration starting from a given z value.
/// Used as fallback when perturbation glitches or reference orbit escapes.
/// c = center + delta (approximated in f32 since we're past the precision-critical part)
fn direct_iterate(zx_start: f32, zy_start: f32, dc_x: f32, dc_y: f32, start_iter: u32) -> f32 {
    // c = center + delta_c
    let cx = uniforms.center.x + dc_x;
    let cy = uniforms.center.y + dc_y;
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

    // Per-pixel delta from center, computed using pixel_step (calculated on CPU in f64)
    let px = f32(id.x) - f32(dims.x) / 2.0;
    let py = f32(id.y) - f32(dims.y) / 2.0;
    let cos_r = cos(uniforms.rotation);
    let sin_r = sin(uniforms.rotation);
    let rpx = px * cos_r - py * sin_r;
    let rpy = px * sin_r + py * cos_r;
    let delta_x = rpx * uniforms.pixel_step_x;
    let delta_y = rpy * uniforms.pixel_step_y;

    let smooth_val = mandelbrot_perturbation(delta_x, delta_y);

    let color = sample_palette(smooth_val);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
