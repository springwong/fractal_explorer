// Julia set with perturbation theory for deep zoom
// Reference orbit computed in f64 on CPU, per-pixel deltas in f32 on GPU.
// Julia perturbation: w_{n+1} = 2 * Z*_n * w_n + w_n^2  (no +delta_c since c is same for all pixels)
// w_0 = pixel_offset (initial z differs per pixel)

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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let delta = pixel_delta(id.xy, dims);
    let smooth_val = julia_perturbation(delta.x, delta.y);
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(smooth_val));
}
