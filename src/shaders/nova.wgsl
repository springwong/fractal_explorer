// Nova fractal: Newton's method with perturbation
// z → z - (z³-1)/(3z²) + c

/// Nova fractal iteration with convergence-based smooth coloring
fn nova(px: f32, py: f32, cx: f32, cy: f32) -> f32 {
    var zx: f32 = px;
    var zy: f32 = py;
    var iter: u32 = 0u;
    let tolerance: f32 = 1.0e-6;
    let bailout: f32 = 1.0e8;

    while (iter < uniforms.max_iter) {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        let mag2 = zx2 + zy2;

        // Bail out if z escapes
        if mag2 > bailout {
            break;
        }

        // Compute z³ = (zx + i*zy)³
        // z² = (zx² - zy²) + i*(2*zx*zy)
        // z³ = z² * z = (zx²-zy²)*zx - 2*zx*zy*zy + i*((zx²-zy²)*zy + 2*zx*zy*zx)
        //     = zx³ - 3*zx*zy² + i*(3*zx²*zy - zy³)
        let z3x = zx * zx2 - 3.0 * zx * zy2;
        let z3y = 3.0 * zx2 * zy - zy * zy2;

        // z³ - 1
        let num_x = z3x - 1.0;
        let num_y = z3y;

        // 3z² = 3*(zx² - zy²) + i*(6*zx*zy)
        let den_x = 3.0 * (zx2 - zy2);
        let den_y = 6.0 * zx * zy;

        // Complex division: (z³-1) / (3z²)
        let den_mag2 = den_x * den_x + den_y * den_y;
        if den_mag2 < 1.0e-20 {
            break;
        }
        let div_x = (num_x * den_x + num_y * den_y) / den_mag2;
        let div_y = (num_y * den_x - num_x * den_y) / den_mag2;

        // z_new = z - (z³-1)/(3z²) + c
        let new_zx = zx - div_x + cx;
        let new_zy = zy - div_y + cy;

        // Check convergence: |z_new - z| < tolerance
        let dx = new_zx - zx;
        let dy = new_zy - zy;
        let delta2 = dx * dx + dy * dy;

        zx = new_zx;
        zy = new_zy;
        iter += 1u;

        if delta2 < tolerance {
            // Smooth coloring based on convergence
            let smooth_val = f32(iter) - log2(log2(delta2 + 1.0e-20) / log2(tolerance)) ;
            return max(smooth_val, 0.0);
        }
    }

    // Did not converge — color as "inside" or based on escape
    if iter == uniforms.max_iter {
        return 0.0;
    }
    // Escaped: use iteration count
    return f32(iter);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let z = pixel_to_plane(id.xy, dims);
    let smooth_val = nova(z.x, z.y, uniforms.c_real, uniforms.c_imag);
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(smooth_val));
}
