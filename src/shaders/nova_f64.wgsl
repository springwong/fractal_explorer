// Nova fractal with emulated double precision (2x f32) for deep zoom
// Arithmetic comes from double_single.wgsl.

/// Nova fractal iteration with double-single precision
fn nova_ds(px_hi: f32, px_lo: f32, py_hi: f32, py_lo: f32, cx: f32, cy: f32) -> f32 {
    var zxh: f32 = px_hi; var zxl: f32 = px_lo;
    var zyh: f32 = py_hi; var zyl: f32 = py_lo;
    var iter: u32 = 0u;
    let tolerance: f32 = 1.0e-6;
    let bailout: f32 = 1.0e8;

    while (iter < uniforms.max_iter) {
        let mag2 = zxh * zxh + zyh * zyh;
        if mag2 > bailout { break; }

        // Compute z² with double-single precision
        let zx2 = ds_mul(zxh, zxl, zxh, zxl);
        let zy2 = ds_mul(zyh, zyl, zyh, zyl);
        let zxzy = ds_mul(zxh, zxl, zyh, zyl);

        // z³ = z² * z
        // z³_real = zx*(zx²-3*zy²), z³_imag = zy*(3*zx²-zy²)
        let three_zy2 = ds_mul(3.0, 0.0, zy2.x, zy2.y);
        let three_zx2 = ds_mul(3.0, 0.0, zx2.x, zx2.y);
        let zx2_minus_3zy2 = ds_add(zx2.x, zx2.y, -three_zy2.x, -three_zy2.y);
        let three_zx2_minus_zy2 = ds_add(three_zx2.x, three_zx2.y, -zy2.x, -zy2.y);
        let z3x = ds_mul(zxh, zxl, zx2_minus_3zy2.x, zx2_minus_3zy2.y);
        let z3y = ds_mul(zyh, zyl, three_zx2_minus_zy2.x, three_zx2_minus_zy2.y);

        // numerator: z³ - 1
        let num_x = ds_add(z3x.x, z3x.y, -1.0, 0.0);
        let num_y = z3y;

        // denominator: 3z²
        let zx2_minus_zy2 = ds_add(zx2.x, zx2.y, -zy2.x, -zy2.y);
        let den_x = ds_mul(3.0, 0.0, zx2_minus_zy2.x, zx2_minus_zy2.y);
        let two_zxzy = ds_mul(2.0, 0.0, zxzy.x, zxzy.y);
        let den_y = ds_mul(3.0, 0.0, two_zxzy.x, two_zxzy.y);

        // denominator magnitude²
        let den_mag2 = den_x.x * den_x.x + den_y.x * den_y.x;
        if den_mag2 < 1.0e-20 { break; }

        // Complex division (use f32 for the division step, sufficient precision)
        let div_x = (num_x.x * den_x.x + num_y.x * den_y.x) / den_mag2;
        let div_y = (num_y.x * den_x.x - num_x.x * den_y.x) / den_mag2;

        // z_new = z - div + c
        let sub = ds_add(zxh, zxl, -div_x, 0.0);
        let new_zx = ds_add(sub.x, sub.y, cx, 0.0);
        let sub_y = ds_add(zyh, zyl, -div_y, 0.0);
        let new_zy = ds_add(sub_y.x, sub_y.y, cy, 0.0);

        // Check convergence
        let dx = new_zx.x - zxh;
        let dy = new_zy.x - zyh;
        let delta2 = dx * dx + dy * dy;

        zxh = new_zx.x; zxl = new_zx.y;
        zyh = new_zy.x; zyl = new_zy.y;
        iter += 1u;

        if delta2 < tolerance {
            let smooth_val = f32(iter) - log2(log2(delta2 + 1.0e-20) / log2(tolerance));
            return max(smooth_val, 0.0);
        }
    }

    if iter == uniforms.max_iter { return 0.0; }
    return f32(iter);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let z = pixel_to_plane_ds(id.xy, dims);
    let smooth_val = nova_ds(z.x, z.y, z.z, z.w, uniforms.c_real, uniforms.c_imag);
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(smooth_val));
}
