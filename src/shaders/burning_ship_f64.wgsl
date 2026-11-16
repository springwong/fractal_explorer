// Burning Ship fractal with emulated double precision (2x f32) for deep zoom
// Arithmetic comes from double_single.wgsl.

/// Burning Ship iteration with full double-single precision
fn burning_ship_ds(cx_hi: f32, cx_lo: f32, cy_hi: f32, cy_lo: f32) -> f32 {
    var zrh: f32 = 0.0; var zrl: f32 = 0.0;
    var zih: f32 = 0.0; var zil: f32 = 0.0;
    var iter: u32 = 0u;

    while (iter < uniforms.max_iter) {
        let mag2 = zrh * zrh + zih * zih;
        if mag2 > 4.0 { break; }

        let abs_zr = ds_abs(zrh, zrl);
        let abs_zi = ds_abs(zih, zil);

        let zr_sq = ds_mul(abs_zr.x, abs_zr.y, abs_zr.x, abs_zr.y);
        let zi_sq = ds_mul(abs_zi.x, abs_zi.y, abs_zi.x, abs_zi.y);
        let zr_zi = ds_mul(abs_zr.x, abs_zr.y, abs_zi.x, abs_zi.y);
        let two_zr_zi_hi = zr_zi.x + zr_zi.x;
        let two_zr_zi_lo = zr_zi.y + zr_zi.y;

        let diff = ds_add(zr_sq.x, zr_sq.y, -zi_sq.x, -zi_sq.y);
        let new_zr = ds_add(diff.x, diff.y, cx_hi, cx_lo);
        let new_zi = ds_add(two_zr_zi_hi, two_zr_zi_lo, cy_hi, cy_lo);

        zrh = new_zr.x; zrl = new_zr.y;
        zih = new_zi.x; zil = new_zi.y;
        iter += 1u;
    }

    if iter == uniforms.max_iter { return 0.0; }
    let log_zn = log2(zrh * zrh + zih * zih) / 2.0;
    let nu = log2(log_zn / log2(2.0));
    return f32(iter) + 1.0 - nu;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let c = pixel_to_plane_ds(id.xy, dims);
    let smooth_val = burning_ship_ds(c.x, c.y, c.z, c.w);
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(smooth_val));
}
