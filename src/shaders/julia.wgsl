/// Julia set iteration with smooth coloring
/// For Julia sets, the initial z = pixel coordinate, c = fixed parameter
fn julia(zx_init: f32, zy_init: f32, cx: f32, cy: f32) -> f32 {
    var zx: f32 = zx_init;
    var zy: f32 = zy_init;
    var iter: u32 = 0u;

    // Escape-time algorithm
    while (iter < uniforms.max_iter && (zx * zx + zy * zy) < 4.0) {
        let temp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = temp;
        iter += 1u;
    }

    // Smooth iteration count to eliminate banding
    if iter == uniforms.max_iter {
        return 0.0; // Inside the set
    }

    let log_zn = log2(zx * zx + zy * zy) / 2.0;
    let nu = log2(log_zn / log2(2.0));
    return f32(iter) + 1.0 - nu;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    // The pixel is the starting z; c comes from the uniforms
    let z = pixel_to_plane(id.xy, dims);
    let smooth_val = julia(z.x, z.y, uniforms.c_real, uniforms.c_imag);
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(smooth_val));
}
