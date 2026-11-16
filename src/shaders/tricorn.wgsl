/// Tricorn (Mandelbar) fractal iteration with smooth coloring
/// Uses conjugate: z = conj(z)² + c = (Re(z) - i*Im(z))² + c
fn tricorn(cx: f32, cy: f32) -> f32 {
    var zx: f32 = 0.0;
    var zy: f32 = 0.0;
    var iter: u32 = 0u;

    // Escape-time algorithm with conjugate
    while (iter < uniforms.max_iter && (zx * zx + zy * zy) < 4.0) {
        // Conjugate: (zx - i*zy)² = zx² - zy² - 2i*zx*zy
        let temp = zx * zx - zy * zy + cx;
        zy = -2.0 * zx * zy + cy; // Note the negative sign
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

    let c = pixel_to_plane(id.xy, dims);
    let smooth_val = tricorn(c.x, c.y);
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(smooth_val));
}
