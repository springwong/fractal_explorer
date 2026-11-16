/// Burning Ship fractal iteration with smooth coloring
/// Uses absolute values: z = (|Re(z)| + i|Im(z)|)² + c
// Bailout radius squared. Large, so the smooth iteration count is accurate and
// matches the deep-zoom shader (no colour jump at the switch).
const BAILOUT2: f32 = 256.0;

fn burning_ship(cx: f32, cy: f32) -> f32 {
    var zx: f32 = 0.0;
    var zy: f32 = 0.0;
    var iter: u32 = 0u;

    // Escape-time algorithm with absolute values
    while (iter < uniforms.max_iter && (zx * zx + zy * zy) < BAILOUT2) {
        let abs_zx = abs(zx);
        let abs_zy = abs(zy);
        let temp = abs_zx * abs_zx - abs_zy * abs_zy + cx;
        zy = 2.0 * abs_zx * abs_zy + cy;
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
    let smooth_val = burning_ship(c.x, c.y);
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(smooth_val));
}
