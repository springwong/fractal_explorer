// Perpendicular Mandelbrot: z -> (|x| - iy)² + c, i.e. x' = x² - y² + cx, y' = -2|x|y + cy
// Bailout radius squared. Large, so the smooth iteration count is accurate and
// matches the deep-zoom shader (no colour jump at the switch).
const BAILOUT2: f32 = 256.0;

/// Smooth escape time of the point `c` (0 = inside)
fn perpendicular(c: vec2<f32>) -> f32 {
    var z = vec2<f32>(0.0, 0.0);
    var iter: u32 = 0u;

    while (iter < uniforms.max_iter && dot(z, z) < BAILOUT2) {
        z = vec2<f32>(z.x * z.x - z.y * z.y, -2.0 * abs(z.x) * z.y) + c;
        iter += 1u;
    }

    if iter == uniforms.max_iter {
        return 0.0; // Inside the set
    }
    return f32(iter) + 1.0 - log2(log2(dot(z, z)) / 2.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let c = pixel_to_plane(id.xy, dims);
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(perpendicular(c)));
}
