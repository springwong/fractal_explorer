// Multibrot set of degree 3: z -> z³ + c
// Each step cubes |z|, which sets the base of the smooth colouring
const DEGREE: f32 = 3.0;
// Bailout radius squared. Large, so the smooth iteration count is accurate and
// matches the deep-zoom shader (no colour jump at the switch).
const BAILOUT2: f32 = 256.0;

/// Smooth escape time of the point `c` (0 = inside)
fn multibrot(c: vec2<f32>) -> f32 {
    var z = vec2<f32>(0.0, 0.0);
    var iter: u32 = 0u;

    while (iter < uniforms.max_iter && dot(z, z) < BAILOUT2) {
        // z³ = (x³ - 3xy²) + i(3x²y - y³)
        let x2 = z.x * z.x;
        let y2 = z.y * z.y;
        z = vec2<f32>(z.x * (x2 - 3.0 * y2), z.y * (3.0 * x2 - y2)) + c;
        iter += 1u;
    }

    if iter == uniforms.max_iter {
        return 0.0; // Inside the set
    }
    return f32(iter) + 1.0 - log2(log2(dot(z, z)) / 2.0) / log2(DEGREE);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let c = pixel_to_plane(id.xy, dims);
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(multibrot(c)));
}
