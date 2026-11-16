// Mandelbrot set, deep zoom: perturbation with rebasing (see rebasing.wgsl).
// z -> z² + c

/// (Z + w)² + (C + dc) - (Z² + C) = 2Zw + w² + dc
fn delta_step(z_ref: vec2<f32>, w: vec2<f32>, dc: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        2.0 * (z_ref.x * w.x - z_ref.y * w.y) + (w.x * w.x - w.y * w.y) + dc.x,
        2.0 * (z_ref.x * w.y + z_ref.y * w.x) + 2.0 * w.x * w.y + dc.y,
    );
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let smooth_val = perturbed_escape_time(pixel_delta(id.xy, dims));
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(smooth_val));
}
