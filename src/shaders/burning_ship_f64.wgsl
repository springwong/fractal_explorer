// Burning Ship, deep zoom: perturbation with rebasing (see rebasing.wgsl).
// z -> (|x| + i|y|)² + c, i.e. x' = x² - y² + cx, y' = 2|xy| + cy

/// |c + d| - |c| without the cancellation the direct form has when |d| << |c|
fn diff_abs(c: f32, d: f32) -> f32 {
    if c >= 0.0 {
        if c + d >= 0.0 { return d; }
        return -(2.0 * c + d);
    }
    if c + d > 0.0 { return 2.0 * c + d; }
    return -d;
}

fn delta_step(z_ref: vec2<f32>, w: vec2<f32>, dc: vec2<f32>) -> vec2<f32> {
    // xy = XY + (X wy + wx Y + wx wy)
    let xy_ref = z_ref.x * z_ref.y;
    let xy_delta = z_ref.x * w.y + w.x * z_ref.y + w.x * w.y;
    return vec2<f32>(
        2.0 * (z_ref.x * w.x - z_ref.y * w.y) + (w.x * w.x - w.y * w.y) + dc.x,
        2.0 * diff_abs(xy_ref, xy_delta) + dc.y,
    );
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let smooth_val = perturbed_escape_time(pixel_delta(id.xy, dims));
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(smooth_val));
}
