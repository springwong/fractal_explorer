// Perpendicular Mandelbrot, deep zoom: perturbation with rebasing (see rebasing.wgsl).
// x' = x² - y² + cx, y' = -2|x|y + cy

const DEGREE: f32 = 2.0;

fn delta_step(z_ref: vec2<f32>, w: vec2<f32>, dc: vec2<f32>) -> vec2<f32> {
    // |x| y = |X| Y + ((|X + wx| - |X|) (Y + wy) + |X| wy)
    let abs_delta = diff_abs(z_ref.x, w.x) * (z_ref.y + w.y) + abs(z_ref.x) * w.y;
    return vec2<f32>(
        2.0 * (z_ref.x * w.x - z_ref.y * w.y) + (w.x * w.x - w.y * w.y) + dc.x,
        -2.0 * abs_delta + dc.y,
    );
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let smooth_val = perturbed_escape_time(pixel_delta(id.xy, dims));
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(smooth_val));
}
