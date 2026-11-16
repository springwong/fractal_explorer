// Celtic Mandelbrot, deep zoom: perturbation with rebasing (see rebasing.wgsl).
// x' = |x² - y²| + cx, y' = 2xy + cy

const DEGREE: f32 = 2.0;

fn delta_step(z_ref: vec2<f32>, w: vec2<f32>, dc: vec2<f32>) -> vec2<f32> {
    // x² - y² = (X² - Y²) + (2(X wx - Y wy) + wx² - wy²)
    let re_ref = z_ref.x * z_ref.x - z_ref.y * z_ref.y;
    let re_delta = 2.0 * (z_ref.x * w.x - z_ref.y * w.y) + (w.x * w.x - w.y * w.y);
    return vec2<f32>(
        diff_abs(re_ref, re_delta) + dc.x,
        2.0 * (z_ref.x * w.y + z_ref.y * w.x + w.x * w.y) + dc.y,
    );
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let smooth_val = perturbed_escape_time(pixel_delta(id.xy, dims));
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(smooth_val));
}
