// Multibrot (degree 3), deep zoom: perturbation with rebasing (see rebasing.wgsl).
// z -> z³ + c

const DEGREE: f32 = 3.0;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/// (Z + w)³ + (C + dc) - (Z³ + C) = w (3Z² + 3Zw + w²) + dc
fn delta_step(z_ref: vec2<f32>, w: vec2<f32>, dc: vec2<f32>) -> vec2<f32> {
    return cmul(w, 3.0 * cmul(z_ref, z_ref + w) + cmul(w, w)) + dc;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let smooth_val = perturbed_escape_time(pixel_delta(id.xy, dims));
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(smooth_val));
}
