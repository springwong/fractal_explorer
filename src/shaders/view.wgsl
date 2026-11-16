// Pixel -> plane mapping shared by the f32 shaders and the perturbation shaders.

/// Pixel offset from the view centre, rotated. Units: pixels.
fn rotated_pixel(id: vec2<u32>, dims: vec2<u32>) -> vec2<f32> {
    let p = vec2<f32>(id) - vec2<f32>(dims) / 2.0;
    let cos_r = cos(uniforms.rotation);
    let sin_r = sin(uniforms.rotation);
    return vec2<f32>(p.x * cos_r - p.y * sin_r, p.x * sin_r + p.y * cos_r);
}

/// Plane coordinate of a pixel in plain f32 (good to about 5e3 zoom).
fn pixel_to_plane(id: vec2<u32>, dims: vec2<u32>) -> vec2<f32> {
    let rotated = rotated_pixel(id, dims) / f32(dims.y);
    return uniforms.center + (rotated / uniforms.zoom) * vec2<f32>(1.0, -1.0);
}

/// Offset of a pixel from the view centre in plane units, using the per-pixel step
/// the CPU computed in f64. Stays accurate at any zoom.
fn pixel_delta(id: vec2<u32>, dims: vec2<u32>) -> vec2<f32> {
    return rotated_pixel(id, dims) * vec2<f32>(uniforms.pixel_step_x, uniforms.pixel_step_y);
}
