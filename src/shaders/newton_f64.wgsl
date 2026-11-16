// Newton fractal of z³ - 1, deep zoom (see newton_perturbation.wgsl)

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(newton_value(newton_perturbation(pixel_delta(id.xy, dims)))));
}
