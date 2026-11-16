// Newton fractal of z³ - 1 (see newton_core.wgsl): z -> z - (z³-1)/(3z²),
// coloured by the root each point ends up at

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(newton_value(newton_direct(pixel_to_plane(id.xy, dims), 0u))));
}
