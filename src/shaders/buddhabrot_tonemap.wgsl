// Buddhabrot tonemap shader
// Reads accumulation buffer and writes colorized output to the storage texture.

@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> accum_buf: array<u32>;
@group(0) @binding(3) var<storage, read> palette_lut: array<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);

    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    let idx = id.y * dims.x + id.x;
    let count = accum_buf[idx];

    if count == 0u {
        textureStore(output_texture, vec2<i32>(id.xy), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Normalize by sample count to get per-frame hit rate, then log tonemap
    // This stays stable as frames accumulate, preserving contrast
    let sample_count = f32(max(uniforms.extra.z, 1u));
    let hits_per_frame = f32(count) / sample_count;
    let t = log2(hits_per_frame + 1.0) * 0.3;

    // Colorize based on scheme
    let color = sample_palette(t);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
