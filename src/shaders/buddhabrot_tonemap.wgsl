// Buddhabrot tonemap shader
// Reads accumulation buffer and writes colorized output to the storage texture.

@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> accum_buf: array<u32>;
@group(0) @binding(3) var<storage, read> palette_lut: array<u32>;

// Pixel count at which an average pixel collects about one hit per batch
const EVEN_SPREAD_PIXELS: f32 = 12000.0;

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

    // Hits per batch, relative to what a pixel would collect if the batch were spread
    // evenly over the image, then log tonemap. Stable as batches accumulate, and the
    // same brightness at every output size: a batch is a fixed number of orbits, so
    // the hits each pixel gets from it shrink as the pixel count grows.
    let sample_count = f32(max(uniforms.extra.z, 1u));
    let density = f32(count) / sample_count * f32(dims.x * dims.y) / EVEN_SPREAD_PIXELS;
    let t = log2(density + 1.0) * 0.3;

    // Colorize based on scheme
    let color = sample_palette(t);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
