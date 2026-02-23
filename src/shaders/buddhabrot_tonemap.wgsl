// Buddhabrot tonemap shader
// Reads accumulation buffer and writes colorized output to the storage texture.

struct Uniforms {
    center: vec2<f32>,
    zoom: f32,
    aspect_ratio: f32,
    max_iter: u32,
    fractal_type: u32,
    color_scheme: u32,
    c_real: f32,
    c_imag: f32,
    center_lo_x: f32,
    center_lo_y: f32,
    zoom_lo: f32,
    pixel_step_x: f32,
    pixel_step_y: f32,
    ref_escape_iter: u32,
    rotation: f32,
    _pad2: vec3<u32>,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
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
    let sample_count = f32(max(uniforms._pad2.z, 1u));
    let hits_per_frame = f32(count) / sample_count;
    let t = log2(hits_per_frame + 1.0) * 0.3;

    // Colorize based on scheme
    let color = sample_palette(t);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}

fn sample_palette(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let index = u32(fract(t * 0.05) * 255.0);
    let packed = palette_lut[index];
    let r = f32((packed >> 0u) & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, 1.0);
}
