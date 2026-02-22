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
    let color = colorize(t, uniforms.color_scheme);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}

/// Smooth rainbow colorization (t is normalized ~0-1+)
fn colorize_smooth(t: f32) -> vec4<f32> {
    let hue = fract(t * 1.5);
    let sat = 0.7;
    let val = min(1.0, t);
    let h = hue * 6.0;
    let i = floor(h);
    let f = h - i;
    let p = val * (1.0 - sat);
    let q = val * (1.0 - sat * f);
    let t_val = val * (1.0 - sat * (1.0 - f));
    var rgb: vec3<f32>;
    if i == 0.0 { rgb = vec3<f32>(val, t_val, p); }
    else if i == 1.0 { rgb = vec3<f32>(q, val, p); }
    else if i == 2.0 { rgb = vec3<f32>(p, val, t_val); }
    else if i == 3.0 { rgb = vec3<f32>(p, q, val); }
    else if i == 4.0 { rgb = vec3<f32>(t_val, p, val); }
    else { rgb = vec3<f32>(val, p, q); }
    return vec4<f32>(rgb, 1.0);
}

/// Fire colorization (t is normalized ~0-1+)
fn colorize_fire(t: f32) -> vec4<f32> {
    let n = min(1.0, t);
    let r = min(1.0, n * 2.0);
    let g = max(0.0, min(1.0, (n - 0.3) * 2.5));
    let b = max(0.0, min(1.0, (n - 0.7) * 3.3));
    return vec4<f32>(r, g, b, 1.0);
}

/// Ocean colorization (t is normalized ~0-1+)
fn colorize_ocean(t: f32) -> vec4<f32> {
    let n = min(1.0, t);
    let r = max(0.0, min(1.0, (n - 0.6) * 2.5));
    let g = max(0.0, min(1.0, (n - 0.2) * 1.8));
    let b = min(1.0, 0.3 + n * 0.7);
    return vec4<f32>(r, g, b, 1.0);
}

/// Grayscale colorization (t is normalized ~0-1+)
fn colorize_grayscale(t: f32) -> vec4<f32> {
    let intensity = min(1.0, t);
    return vec4<f32>(intensity, intensity, intensity, 1.0);
}

/// Nebula colorization - tuned for Buddhabrot's density distribution (t is normalized ~0-1+)
fn colorize_nebula(t: f32) -> vec4<f32> {
    let n = min(1.0, t);
    let r = pow(n, 0.4) * 0.8;
    let g = pow(n, 0.7) * 0.6;
    let b = pow(n, 1.2);
    return vec4<f32>(r, g, b, 1.0);
}

fn colorize(t: f32, scheme: u32) -> vec4<f32> {
    switch scheme {
        case 0u: { return colorize_nebula(t); }
        case 1u: { return colorize_fire(t); }
        case 2u: { return colorize_ocean(t); }
        case 3u: { return colorize_grayscale(t); }
        default: { return colorize_smooth(t); }
    }
}
