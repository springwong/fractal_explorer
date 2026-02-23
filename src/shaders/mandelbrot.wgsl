// Fractal uniforms matching Rust FractalUniforms struct (64 bytes)
struct Uniforms {
    center: vec2<f32>,       // offset 0, 8 bytes
    zoom: f32,               // offset 8, 4 bytes
    aspect_ratio: f32,       // offset 12, 4 bytes
    max_iter: u32,           // offset 16, 4 bytes
    fractal_type: u32,       // offset 20, 4 bytes
    color_scheme: u32,       // offset 24, 4 bytes
    c_real: f32,             // offset 28, 4 bytes
    c_imag: f32,             // offset 32, 4 bytes
    center_lo_x: f32,        // offset 36, 4 bytes (emulated double)
    center_lo_y: f32,        // offset 40, 4 bytes (emulated double)
    zoom_lo: f32,            // offset 44, 4 bytes (emulated double)
    pixel_step_x: f32,       // offset 48, 4 bytes
    pixel_step_y: f32,       // offset 52, 4 bytes
    ref_escape_iter: u32,    // offset 56, 4 bytes
    rotation: f32,           // offset 60, 4 bytes
    _pad2: vec3<u32>,        // offset 64, 12 bytes
    _pad3: u32,              // offset 76, 4 bytes
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> palette_lut: array<u32>;

/// Mandelbrot set iteration with smooth coloring
fn mandelbrot(cx: f32, cy: f32) -> f32 {
    var zx: f32 = 0.0;
    var zy: f32 = 0.0;
    var iter: u32 = 0u;

    // Escape-time algorithm
    while (iter < uniforms.max_iter && (zx * zx + zy * zy) < 4.0) {
        let temp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = temp;
        iter += 1u;
    }

    // Smooth iteration count to eliminate banding
    if iter == uniforms.max_iter {
        return 0.0; // Inside the set
    }

    let log_zn = log2(zx * zx + zy * zy) / 2.0;
    let nu = log2(log_zn / log2(2.0));
    return f32(iter) + 1.0 - nu;
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

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);

    // Bounds check
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    // Convert pixel coordinates to complex plane coordinates
    // Center at origin, normalize by height to maintain aspect ratio
    let uv = (vec2<f32>(id.xy) - vec2<f32>(dims) / 2.0) / f32(dims.y);
    let cos_r = cos(uniforms.rotation);
    let sin_r = sin(uniforms.rotation);
    let rotated = vec2<f32>(uv.x * cos_r - uv.y * sin_r, uv.x * sin_r + uv.y * cos_r);
    let c = uniforms.center + (rotated / uniforms.zoom) * vec2<f32>(1.0, -1.0);

    // Calculate smooth iteration count
    let smooth_val = mandelbrot(c.x, c.y);

    // Colorize and write to output texture
    let color = sample_palette(smooth_val);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
