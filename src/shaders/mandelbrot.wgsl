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

/// Smooth rainbow colorization
fn colorize_smooth(t: f32) -> vec4<f32> {
    if t == 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let hue = fract(t * 0.05);
    let sat = 0.8;
    let val = 0.9;
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

/// Fire colorization
fn colorize_fire(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let n = fract(t * 0.03);
    let r = min(1.0, n * 2.0);
    let g = max(0.0, min(1.0, (n - 0.3) * 2.5));
    let b = max(0.0, min(1.0, (n - 0.7) * 3.3));
    return vec4<f32>(r, g, b, 1.0);
}

/// Ocean colorization
fn colorize_ocean(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.1, 1.0); }
    let n = fract(t * 0.04);
    let r = max(0.0, min(1.0, (n - 0.6) * 2.5));
    let g = max(0.0, min(1.0, (n - 0.2) * 1.8));
    let b = min(1.0, 0.3 + n * 0.7);
    return vec4<f32>(r, g, b, 1.0);
}

/// Grayscale colorization
fn colorize_grayscale(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let intensity = fract(t * 0.05);
    return vec4<f32>(intensity, intensity, intensity, 1.0);
}

/// Main colorization dispatcher
fn colorize(t: f32, scheme: u32) -> vec4<f32> {
    switch scheme {
        case 0u: { return colorize_smooth(t); }
        case 1u: { return colorize_fire(t); }
        case 2u: { return colorize_ocean(t); }
        case 3u: { return colorize_grayscale(t); }
        default: { return colorize_smooth(t); }
    }
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
    let color = colorize(smooth_val, uniforms.color_scheme);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
