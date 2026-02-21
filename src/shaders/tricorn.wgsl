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
    _pad: u32,               // offset 60, 4 bytes
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

/// Tricorn (Mandelbar) fractal iteration with smooth coloring
/// Uses conjugate: z = conj(z)² + c = (Re(z) - i*Im(z))² + c
fn tricorn(cx: f32, cy: f32) -> f32 {
    var zx: f32 = 0.0;
    var zy: f32 = 0.0;
    var iter: u32 = 0u;

    // Escape-time algorithm with conjugate
    while (iter < uniforms.max_iter && (zx * zx + zy * zy) < 4.0) {
        // Conjugate: (zx - i*zy)² = zx² - zy² - 2i*zx*zy
        let temp = zx * zx - zy * zy + cx;
        zy = -2.0 * zx * zy + cy; // Note the negative sign
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

/// Coloring functions
fn colorize_smooth(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
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

fn colorize_fire(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let n = fract(t * 0.03);
    return vec4<f32>(min(1.0, n * 2.0), max(0.0, min(1.0, (n - 0.3) * 2.5)), max(0.0, min(1.0, (n - 0.7) * 3.3)), 1.0);
}

fn colorize_ocean(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.1, 1.0); }
    let n = fract(t * 0.04);
    return vec4<f32>(max(0.0, min(1.0, (n - 0.6) * 2.5)), max(0.0, min(1.0, (n - 0.2) * 1.8)), min(1.0, 0.3 + n * 0.7), 1.0);
}

fn colorize_grayscale(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let i = fract(t * 0.05);
    return vec4<f32>(i, i, i, 1.0);
}

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
    let uv = (vec2<f32>(id.xy) - vec2<f32>(dims) / 2.0) / f32(dims.y);
    let c = uniforms.center + (uv / uniforms.zoom) * vec2<f32>(1.0, -1.0);

    // Calculate smooth iteration count
    let smooth_val = tricorn(c.x, c.y);

    // Colorize and write to output texture
    let color = colorize(smooth_val, uniforms.color_scheme);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
