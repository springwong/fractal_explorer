// Fractal uniforms matching Rust FractalUniforms struct
struct Uniforms {
    center: vec2<f32>,
    zoom: f32,
    aspect_ratio: f32,
    max_iter: u32,
    _padding: vec3<u32>,
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

/// Simple HSV-based colorization
fn colorize(t: f32) -> vec4<f32> {
    if t == 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0); // Inside set = black
    }

    // Map iteration count to hue
    let hue = fract(t * 0.05);
    let sat = 0.8;
    let val = 0.9;

    // HSV to RGB conversion
    let h = hue * 6.0;
    let i = floor(h);
    let f = h - i;
    let p = val * (1.0 - sat);
    let q = val * (1.0 - sat * f);
    let t_val = val * (1.0 - sat * (1.0 - f));

    var rgb: vec3<f32>;
    if i == 0.0 {
        rgb = vec3<f32>(val, t_val, p);
    } else if i == 1.0 {
        rgb = vec3<f32>(q, val, p);
    } else if i == 2.0 {
        rgb = vec3<f32>(p, val, t_val);
    } else if i == 3.0 {
        rgb = vec3<f32>(p, q, val);
    } else if i == 4.0 {
        rgb = vec3<f32>(t_val, p, val);
    } else {
        rgb = vec3<f32>(val, p, q);
    }

    return vec4<f32>(rgb, 1.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);

    // Bounds check
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    // Convert pixel coordinates to complex plane coordinates
    let uv = (vec2<f32>(id.xy) - vec2<f32>(dims) / 2.0) / f32(dims.y) / uniforms.zoom;
    let c = uniforms.center + uv * vec2<f32>(uniforms.aspect_ratio, -1.0);

    // Calculate smooth iteration count
    let smooth_val = mandelbrot(c.x, c.y);

    // Colorize and write to output texture
    let color = colorize(smooth_val);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
