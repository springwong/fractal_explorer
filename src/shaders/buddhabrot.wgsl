// Buddhabrot accumulation compute shader
// Each thread traces one random orbit; escaping orbits atomically increment the accumulation buffer.

struct Uniforms {
    center: vec2<f32>,       // offset 0, 8 bytes
    zoom: f32,               // offset 8, 4 bytes
    aspect_ratio: f32,       // offset 12, 4 bytes
    max_iter: u32,           // offset 16, 4 bytes
    fractal_type: u32,       // offset 20, 4 bytes
    color_scheme: u32,       // offset 24, 4 bytes
    c_real: f32,             // offset 28, 4 bytes
    c_imag: f32,             // offset 32, 4 bytes
    center_lo_x: f32,        // offset 36, 4 bytes
    center_lo_y: f32,        // offset 40, 4 bytes
    zoom_lo: f32,            // offset 44, 4 bytes
    pixel_step_x: f32,       // offset 48, 4 bytes
    pixel_step_y: f32,       // offset 52, 4 bytes
    ref_escape_iter: u32,    // offset 56, 4 bytes (repurposed as frame seed)
    rotation: f32,           // offset 60, 4 bytes
    _pad2: vec3<u32>,        // offset 64, 12 bytes
    _pad3: u32,              // offset 76, 4 bytes
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> accum_buf: array<atomic<u32>>;

// Hash function for pseudo-random number generation
fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate a float in [0, 1) from a u32
fn rand_float(hash: u32) -> f32 {
    return f32(hash) / 4294967296.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Screen dimensions passed directly via _pad2 to avoid float roundtrip precision issues
    let width = uniforms._pad2.x;
    let height = uniforms._pad2.y;

    if width == 0u || height == 0u {
        return;
    }

    // Generate unique seed from thread ID and frame seed
    let thread_id = id.x;
    var seed = pcg_hash(thread_id ^ (uniforms.ref_escape_iter * 1664525u + 1013904223u));

    // Generate random c in the range [-2.5, 1.0] x [-1.25, 1.25]
    seed = pcg_hash(seed);
    let cx = rand_float(seed) * 3.5 - 2.5;
    seed = pcg_hash(seed);
    let cy = rand_float(seed) * 2.5 - 1.25;

    // First pass: iterate to check if orbit escapes
    var zx: f32 = 0.0;
    var zy: f32 = 0.0;
    var iter: u32 = 0u;
    let max_i = uniforms.max_iter;

    while (iter < max_i && (zx * zx + zy * zy) < 4.0) {
        let temp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = temp;
        iter += 1u;
    }

    // Only process escaping orbits (not in the Mandelbrot set)
    if iter == max_i {
        return;
    }

    // Second pass: replay orbit and accumulate
    zx = 0.0;
    zy = 0.0;

    for (var i: u32 = 0u; i < iter; i += 1u) {
        let temp = zx * zx - zy * zy + cx;
        zy = 2.0 * zx * zy + cy;
        zx = temp;

        // Map orbit point to pixel coordinates via current camera view
        let rel = vec2<f32>(zx, zy) - uniforms.center;
        let cos_r = cos(uniforms.rotation);
        let sin_r = sin(uniforms.rotation);
        let rotated = vec2<f32>(
            rel.x * cos_r + rel.y * sin_r,
            -rel.x * sin_r + rel.y * cos_r,
        );
        let screen = rotated * uniforms.zoom * vec2<f32>(1.0, -1.0) * f32(height) + vec2<f32>(f32(width), f32(height)) / 2.0;

        let px = i32(screen.x);
        let py = i32(screen.y);

        if px >= 0 && px < i32(width) && py >= 0 && py < i32(height) {
            let idx = u32(py) * width + u32(px);
            atomicAdd(&accum_buf[idx], 1u);
        }
    }
}
