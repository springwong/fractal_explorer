// Buddhabrot accumulation compute shader
// Each thread traces one random orbit; escaping orbits atomically increment the accumulation buffer.

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
    // Screen dimensions passed directly via `extra` to avoid float roundtrip precision issues
    let width = uniforms.extra.x;
    let height = uniforms.extra.y;

    if width == 0u || height == 0u {
        return;
    }

    // Generate unique seed from thread ID and frame seed
    let thread_id = id.x;
    var seed = pcg_hash(thread_id ^ (uniforms.ref_escape_iter * 1664525u + 1013904223u));

    // Generate random c in a region covering the Mandelbrot set with margin
    // Using [-2.5, 1.5] x [-2.0, 2.0] to push the sampling boundary well outside typical views
    seed = pcg_hash(seed);
    let cx = rand_float(seed) * 4.0 - 2.5;
    seed = pcg_hash(seed);
    let cy = rand_float(seed) * 4.0 - 2.0;

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
    // Require minimum 5 iterations to filter out short orbits that create uniform noise
    if iter == max_i || iter < 5u {
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
        // Inverse of the pixel -> plane mapping used by the other shaders
        // (same rotation sense, so camera pan/zoom maths applies here too)
        let rotated = vec2<f32>(
            rel.x * cos_r - rel.y * sin_r,
            rel.x * sin_r + rel.y * cos_r,
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
