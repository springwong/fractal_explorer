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

// Golden-Silver Gasket: the Sierpinski triangle rebuilt in the golden-ratio base.
//
// Write |x| and |y| greedily as sums of distinct powers of phi with no two
// consecutive powers (Zeckendorf-style). A point belongs to the gasket iff x and y
// never use the same power of phi. Level counts grow by the silver ratio 1+sqrt(2)
// while cells shrink by phi, so the dimension is ln(1+sqrt(2)) / ln(phi) ~ 1.8316.
//
// Rendering: a pixel falls into a hole the first time both expansions use the same
// power; it is coloured by that level. Pixels that survive down to pixel size are
// the gasket itself (black, like the inside of the other sets).

const INV_PHI: f32 = 0.6180339887;
const LN_PHI: f32 = 0.4812118251;
// Expansion starts at phi^TOP_POW, so the pattern covers |x|, |y| < phi^(TOP_POW + 1).
const TOP_POW: i32 = 10;
const W_TOP: f32 = 122.99186938;   // phi^10
const EXTENT: f32 = 199.0;         // just under phi^11
// f32 runs out of bits beyond roughly phi^-31.
const MAX_FRAC_LEVELS: i32 = 31;

/// Returns the level at which the point is removed (>= 1), or 0 if it survives `levels` levels.
fn gasket(px: f32, py: f32, levels: u32) -> f32 {
    var x = abs(px);
    var y = abs(py);
    if max(x, y) >= EXTENT { return 0.0; }

    var w = W_TOP;
    for (var l: u32 = 1u; l <= levels; l += 1u) {
        let dx = x >= w;
        let dy = y >= w;
        if dx && dy { return f32(l); }
        if dx { x -= w; }
        if dy { y -= w; }
        w *= INV_PHI;
    }
    return 0.0;
}

/// Number of levels to resolve: stop once cells are about one pixel wide.
fn level_count(dims_y: u32) -> u32 {
    let frac = i32(round(log(max(uniforms.zoom * f32(dims_y), 1e-6)) / LN_PHI));
    let total = clamp(TOP_POW + 1 + min(frac, MAX_FRAC_LEVELS), 1, 96);
    return min(u32(total), uniforms.max_iter);
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

    // Convert pixel coordinates to plane coordinates
    let uv = (vec2<f32>(id.xy) - vec2<f32>(dims) / 2.0) / f32(dims.y);
    let cos_r = cos(uniforms.rotation);
    let sin_r = sin(uniforms.rotation);
    let rotated = vec2<f32>(uv.x * cos_r - uv.y * sin_r, uv.x * sin_r + uv.y * cos_r);
    let c = uniforms.center + (rotated / uniforms.zoom) * vec2<f32>(1.0, -1.0);

    // Hole level (0 = on the gasket); 8 levels per palette cycle
    let smooth_val = gasket(c.x, c.y, level_count(dims.y)) * 2.5;

    // Colorize and write to output texture
    let color = sample_palette(smooth_val);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
