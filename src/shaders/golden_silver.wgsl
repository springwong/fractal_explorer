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
//
// Precision: the CPU works out, in f64, which coarse cells the view touches (at most
// two per axis) and what happens in them down to roughly the size of the view
// (see fractals/golden_silver.rs). This shader only resolves the remaining levels
// inside the view, in coordinates rescaled to O(1), so plain f32 is enough at any zoom.

// Same 80-byte layout as the other shaders; several slots carry gasket-specific data.
struct Uniforms {
    center: vec2<f32>,       // offset 0  (unused here)
    zoom: f32,               // offset 8  (unused here)
    aspect_ratio: f32,       // offset 12
    max_iter: u32,           // offset 16
    fractal_type: u32,       // offset 20
    coarse: u32,             // offset 24: coarse result per cell pair, 8 bits each
    ref_x: f32,              // offset 28: view centre relative to cell A, rescaled
    ref_y: f32,              // offset 32
    center_lo_x: f32,        // offset 36 (unused here)
    center_lo_y: f32,        // offset 40 (unused here)
    zoom_lo: f32,            // offset 44 (unused here)
    cell_b_x: f32,           // offset 48: where cell B starts, rescaled
    cell_b_y: f32,           // offset 52
    base_levels: u32,        // offset 56: levels already resolved on the CPU
    rotation: f32,           // offset 60
    extra: vec3<u32>,        // offset 64: x = bits of the rescaled per-pixel step (f32)
    _pad3: u32,              // offset 76
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> palette_lut: array<u32>;

const INV_PHI: f32 = 0.6180339887;
const LN_PHI: f32 = 0.4812118251;
// Coarse code meaning "outside the pattern"
const OUTSIDE: u32 = 255u;
// f32 comfortably resolves this many levels below the view-sized cells
const MAX_FINE_LEVELS: u32 = 30u;

/// Returns the level at which the point is removed (>= 1), or 0 for the gasket itself.
fn gasket(dx: f32, dy: f32, fine_levels: u32) -> f32 {
    // Position inside cell A; step over into cell B if past its start
    var x = abs(uniforms.ref_x + dx);
    var y = abs(uniforms.ref_y + dy);
    var cell = 0u;
    if x >= uniforms.cell_b_x { x -= uniforms.cell_b_x; cell += 2u; }
    if y >= uniforms.cell_b_y { y -= uniforms.cell_b_y; cell += 1u; }

    // What the CPU found for this pair of cells
    let code = (uniforms.coarse >> (cell * 8u)) & 255u;
    if code == OUTSIDE { return 0.0; }
    if code != 0u { return f32(code); }

    // Remaining levels, greedy golden-base digits of both coordinates
    var w = INV_PHI;
    for (var l: u32 = 1u; l <= fine_levels; l += 1u) {
        let bx = x >= w;
        let by = y >= w;
        if bx && by { return f32(uniforms.base_levels + l); }
        if bx { x -= w; }
        if by { y -= w; }
        w *= INV_PHI;
    }
    return 0.0;
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
    if id.x >= dims.x || id.y >= dims.y { return; }

    // Pixel offset from the view centre, rotated, in rescaled plane units
    let step = bitcast<f32>(uniforms.extra.x);
    let px = f32(id.x) - f32(dims.x) / 2.0;
    let py = f32(id.y) - f32(dims.y) / 2.0;
    let cos_r = cos(uniforms.rotation);
    let sin_r = sin(uniforms.rotation);
    let dx = (px * cos_r - py * sin_r) * step;
    let dy = -(px * sin_r + py * cos_r) * step;

    // Stop once cells are about one pixel wide
    let wanted = u32(clamp(round(log(1.0 / step) / LN_PHI), 1.0, f32(MAX_FINE_LEVELS)));
    let fine_levels = min(wanted, uniforms.max_iter);

    // 8 levels per palette cycle
    let smooth_val = gasket(dx, dy, fine_levels) * 2.5;

    let color = sample_palette(smooth_val);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
