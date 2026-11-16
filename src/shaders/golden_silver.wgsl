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

// Uniform slots this shader reads (see common.wgsl), filled by `apply_view` on the CPU:
//   color_scheme      coarse result per cell pair, 8 bits each
//   c_real, c_imag    view centre relative to cell A, rescaled
//   pixel_step_x/y    where cell B starts, rescaled
//   ref_escape_iter   levels already resolved on the CPU
//   extra.x           bits of the rescaled per-pixel step (f32)

const INV_PHI: f32 = 0.6180339887;
const LN_PHI: f32 = 0.4812118251;
// Coarse code meaning "outside the pattern"
const OUTSIDE: u32 = 255u;
// f32 comfortably resolves this many levels below the view-sized cells
const MAX_FINE_LEVELS: u32 = 30u;

/// Returns the level at which the point is removed (>= 1), or 0 for the gasket itself.
fn gasket(dx: f32, dy: f32, fine_levels: u32) -> f32 {
    // Position inside cell A; step over into cell B if past its start
    var x = abs(uniforms.c_real + dx);
    var y = abs(uniforms.c_imag + dy);
    var cell = 0u;
    if x >= uniforms.pixel_step_x { x -= uniforms.pixel_step_x; cell += 2u; }
    if y >= uniforms.pixel_step_y { y -= uniforms.pixel_step_y; cell += 1u; }

    // What the CPU found for this pair of cells
    let code = (uniforms.color_scheme >> (cell * 8u)) & 255u;
    if code == OUTSIDE { return 0.0; }
    if code != 0u { return f32(code); }

    // Remaining levels, greedy golden-base digits of both coordinates
    var w = INV_PHI;
    for (var l: u32 = 1u; l <= fine_levels; l += 1u) {
        let bx = x >= w;
        let by = y >= w;
        if bx && by { return f32(uniforms.ref_escape_iter + l); }
        if bx { x -= w; }
        if by { y -= w; }
        w *= INV_PHI;
    }
    return 0.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    // Pixel offset from the view centre, rotated, in rescaled plane units
    let step = bitcast<f32>(uniforms.extra.x);
    let d = rotated_pixel(id.xy, dims) * vec2<f32>(step, -step);
    // Stop once cells are about one pixel wide
    let wanted = u32(clamp(round(log(1.0 / step) / LN_PHI), 1.0, f32(MAX_FINE_LEVELS)));
    let fine_levels = min(wanted, uniforms.max_iter);

    // 8 levels per palette cycle
    let smooth_val = gasket(d.x, d.y, fine_levels) * 2.5;

    let color = sample_palette(smooth_val);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
