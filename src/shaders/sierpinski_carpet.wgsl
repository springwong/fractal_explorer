// Sierpinski carpet on the unit square.
//
// A point is in a hole of some level when the base-3 digits of x and y at that level
// are both 1. The lattice coordinates are x + 1 and y + 1, so that the carpet is the
// middle cell of a 3 x 3 block and "outside" is just one more digit to look at.
//
// One unit is 1/9 .. 1/3 pixel. Holes smaller than a pixel or two are left out:
// what survives down to there is drawn as the carpet itself.

// Digit levels below the smallest hole drawn
const SKIPPED_LEVELS: u32 = 2u;

/// 0 outside the square and on the set itself, else the hole's palette value
fn sierpinski_carpet(p: LatticePoint) -> f32 {
    let levels = uniforms.ref_escape_iter;
    var x = p.x;
    var y = p.y;
    if i64_is_negative(x) || i64_is_negative(y) { return 0.0; }

    // Digits from the finest level up; the last hole found is the biggest
    var value = 0.0;
    for (var level = 0u; level < levels; level += 1u) {
        let dx = i64_div3(x);
        let dy = i64_div3(y);
        x = dx.quotient;
        y = dy.quotient;
        if dx.remainder == 1u && dy.remainder == 1u && level >= SKIPPED_LEVELS {
            value = hole_value(levels - 1u - level);
        }
    }
    // What is left is the cell of the 3 x 3 block: the carpet is the middle one
    if x.hi != 0u || y.hi != 0u || x.lo != 1u || y.lo != 1u { return 0.0; }
    return value;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let value = sierpinski_carpet(lattice_point(id.xy, dims));
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(value));
}
