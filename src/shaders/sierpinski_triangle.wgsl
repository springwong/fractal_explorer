// Sierpinski triangle with corners (0, 0), (1, 0), (1/2, sqrt(3)/2).
//
// The lattice coordinates (u, v) are the point's weights on the two edges leaving
// the corner (0, 0), so the triangle is u, v >= 0, u + v < 1. At every level the
// point is in the corner triangle its leading binary digit selects, or else in the
// middle triangle, which is the hole.
//
// One unit is 1/8 .. 1/4 pixel. Holes smaller than a pixel or two are left out:
// what survives down to there is drawn as the triangle itself.

// log2 of the smallest hole drawn, in units
const SMALLEST_HOLE_BIT: u32 = 3u;

/// 0 outside the triangle and on the set itself, else the hole's palette value
fn sierpinski_triangle(p: LatticePoint) -> f32 {
    let levels = uniforms.ref_escape_iter;
    var u = p.x;
    var v = p.y;
    if i64_is_negative(u) || i64_is_negative(v) || !i64_below_pow2(i64_add(u, v), levels) {
        return 0.0;
    }

    // Here u + v < 2^(k + 1): the point is in a triangle of that size
    for (var k = levels; k > SMALLEST_HOLE_BIT; ) {
        k -= 1u;
        if i64_bit(u, k) == 1u {
            u = i64_clear_bit(u, k);
        } else if i64_bit(v, k) == 1u {
            v = i64_clear_bit(v, k);
        } else if i64_bit(i64_add(u, v), k) == 1u {
            return hole_value(levels - 1u - k);
        }
    }
    return 0.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let value = sierpinski_triangle(lattice_point(id.xy, dims));
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(value));
}
