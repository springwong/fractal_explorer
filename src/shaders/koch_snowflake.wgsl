// Koch snowflake grown on an equilateral triangle of side 1 centred on the origin.
//
// Each edge carries a Koch curve bulging outwards. A point is in the snowflake if it
// is in the triangle or under one of the three curves; it is coloured by the
// generation of the bump it sits in (the triangle is generation 0).

const SQRT3: f32 = 1.7320508;
// Height of a Koch curve over a base of length 1
const CURVE_HEIGHT: f32 = 0.28867513;
// Generations followed; the bumps of the last one are 3^-16 across
const GENERATIONS: u32 = 16u;

/// For a point above the unit base (0,0)-(1,0) of a Koch curve (q.y > 0): the
/// generation of the bump it is in, or 0 if it is above the curve.
fn koch_bump(start: vec2<f32>) -> u32 {
    var q = start;
    for (var generation = 1u; generation <= GENERATIONS; generation += 1u) {
        if q.y > CURVE_HEIGHT { return 0u; }
        // The curve is symmetric: look at its left half only
        q.x = min(q.x, 1.0 - q.x);
        // The left half is a flat third and a third rising at 60° from (1/3, 0).
        // Which one is nearer changes at the bisector through that corner.
        let d = q - vec2<f32>(1.0 / 3.0, 0.0);
        if d.x < -d.y / SQRT3 {
            q = 3.0 * q;
        } else {
            // Into the frame of the rising third
            q = 1.5 * vec2<f32>(d.x + SQRT3 * d.y, d.y - SQRT3 * d.x);
            if q.y <= 0.0 { return generation; }
        }
    }
    // Too close to the curve to tell: call it the edge of the finest bumps
    return GENERATIONS;
}

/// 0 outside the snowflake, else the palette value of the point's generation
fn koch_snowflake(p: vec2<f32>) -> f32 {
    var generation = 0u;
    var in_triangle = true;
    // The three edges, each seen as the base (0,0)-(1,0) with outside = up
    for (var edge = 0u; edge < 3u; edge += 1u) {
        let angle = f32(edge) * 2.0943951;
        let rotated = vec2<f32>(
            p.x * cos(angle) - p.y * sin(angle),
            p.x * sin(angle) + p.y * cos(angle),
        );
        let q = vec2<f32>(rotated.x + 0.5, -rotated.y - CURVE_HEIGHT);
        if q.y > 0.0 {
            in_triangle = false;
            generation = max(generation, koch_bump(q));
        }
    }
    if in_triangle { return 1.0; }
    if generation == 0u { return 0.0; }
    return 1.0 + 2.9 * f32(generation);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let value = koch_snowflake(pixel_to_plane(id.xy, dims));
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(value));
}
