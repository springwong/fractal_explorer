// Apollonian gasket: the unit circle, three equal circles packed inside it, and
// every gap filled with the largest circle that fits, for ever.
//
// Four "mirror" circles pass through the points where the four starting circles
// touch. A point in a gap between the starting discs is inverted in the mirror circle
// covering that gap, which moves it to a bigger disc of the packing; repeating that
// ends in one of the four starting discs.
// Carrying the composed map along tells how big the disc the point started in is,
// and the disc is coloured by its size.

// Radius of the central mirror circle: 2 - sqrt(3)
const CENTRAL_MIRROR: f32 = 0.26794919;
// Radius of each of the three starting circles: 2 sqrt(3) - 3
const INNER_RADIUS: f32 = 0.46410162;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn conj(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

/// Direction of starting circle k = 1, 2, 3 (90°, 210°, 330°)
fn direction(k: u32) -> vec2<f32> {
    let angle = 1.5707963 + f32(k - 1u) * 2.0943951;
    return vec2<f32>(cos(angle), sin(angle));
}

/// 0 outside the unit circle and on the gasket itself, else the palette value of the
/// disc the point is in
fn apollonian(start: vec2<f32>) -> f32 {
    if dot(start, start) >= 1.0 { return 0.0; }
    var p = start;
    // The map from the current point back to the start: z -> (a z + b) / (c z + d),
    // applied to conj(z) instead of z when `flipped`
    var a = vec2<f32>(1.0, 0.0);
    var b = vec2<f32>(0.0, 0.0);
    var c = vec2<f32>(0.0, 0.0);
    var d = vec2<f32>(1.0, 0.0);
    var flipped = false;

    for (var iter = 0u; iter < uniforms.max_iter; iter += 1u) {
        // In one of the four starting discs? (The unit circle's disc is its outside.)
        var centre = vec2<f32>(0.0, 0.0);
        var radius = 1.0;
        var arrived = dot(p, p) >= 1.0;
        for (var k = 1u; k <= 3u && !arrived; k += 1u) {
            centre = (1.0 - INNER_RADIUS) * direction(k);
            radius = INNER_RADIUS;
            arrived = dot(p - centre, p - centre) <= radius * radius;
        }
        if arrived {
            // Map that disc back to where the point started
            if flipped { centre = conj(centre); }
            // Radius of the image of a circle under a Möbius map
            let pole = cmul(c, centre) + d;
            let det = cmul(a, d) - cmul(b, c);
            let size = radius * length(det) / abs(dot(pole, pole) - radius * radius * dot(c, c));
            return 1.0 + 4.0 * log2(1.0 / size);
        }

        // Otherwise p is in a gap between them, and each gap lies inside one mirror
        // circle: centre m, radius² r2
        var m = vec2<f32>(0.0, 0.0);
        var r2 = CENTRAL_MIRROR * CENTRAL_MIRROR;
        var found = dot(p, p) < r2;
        for (var k = 1u; k <= 3u && !found; k += 1u) {
            m = -2.0 * direction(k);
            r2 = 3.0;
            found = dot(p - m, p - m) < r2;
        }
        if !found { return 0.0; }

        // Invert p in the mirror circle
        let offset = p - m;
        // (the very centre of a mirror circle goes off to a far point rather than to infinity)
        p = m + offset * (r2 / max(dot(offset, offset), 1.0e-30));

        // ... and append the inversion, conj(z) -> (m conj(z) + r2 - |m|²) / (conj(z) - conj(m)),
        // to the map back
        var sa = m;
        var sb = vec2<f32>(r2 - dot(m, m), 0.0);
        var sc = vec2<f32>(1.0, 0.0);
        var sd = -conj(m);
        if flipped {
            sa = conj(sa);
            sd = conj(sd);
        }
        let na = cmul(a, sa) + cmul(b, sc);
        let nb = cmul(a, sb) + cmul(b, sd);
        let nc = cmul(c, sa) + cmul(d, sc);
        let nd = cmul(c, sb) + cmul(d, sd);
        // Only ratios matter: keep the entries near 1
        let norm = 1.0 / max(max(length(na), length(nb)), max(length(nc), length(nd)));
        a = na * norm;
        b = nb * norm;
        c = nc * norm;
        d = nd * norm;
        flipped = !flipped;
    }
    return 0.0;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let value = apollonian(pixel_to_plane(id.xy, dims));
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(value));
}
