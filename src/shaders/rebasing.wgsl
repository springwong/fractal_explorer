// Perturbation with rebasing, for maps iterated from z_0 = 0 with c = the pixel
// (Mandelbrot, Burning Ship, Tricorn).
//
// The CPU computes one reference orbit Z_m in f64 for c = view centre. A pixel only
// tracks its offset w from that orbit, which stays accurate in f32 at any zoom.
// The fractal body supplies the map for the offset:
//
//   fn delta_step(z_ref: vec2<f32>, w: vec2<f32>, dc: vec2<f32>) -> vec2<f32>
//
// returning f(Z + w, C + dc) - f(Z, C) in a form free of cancellation.
//
// Rebasing: whenever the pixel's orbit comes closer to the origin than its offset is
// large (|z| < |w|), or the reference orbit has run out, the pixel restarts on the
// reference orbit from Z_0 = 0 with w = z. That is exact, and it removes the glitches
// plain perturbation shows, without ever falling back to f32 coordinates.

// Bailout radius squared (large, for smooth colouring)
const BAILOUT2: f32 = 256.0;

/// Smooth escape time of the pixel at offset `dc` from the view centre (0 = inside).
fn perturbed_escape_time(dc: vec2<f32>) -> f32 {
    var w = vec2<f32>(0.0, 0.0);
    var m: u32 = 0u;
    // Index of the last valid reference entry
    let ref_len = uniforms.ref_escape_iter;

    for (var iter: u32 = 0u; iter < uniforms.max_iter; iter += 1u) {
        var z_ref = vec2<f32>(ref_orbit[m * 2u], ref_orbit[m * 2u + 1u]);
        let z = z_ref + w;
        let mag2 = dot(z, z);

        if mag2 > BAILOUT2 {
            let nu = log2(log2(mag2) / 2.0);
            return f32(iter) + 1.0 - nu;
        }

        if mag2 < dot(w, w) || m >= ref_len {
            w = z;
            m = 0u;
            z_ref = vec2<f32>(0.0, 0.0);
        }

        w = delta_step(z_ref, w, dc);
        m += 1u;
    }

    return 0.0; // Inside the set
}
