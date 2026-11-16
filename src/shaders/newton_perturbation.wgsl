// Newton / Nova, deep zoom: perturbation around a reference orbit.
//
// The offset w of a pixel from the reference orbit Z follows, exactly and without
// cancellation,
//
//   w' = N(Z + w) - N(Z) = 2w/3 - w (2Z + w) / (3 Z² (Z + w)²)
//
// Every pixel starts on its own z_0 (w_0 = its offset from the view centre); c is the
// same for all pixels. If the reference orbit ends (it flew off or hit the pole at 0)
// the pixel carries on with plain f32 iteration from its current z.

/// Iterate the pixel at offset `delta` from the view centre
fn newton_perturbation(delta: vec2<f32>) -> NewtonResult {
    var w = delta;
    var iter: u32 = 0u;
    // Index of the last valid reference entry
    let ref_len = uniforms.ref_escape_iter;

    while (iter < uniforms.max_iter) {
        let z_ref = vec2<f32>(ref_orbit[iter * 2u], ref_orbit[iter * 2u + 1u]);
        let z = z_ref + w;
        if iter >= ref_len { return newton_direct(z, iter); }

        if dot(z, z) > NOVA_BAILOUT { return NewtonResult(DIVERGED, iter, z, 0.0); }
        let z2 = cmul(z, z);
        if 9.0 * dot(z2, z2) < 1.0e-20 { return NewtonResult(DIVERGED, iter, z, 0.0); }

        // Divide by Z (Z + w) twice rather than by its square, to stay clear of underflow
        let zz = cmul(z_ref, z);
        let fraction = cdiv(cdiv(cmul(w, 2.0 * z_ref + w), zz), zz) / 3.0;
        let w_new = w * (2.0 / 3.0) - fraction;

        // This pixel's step: the reference's own step plus the change in offset
        let ref_next = vec2<f32>(ref_orbit[iter * 2u + 2u], ref_orbit[iter * 2u + 3u]);
        let step = (ref_next - z_ref) + (w_new - w);
        w = w_new;
        iter += 1u;
        if dot(step, step) < TOLERANCE {
            return NewtonResult(CONVERGED, iter, ref_next + w, dot(step, step));
        }
    }
    return NewtonResult(UNDECIDED, iter, vec2<f32>(0.0, 0.0), 0.0);
}
