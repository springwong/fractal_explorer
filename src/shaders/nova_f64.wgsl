// Nova fractal, deep zoom: perturbation around a reference orbit.
//
// The Nova step z -> z - (z³-1)/(3z²) + c simplifies to N(z) = 2z/3 + 1/(3z²) + c,
// so the offset w of a pixel from the reference orbit Z follows, exactly and without
// cancellation,
//
//   w' = N(Z + w) - N(Z) = 2w/3 - w (2Z + w) / (3 Z² (Z + w)²)
//
// Every pixel starts on its own z_0 (w_0 = its offset from the view centre); c is the
// same for all pixels. If the reference orbit ends (it flew off or hit the pole at 0)
// the pixel carries on with plain f32 iteration from its current z.

const TOLERANCE: f32 = 1.0e-6;
const NOVA_BAILOUT: f32 = 1.0e8;

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b);
}

/// Smooth value once a step of squared length `delta2` has converged
fn converged_value(iter: u32, delta2: f32) -> f32 {
    let smooth_val = f32(iter) - log2(log2(delta2 + 1.0e-20) / log2(TOLERANCE));
    return max(smooth_val, 0.0);
}

/// Plain f32 iteration from `z_start` at iteration `start_iter`
fn nova_direct(z_start: vec2<f32>, start_iter: u32) -> f32 {
    let c = vec2<f32>(uniforms.c_real, uniforms.c_imag);
    var z = z_start;
    var iter = start_iter;

    while (iter < uniforms.max_iter) {
        if dot(z, z) > NOVA_BAILOUT { break; }
        let z2 = cmul(z, z);
        if 9.0 * dot(z2, z2) < 1.0e-20 { break; }
        let z_new = z * (2.0 / 3.0) + cdiv(vec2<f32>(1.0 / 3.0, 0.0), z2) + c;
        let step = z_new - z;
        z = z_new;
        iter += 1u;
        if dot(step, step) < TOLERANCE { return converged_value(iter, dot(step, step)); }
    }

    if iter == uniforms.max_iter { return 0.0; }
    return f32(iter);
}

fn nova_perturbation(delta: vec2<f32>) -> f32 {
    var w = delta;
    var iter: u32 = 0u;
    // Index of the last valid reference entry
    let ref_len = uniforms.ref_escape_iter;

    while (iter < uniforms.max_iter) {
        let z_ref = vec2<f32>(ref_orbit[iter * 2u], ref_orbit[iter * 2u + 1u]);
        let z = z_ref + w;
        if iter >= ref_len { return nova_direct(z, iter); }

        if dot(z, z) > NOVA_BAILOUT { break; }
        let z2 = cmul(z, z);
        if 9.0 * dot(z2, z2) < 1.0e-20 { break; }

        // Divide by Z (Z + w) twice rather than by its square, to stay clear of underflow
        let zz = cmul(z_ref, z);
        let fraction = cdiv(cdiv(cmul(w, 2.0 * z_ref + w), zz), zz) / 3.0;
        let w_new = w * (2.0 / 3.0) - fraction;

        // This pixel's step: the reference's own step plus the change in offset
        let ref_next = vec2<f32>(ref_orbit[iter * 2u + 2u], ref_orbit[iter * 2u + 3u]);
        let step = (ref_next - z_ref) + (w_new - w);
        w = w_new;
        iter += 1u;
        if dot(step, step) < TOLERANCE { return converged_value(iter, dot(step, step)); }
    }

    if iter == uniforms.max_iter { return 0.0; }
    return f32(iter);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let smooth_val = nova_perturbation(pixel_delta(id.xy, dims));
    textureStore(output_texture, vec2<i32>(id.xy), sample_palette(smooth_val));
}
