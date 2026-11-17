// Newton's method for z³ - 1 with an offset c added to every step:
//
//   N(z) = z - (z³ - 1) / (3z²) + c = 2z/3 + 1/(3z²) + c
//
// c = 0 is the plain Newton fractal; any other c is a Nova fractal. This module holds
// the f32 iteration and the two ways of colouring its outcome.

const TOLERANCE: f32 = 1.0e-6;
const NOVA_BAILOUT: f32 = 1.0e8;

// How an iteration ended
const UNDECIDED: u32 = 0u; // still moving after max_iter steps
const CONVERGED: u32 = 1u; // the last step was shorter than the tolerance
const DIVERGED: u32 = 2u;  // flew off, or hit the pole at 0

struct NewtonResult {
    outcome: u32,
    /// Steps taken
    iter: u32,
    /// Where the iteration ended
    z: vec2<f32>,
    /// Squared length of the last step (when converged)
    delta2: f32,
}

fn cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cdiv(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b);
}

/// Plain f32 iteration from `z_start`, counting on from `start_iter`
fn newton_direct(z_start: vec2<f32>, start_iter: u32) -> NewtonResult {
    let c = vec2<f32>(uniforms.c_real, uniforms.c_imag);
    var z = z_start;
    var iter = start_iter;

    while (iter < uniforms.max_iter) {
        if dot(z, z) > NOVA_BAILOUT { return NewtonResult(DIVERGED, iter, z, 0.0); }
        let z2 = cmul(z, z);
        if 9.0 * dot(z2, z2) < 1.0e-20 { return NewtonResult(DIVERGED, iter, z, 0.0); }
        let z_new = z * (2.0 / 3.0) + cdiv(vec2<f32>(1.0 / 3.0, 0.0), z2) + c;
        let step = z_new - z;
        z = z_new;
        iter += 1u;
        if dot(step, step) < TOLERANCE { return NewtonResult(CONVERGED, iter, z, dot(step, step)); }
    }
    return NewtonResult(UNDECIDED, iter, z, 0.0);
}

/// Smooth step count of a converged iteration
fn converged_value(r: NewtonResult) -> f32 {
    let smooth_val = f32(r.iter) - log2(log2(r.delta2 + 1.0e-20) / log2(TOLERANCE));
    return max(smooth_val, 0.0);
}

/// Nova colouring: how long the point took to settle (0 = never did)
fn nova_value(r: NewtonResult) -> f32 {
    if r.outcome == CONVERGED { return converged_value(r); }
    if r.outcome == UNDECIDED { return 0.0; }
    return f32(r.iter);
}

/// Newton colouring: each of the three roots of z³ = 1 owns a third of the palette,
/// shaded by how long the point took to reach it (0 = reached none).
fn newton_value(r: NewtonResult) -> f32 {
    if r.outcome != CONVERGED { return 0.0; }
    // The roots sit at angles 0 and ±120°
    let third = 2.0943951;
    let root = (i32(round(atan2(r.z.y, r.z.x) / third)) + 3) % 3;
    // The palette repeats every 20 units (see sample_palette)
    return 0.5 + f32(root) * (20.0 / 3.0) + min(converged_value(r) * 0.15, 5.5);
}
