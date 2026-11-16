//! Golden-Silver Gasket: the Sierpinski triangle rebuilt in the golden-ratio base.
//!
//! Points whose golden-base (Zeckendorf-style) expansions of x and y never share a
//! power of phi. Dimension = ln(1 + sqrt(2)) / ln(phi) ~ 1.8316.
//!
//! One shader serves every zoom level: the CPU handles the deep part (see `apply_view`).

use crate::renderer::FractalUniforms;

/// Highest power of phi used: the pattern covers |x|, |y| < phi^(TOP_POW + 1) ~ 199.
const TOP_POW: i32 = 10;
/// Coarse code for "outside the pattern" (must match the shader).
const OUTSIDE: u32 = 255;

/// One coarse cell along an axis: the golden-base digits of its start, from phi^TOP_POW down.
struct Cell {
    digits: Vec<bool>,
    outside: bool,
}

/// What the view touches along one axis.
struct Axis {
    /// View centre relative to the start of cell A, rescaled (signed: |ref + d| is the position)
    reference: f64,
    /// Start of cell B relative to cell A, rescaled
    cell_b: f64,
    cells: [Cell; 2],
}

fn phi() -> f64 {
    (1.0 + 5f64.sqrt()) / 2.0
}

/// Greedy golden-base digits of `v` over the powers TOP_POW..=low_pow. Returns (digits, remainder).
fn digits(mut v: f64, low_pow: i32) -> (Vec<bool>, f64) {
    let phi = phi();
    let mut out = Vec::new();
    for e in (low_pow..=TOP_POW).rev() {
        let w = phi.powi(e);
        let one = v >= w;
        if one {
            v -= w;
        }
        out.push(one);
    }
    (out, v)
}

/// Work out the (at most two) cells at level `low_pow` that |coordinate| can fall in
/// across a view centred at `center` with half-extent `ext`.
fn axis(center: f64, ext: f64, low_pow: i32) -> Axis {
    let phi = phi();
    let scale = phi.powi(-low_pow);
    let extent = phi.powi(TOP_POW + 1);

    // Lowest |coordinate| in the view (0 if the view straddles the mirror axis)
    let lo = (center.abs() - ext).max(0.0);
    if lo >= extent {
        let gone = || Cell { digits: Vec::new(), outside: true };
        return Axis { reference: 0.0, cell_b: 1e30, cells: [gone(), gone()] };
    }

    let (digits_a, rem) = digits(lo, low_pow);
    let start_a = lo - rem;
    // A cell is phi^low_pow wide, or phi^(low_pow - 1) if its last digit is 1.
    // With no digits at all (low_pow above TOP_POW) the single cell is the whole pattern.
    let size_a = match digits_a.last() {
        None => extent,
        Some(true) => phi.powi(low_pow - 1),
        Some(false) => phi.powi(low_pow),
    };
    let start_b = start_a + size_a;
    let outside_b = start_b >= extent * (1.0 - 1e-12);
    // Probe safely inside cell B so rounding at the boundary cannot matter
    let (digits_b, _) = digits(start_b + 0.25 * phi.powi(low_pow - 1), low_pow);

    Axis {
        reference: center.signum() * (center.abs() - start_a) * scale,
        cell_b: size_a * scale,
        cells: [
            Cell { digits: digits_a, outside: false },
            Cell { digits: digits_b, outside: outside_b },
        ],
    }
}

/// Fill the gasket-specific uniform slots for the view described by the uniforms'
/// own centre / zoom / rotation and the given output size.
///
/// The view is located in f64 here; the shader then only needs f32 for the levels
/// inside the view, so deep zoom does not depend on GPU double emulation.
pub fn apply_view(u: &mut FractalUniforms, width: u32, height: u32) {
    let phi = phi();
    let cx = u.center[0] as f64 + u.center_lo_x as f64;
    let cy = u.center[1] as f64 + u.center_lo_y as f64;
    let zoom = u.zoom as f64 + u.zoom_lo as f64;
    let (sin_r, cos_r) = (u.rotation as f64).sin_cos();

    // Half-extent of the (rotated) view along each axis, with a little slack
    let half_h = 0.5 / zoom;
    let half_w = half_h * width as f64 / height as f64;
    let ext_x = (half_w * cos_r.abs() + half_h * sin_r.abs()) * 1.05;
    let ext_y = (half_w * sin_r.abs() + half_h * cos_r.abs()) * 1.05;

    // Coarsest level whose smallest cells (phi^(low_pow - 1)) are still wider than the
    // view, so the view meets at most two cells per axis.
    let span = 2.0 * ext_x.max(ext_y);
    let low_pow = ((span.ln() / phi.ln()).ceil() as i32 + 1).clamp(-70, TOP_POW + 1);

    let ax = axis(cx, ext_x, low_pow);
    let ay = axis(cy, ext_y, low_pow);

    // For each pair of cells: outside, removed at a coarse level, or still alive
    let mut coarse = 0u32;
    for (i, cell_x) in ax.cells.iter().enumerate() {
        for (j, cell_y) in ay.cells.iter().enumerate() {
            let code = if cell_x.outside || cell_y.outside {
                OUTSIDE
            } else {
                cell_x.digits.iter().zip(&cell_y.digits)
                    .position(|(a, b)| *a && *b)
                    .map_or(0, |k| k as u32 + 1)
            };
            coarse |= code << ((i * 2 + j) * 8);
        }
    }

    let scale = phi.powi(-low_pow);
    u.color_scheme = coarse;
    u.c_real = ax.reference as f32;
    u.c_imag = ay.reference as f32;
    u.pixel_step_x = ax.cell_b as f32;
    u.pixel_step_y = ay.cell_b as f32;
    u.ref_escape_iter = (TOP_POW - low_pow + 1).max(0) as u32;
    u.extra[0] = ((scale / (zoom * height as f64)) as f32).to_bits();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fractals::FractalType;
    use crate::renderer::View;
    use glam::DVec2;

    /// Straightforward f64 definition: level at which (px, py) is removed, 0 if it survives.
    fn removal_level(px: f64, py: f64, levels: u32) -> u32 {
        let inv_phi = 1.0 / phi();
        let (mut x, mut y) = (px.abs(), py.abs());
        let mut w = phi().powi(TOP_POW);
        if x.max(y) >= w * phi() {
            return 0;
        }
        for l in 1..=levels {
            let (dx, dy) = (x >= w, y >= w);
            if dx && dy {
                return l;
            }
            if dx {
                x -= w;
            }
            if dy {
                y -= w;
            }
            w *= inv_phi;
        }
        0
    }

    /// Cells of the unit square surviving n levels follow 3, 7, 17, 41, 99, ...
    /// (numerators of the continued-fraction convergents of sqrt(2)).
    #[test]
    fn surviving_cells_follow_sqrt2_convergents() {
        let phi = phi();
        let expected = [3usize, 7, 17, 41, 99, 239, 577];
        for (i, &want) in expected.iter().enumerate() {
            let n = i as i32 + 1;
            let mut coords = vec![(0.0f64, false)];
            for k in 1..=n {
                let w = phi.powi(-k);
                let mut next = Vec::new();
                for &(v, last_one) in &coords {
                    next.push((v, false));
                    if !last_one {
                        next.push((v + w, true));
                    }
                }
                coords = next;
            }
            let eps = phi.powi(-n) * 1e-3;
            let mut alive = 0;
            for &(x, _) in &coords {
                for &(y, _) in &coords {
                    if removal_level(x + eps, y + eps, 11 + n as u32) == 0 {
                        alive += 1;
                    }
                }
            }
            assert_eq!(alive, want, "level {n}");
        }
    }

    /// The shader's per-pixel logic, in f32 exactly as the WGSL does it.
    /// Returns (level, total levels resolved).
    fn shade(u: &FractalUniforms, px: f32, py: f32) -> (u32, u32) {
        const INV_PHI: f32 = 0.6180339887;
        const LN_PHI: f32 = 0.4812118251;
        let step = f32::from_bits(u.extra[0]);
        let (sin_r, cos_r) = (u.rotation.sin(), u.rotation.cos());
        let dx = (px * cos_r - py * sin_r) * step;
        let dy = -(px * sin_r + py * cos_r) * step;
        let fine = ((1.0 / step).ln() / LN_PHI).round().clamp(1.0, 30.0) as u32;
        let total = u.ref_escape_iter + fine;

        let mut x = (u.c_real + dx).abs();
        let mut y = (u.c_imag + dy).abs();
        let mut cell = 0;
        if x >= u.pixel_step_x {
            x -= u.pixel_step_x;
            cell += 2;
        }
        if y >= u.pixel_step_y {
            y -= u.pixel_step_y;
            cell += 1;
        }
        let code = (u.color_scheme >> (cell * 8)) & 255;
        if code == OUTSIDE {
            return (0, total);
        }
        if code != 0 {
            return (code, total);
        }
        let mut w = INV_PHI;
        for l in 1..=fine {
            let (bx, by) = (x >= w, y >= w);
            if bx && by {
                return (u.ref_escape_iter + l, total);
            }
            if bx {
                x -= w;
            }
            if by {
                y -= w;
            }
            w *= INV_PHI;
        }
        (0, total)
    }

    /// The split CPU(f64) + shader(f32) evaluation must agree with the plain f64
    /// definition at every zoom, anywhere on the pattern, at any rotation.
    #[test]
    fn deep_zoom_matches_reference() {
        let (width, height) = (640u32, 360u32);
        // Off-lattice centres: integers are finite golden-base numbers, i.e. exact cell
        // edges, and pixels landing exactly on an edge may round either way.
        let centers = [
            (0.5137, 0.4871), (0.809016994, 0.5003), (0.0013, -0.0007), (-0.3011, 0.7003),
            (0.6183, 0.2503), (37.2013, -150.9007), (198.9003, 198.9001),
            (-120.0701, 3.3101), (2.6173, 1.0041), (-0.0004, 61.3307),
        ];
        let zooms = [0.0022, 0.05, 0.9, 30.0, 6.93e3, 2.5e5, 1e7, 1e9];
        let rotations = [0.0, 0.6, -2.4];
        // Deterministic pseudo-random pixel picks
        let mut seed = 12345u64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        for &(cx, cy) in &centers {
            for &zoom in &zooms {
                for &rotation in &rotations {
                    let view = View { center: DVec2::new(cx, cy), zoom, rotation, max_iter: 256 };
                    // Building the uniforms for the gasket runs `apply_view`
                    let u = FractalUniforms::for_view(&FractalType::GoldenSilver, &view, width, height);
                    // The uniforms carry the view in f32 pairs; use exactly what they hold
                    let (cx, cy) = (u.center[0] as f64 + u.center_lo_x as f64, u.center[1] as f64 + u.center_lo_y as f64);
                    let (zoom, rot) = (u.zoom as f64 + u.zoom_lo as f64, u.rotation as f64);

                    let (mut bad, samples) = (0, 2000);
                    for _ in 0..samples {
                        let px = (rand() * width as f64).floor() - width as f64 / 2.0;
                        let py = (rand() * height as f64).floor() - height as f64 / 2.0;
                        let (got, total) = shade(&u, px as f32, py as f32);
                        let (ux, uy) = (px / height as f64, py / height as f64);
                        let (rx, ry) = (ux * rot.cos() - uy * rot.sin(), ux * rot.sin() + uy * rot.cos());
                        let want = removal_level(cx + rx / zoom, cy - ry / zoom, total);
                        if got != want {
                            bad += 1;
                        }
                    }
                    // Pixels sitting exactly on a cell edge may round either way
                    assert!(
                        bad * 100 <= samples,
                        "centre ({cx}, {cy}) zoom {zoom} rot {rotation}: {bad}/{samples} pixels differ"
                    );
                }
            }
        }
    }
}
