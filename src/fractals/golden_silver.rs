use super::{Fractal, FractalParams};
use glam::DVec2;

/// Golden-Silver Gasket: the Sierpinski triangle rebuilt in the golden-ratio base.
///
/// Points whose golden-base (Zeckendorf-style) expansions of x and y never share a
/// power of phi. Dimension = ln(1 + sqrt(2)) / ln(phi) ~ 1.8316.
pub struct GoldenSilver;

impl Fractal for GoldenSilver {
    fn shader_source(&self) -> &'static str {
        include_str!("../shaders/golden_silver.wgsl")
    }

    fn shader_source_f64(&self) -> &'static str {
        // One shader for every zoom level: it is already double precision
        self.shader_source()
    }

    fn type_id(&self) -> u32 {
        6
    }

    fn get_params(&self) -> FractalParams {
        FractalParams::default()
    }

    fn default_center(&self) -> DVec2 {
        DVec2::new(0.5, 0.5)
    }

    fn default_zoom(&self) -> f64 {
        0.9
    }

    fn name(&self) -> &'static str {
        "Golden-Silver Gasket"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::naga;

    fn validate(source: &str) {
        let module = naga::front::wgsl::parse_str(source).expect("WGSL parse failed");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("WGSL validation failed");
    }

    #[test]
    fn shaders_are_valid_wgsl() {
        validate(GoldenSilver.shader_source());
    }

    /// CPU mirror of the shader's level test.
    fn removal_level(px: f64, py: f64, levels: u32) -> u32 {
        let inv_phi = 2.0 / (1.0 + 5f64.sqrt());
        let (mut x, mut y) = (px.abs(), py.abs());
        let mut w = inv_phi.powi(-10);
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
        let phi = (1.0 + 5f64.sqrt()) / 2.0;
        let expected = [3usize, 7, 17, 41, 99, 239, 577];
        for (i, &want) in expected.iter().enumerate() {
            let n = i as i32 + 1;
            // Enumerate every admissible digit string of length n and probe just inside its cell.
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
}
