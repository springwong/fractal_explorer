use super::{Fractal, FractalParams};
use glam::DVec2;

/// Buddhabrot fractal - density map of escaping Mandelbrot orbits
pub struct Buddhabrot;

impl Fractal for Buddhabrot {
    fn shader_source(&self) -> &'static str {
        include_str!("../shaders/buddhabrot.wgsl")
    }

    fn shader_source_f64(&self) -> &'static str {
        // No deep zoom needed for Buddhabrot
        include_str!("../shaders/buddhabrot.wgsl")
    }

    fn type_id(&self) -> u32 {
        4
    }

    fn get_params(&self) -> FractalParams {
        FractalParams::default()
    }

    fn default_center(&self) -> DVec2 {
        DVec2::new(-0.5, 0.0)
    }

    fn default_zoom(&self) -> f64 {
        0.5
    }

    fn name(&self) -> &'static str {
        "Buddhabrot"
    }
}
