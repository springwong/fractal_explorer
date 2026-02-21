use super::{Fractal, FractalParams};
use glam::DVec2;

/// Mandelbrot set fractal
pub struct Mandelbrot;

impl Fractal for Mandelbrot {
    fn shader_source(&self) -> &'static str {
        include_str!("../shaders/mandelbrot.wgsl")
    }

    fn shader_source_f64(&self) -> &'static str {
        include_str!("../shaders/mandelbrot_f64.wgsl")
    }

    fn type_id(&self) -> u32 {
        0
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
        "Mandelbrot Set"
    }
}
