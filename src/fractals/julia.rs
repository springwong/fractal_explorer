use super::{Fractal, FractalParams};
use glam::{DVec2, Vec2};

/// Julia set fractal
pub struct Julia {
    pub c: Vec2,
}

impl Julia {
    pub fn new(c: Vec2) -> Self {
        Self { c }
    }

    pub fn default() -> Self {
        // Classic Julia set parameters
        Self {
            c: Vec2::new(-0.7, 0.27015),
        }
    }
}

impl Fractal for Julia {
    fn shader_source(&self) -> &'static str {
        include_str!("../shaders/julia.wgsl")
    }

    fn type_id(&self) -> u32 {
        1
    }

    fn get_params(&self) -> FractalParams {
        FractalParams {
            c_real: self.c.x,
            c_imag: self.c.y,
        }
    }

    fn default_center(&self) -> DVec2 {
        DVec2::new(0.0, 0.0)
    }

    fn default_zoom(&self) -> f64 {
        0.5
    }

    fn name(&self) -> &'static str {
        "Julia Set"
    }
}
