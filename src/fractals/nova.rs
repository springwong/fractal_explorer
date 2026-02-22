use super::{Fractal, FractalParams};
use glam::{DVec2, Vec2};

/// Nova fractal: Newton's method with perturbation
/// z → z - (z³-1)/(3z²) + c
pub struct Nova {
    pub c: Vec2,
}

impl Nova {
    pub fn new(c: Vec2) -> Self {
        Self { c }
    }

    pub fn default() -> Self {
        Self {
            c: Vec2::new(1.0, 0.0),
        }
    }
}

impl Fractal for Nova {
    fn shader_source(&self) -> &'static str {
        include_str!("../shaders/nova.wgsl")
    }

    fn shader_source_f64(&self) -> &'static str {
        include_str!("../shaders/nova_f64.wgsl")
    }

    fn type_id(&self) -> u32 {
        5
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
        "Nova Fractal"
    }
}
