use super::{Fractal, FractalParams};
use glam::DVec2;

/// Burning Ship fractal
pub struct BurningShip;

impl Fractal for BurningShip {
    fn shader_source(&self) -> &'static str {
        include_str!("../shaders/burning_ship.wgsl")
    }

    fn type_id(&self) -> u32 {
        2
    }

    fn get_params(&self) -> FractalParams {
        FractalParams::default()
    }

    fn default_center(&self) -> DVec2 {
        DVec2::new(-0.5, -0.5)
    }

    fn default_zoom(&self) -> f64 {
        0.35
    }

    fn name(&self) -> &'static str {
        "Burning Ship"
    }
}
