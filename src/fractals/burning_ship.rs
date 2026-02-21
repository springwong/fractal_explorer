use super::{Fractal, FractalParams};
use glam::DVec2;

/// Burning Ship fractal
pub struct BurningShip;

impl Fractal for BurningShip {
    fn shader_source(&self) -> &'static str {
        include_str!("../shaders/burning_ship.wgsl")
    }

    fn shader_source_f64(&self) -> &'static str {
        include_str!("../shaders/burning_ship_f64.wgsl")
    }

    fn type_id(&self) -> u32 {
        2
    }

    fn get_params(&self) -> FractalParams {
        FractalParams::default()
    }

    fn default_center(&self) -> DVec2 {
        // Classic "burning ship" view at the bottom of the main body
        DVec2::new(-1.755, -0.035)
    }

    fn default_zoom(&self) -> f64 {
        150.0  // Zoomed in to show the ship detail
    }

    fn name(&self) -> &'static str {
        "Burning Ship"
    }
}
