/// Fractal trait and type system
mod mandelbrot;
mod julia;
mod burning_ship;
mod tricorn;
mod buddhabrot;
mod nova;
mod golden_silver;

pub use mandelbrot::Mandelbrot;
pub use julia::Julia;
pub use burning_ship::BurningShip;
pub use tricorn::Tricorn;
pub use buddhabrot::Buddhabrot;
pub use nova::Nova;
pub use golden_silver::GoldenSilver;

use crate::camera::ViewBounds;
use glam::{DVec2, Vec2};

/// Trait for fractal types
pub trait Fractal: Send + Sync {
    /// Get the WGSL shader source code for this fractal
    fn shader_source(&self) -> &'static str;

    /// Get the WGSL shader source with emulated f64 precision for deep zoom
    fn shader_source_f64(&self) -> &'static str;

    /// Get the fractal type ID for uniform buffer
    fn type_id(&self) -> u32;

    /// Get fractal-specific parameters (e.g., Julia's c value)
    fn get_params(&self) -> FractalParams;

    /// Get the default center point for this fractal
    fn default_center(&self) -> DVec2;

    /// Get the default zoom level for this fractal
    fn default_zoom(&self) -> f64;

    /// Get the human-readable name of this fractal
    fn name(&self) -> &'static str;
}

/// Fractal-specific parameters
#[derive(Clone, Copy, Debug)]
pub struct FractalParams {
    pub c_real: f32,
    pub c_imag: f32,
}

impl Default for FractalParams {
    fn default() -> Self {
        Self {
            c_real: 0.0,
            c_imag: 0.0,
        }
    }
}

/// Enum for all supported fractal types
#[derive(Clone, PartialEq, Debug)]
pub enum FractalType {
    Mandelbrot,
    Julia { c: Vec2 },
    BurningShip,
    Tricorn,
    Buddhabrot,
    Nova { c: Vec2 },
    GoldenSilver,
}

impl FractalType {
    /// Get the type ID for uniform buffer
    pub fn type_id(&self) -> u32 {
        match self {
            FractalType::Mandelbrot => 0,
            FractalType::Julia { .. } => 1,
            FractalType::BurningShip => 2,
            FractalType::Tricorn => 3,
            FractalType::Buddhabrot => 4,
            FractalType::Nova { .. } => 5,
            FractalType::GoldenSilver => 6,
        }
    }

    /// Get fractal-specific parameters
    pub fn params(&self) -> FractalParams {
        match self {
            FractalType::Mandelbrot => FractalParams::default(),
            FractalType::Julia { c } => FractalParams {
                c_real: c.x,
                c_imag: c.y,
            },
            FractalType::BurningShip => FractalParams::default(),
            FractalType::Tricorn => FractalParams::default(),
            FractalType::Buddhabrot => FractalParams::default(),
            FractalType::Nova { c } => FractalParams {
                c_real: c.x,
                c_imag: c.y,
            },
            FractalType::GoldenSilver => FractalParams::default(),
        }
    }

    /// Get the default center for this fractal
    pub fn default_center(&self) -> DVec2 {
        match self {
            FractalType::Mandelbrot => Mandelbrot.default_center(),
            FractalType::Julia { .. } => Julia::default().default_center(),
            FractalType::BurningShip => BurningShip.default_center(),
            FractalType::Tricorn => Tricorn.default_center(),
            FractalType::Buddhabrot => Buddhabrot.default_center(),
            FractalType::Nova { .. } => Nova::default().default_center(),
            FractalType::GoldenSilver => GoldenSilver.default_center(),
        }
    }

    /// Get the default zoom for this fractal
    pub fn default_zoom(&self) -> f64 {
        match self {
            FractalType::Mandelbrot => Mandelbrot.default_zoom(),
            FractalType::Julia { .. } => Julia::default().default_zoom(),
            FractalType::BurningShip => BurningShip.default_zoom(),
            FractalType::Tricorn => Tricorn.default_zoom(),
            FractalType::Buddhabrot => Buddhabrot.default_zoom(),
            FractalType::Nova { .. } => Nova::default().default_zoom(),
            FractalType::GoldenSilver => GoldenSilver.default_zoom(),
        }
    }

    /// Where the camera may go for this fractal: the centre stays inside a box
    /// around the interesting region and zoom stays within what can be shown.
    pub fn view_bounds(&self) -> ViewBounds {
        // Deepest useful zoom: f64 camera with perturbation, emulated double, or plain f32
        const PERTURBATION: f64 = 1e13;
        const EMULATED_DOUBLE: f64 = 1e10;
        match self {
            FractalType::Mandelbrot => ViewBounds::new(-2.5, -1.5, 1.0, 1.5, 0.2, PERTURBATION),
            FractalType::Julia { .. } => ViewBounds::new(-2.0, -2.0, 2.0, 2.0, 0.2, PERTURBATION),
            FractalType::BurningShip => ViewBounds::new(-2.5, -2.0, 1.5, 1.0, 0.2, EMULATED_DOUBLE),
            FractalType::Tricorn => ViewBounds::new(-2.5, -2.0, 1.5, 2.0, 0.2, EMULATED_DOUBLE),
            // Samples thin out quadratically with zoom, so deep views stay empty
            FractalType::Buddhabrot => ViewBounds::new(-2.5, -1.5, 1.0, 1.5, 0.2, 200.0),
            FractalType::Nova { .. } => ViewBounds::new(-3.0, -3.0, 3.0, 3.0, 0.2, EMULATED_DOUBLE),
            // The pattern is drawn out to |x|, |y| < 199 and is black beyond. Let the view
            // roam all of it, with at most a thin black strip showing past the edge.
            FractalType::GoldenSilver => {
                ViewBounds::new(-199.0, -199.0, 199.0, 199.0, 0.0022, EMULATED_DOUBLE).with_edge_margin(0.08)
            }
        }
    }

    /// Get the shader source for this fractal
    pub fn shader_source(&self) -> &'static str {
        match self {
            FractalType::Mandelbrot => Mandelbrot.shader_source(),
            FractalType::Julia { .. } => Julia::default().shader_source(),
            FractalType::BurningShip => BurningShip.shader_source(),
            FractalType::Tricorn => Tricorn.shader_source(),
            FractalType::Buddhabrot => Buddhabrot.shader_source(),
            FractalType::Nova { .. } => Nova::default().shader_source(),
            FractalType::GoldenSilver => GoldenSilver.shader_source(),
        }
    }

    /// Get the emulated f64 shader source for deep zoom
    pub fn shader_source_f64(&self) -> &'static str {
        match self {
            FractalType::Mandelbrot => Mandelbrot.shader_source_f64(),
            FractalType::Julia { .. } => Julia::default().shader_source_f64(),
            FractalType::BurningShip => BurningShip.shader_source_f64(),
            FractalType::Tricorn => Tricorn.shader_source_f64(),
            FractalType::Buddhabrot => Buddhabrot.shader_source_f64(),
            FractalType::Nova { .. } => Nova::default().shader_source_f64(),
            FractalType::GoldenSilver => GoldenSilver.shader_source_f64(),
        }
    }

    /// Get the human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            FractalType::Mandelbrot => "Mandelbrot Set",
            FractalType::Julia { .. } => "Julia Set",
            FractalType::BurningShip => "Burning Ship",
            FractalType::Tricorn => "Tricorn",
            FractalType::Buddhabrot => "Buddhabrot",
            FractalType::Nova { .. } => "Nova Fractal",
            FractalType::GoldenSilver => "Golden-Silver Gasket",
        }
    }

    /// Whether this fractal type uses accumulation buffer rendering
    pub fn is_buddhabrot(&self) -> bool {
        matches!(self, FractalType::Buddhabrot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wgpu::naga;

    /// Each fractal's default view must sit inside its own bounds.
    #[test]
    fn default_views_are_inside_bounds() {
        let all = [
            FractalType::Mandelbrot,
            FractalType::Julia { c: Vec2::ZERO },
            FractalType::BurningShip,
            FractalType::Tricorn,
            FractalType::Buddhabrot,
            FractalType::Nova { c: Vec2::ZERO },
            FractalType::GoldenSilver,
        ];
        for f in all {
            let b = f.view_bounds();
            let (c, z) = (f.default_center(), f.default_zoom());
            assert!(c.cmpge(b.min).all() && c.cmple(b.max).all(), "{}: centre", f.name());
            assert!(z >= b.min_zoom && z <= b.max_zoom, "{}: zoom", f.name());
        }
    }

    /// Every shipped shader must parse and validate.
    #[test]
    fn all_shaders_are_valid_wgsl() {
        let shaders = [
            ("mandelbrot", include_str!("../shaders/mandelbrot.wgsl")),
            ("mandelbrot_f64", include_str!("../shaders/mandelbrot_f64.wgsl")),
            ("julia", include_str!("../shaders/julia.wgsl")),
            ("julia_f64", include_str!("../shaders/julia_f64.wgsl")),
            ("burning_ship", include_str!("../shaders/burning_ship.wgsl")),
            ("burning_ship_f64", include_str!("../shaders/burning_ship_f64.wgsl")),
            ("tricorn", include_str!("../shaders/tricorn.wgsl")),
            ("tricorn_f64", include_str!("../shaders/tricorn_f64.wgsl")),
            ("nova", include_str!("../shaders/nova.wgsl")),
            ("nova_f64", include_str!("../shaders/nova_f64.wgsl")),
            ("buddhabrot", include_str!("../shaders/buddhabrot.wgsl")),
            ("buddhabrot_tonemap", include_str!("../shaders/buddhabrot_tonemap.wgsl")),
            ("fullscreen", include_str!("../shaders/fullscreen.wgsl")),
        ];
        for (name, source) in shaders {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|e| panic!("{name}: parse failed: {e}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::empty(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: validation failed: {e:?}"));
        }
    }
}
