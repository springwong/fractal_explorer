/// Fractal trait and type system
mod mandelbrot;
mod julia;
mod burning_ship;
mod tricorn;
mod buddhabrot;
mod nova;

pub use mandelbrot::Mandelbrot;
pub use julia::Julia;
pub use burning_ship::BurningShip;
pub use tricorn::Tricorn;
pub use buddhabrot::Buddhabrot;
pub use nova::Nova;

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
        }
    }

    /// Whether this fractal type uses accumulation buffer rendering
    pub fn is_buddhabrot(&self) -> bool {
        matches!(self, FractalType::Buddhabrot)
    }
}
