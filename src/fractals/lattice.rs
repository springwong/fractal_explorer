//! Exact view placement for the fractals drawn from the digits of a point's
//! coordinates (Sierpinski triangle, Sierpinski carpet).
//!
//! The shader works in integers (see `shaders/lattice.wgsl`): each coordinate is
//! scaled by base^levels, with `levels` chosen so that one unit is a fraction of a
//! pixel. The view centre is converted here in f64; a pixel only adds a small offset.

use crate::renderer::FractalUniforms;
use glam::{DMat2, DVec2};

/// How a fractal's plane maps onto its digit lattice
pub struct Lattice {
    /// Plane -> lattice coordinates
    pub to_lattice: DMat2,
    /// Added to the lattice coordinates (keeps the picture's surroundings positive)
    pub offset: DVec2,
    /// Digit base
    pub base: f64,
    /// Lattice units per pixel, at least
    pub units_per_pixel: f64,
    /// Most digit levels the scaled coordinates can hold in 63 bits
    pub max_levels: u32,
}

/// Sierpinski triangle with corners (0, 0), (1, 0), (1/2, sqrt(3)/2): the coordinates
/// are the weights on the two edges leaving the first corner, read in base 2.
pub const SIERPINSKI_TRIANGLE: Lattice = Lattice {
    // u = x - y / sqrt(3), v = 2y / sqrt(3)
    to_lattice: DMat2::from_cols_array(&[1.0, 0.0, -0.577_350_269_189_625_8, 1.154_700_538_379_251_5]),
    offset: DVec2::ZERO,
    base: 2.0,
    units_per_pixel: 4.0,
    max_levels: 60,
};

/// Sierpinski carpet on the unit square, read in base 3. Shifted by one so that the
/// square is the middle cell of a 3 x 3 block.
pub const SIERPINSKI_CARPET: Lattice = Lattice {
    to_lattice: DMat2::IDENTITY,
    offset: DVec2::ONE,
    base: 3.0,
    units_per_pixel: 3.0,
    // (x + 1) * 3^levels < 3^(levels + 1) must stay below 2^63
    max_levels: 38,
};

/// Number of digit levels for a view at `zoom`, `height` pixels high
fn levels(lattice: &Lattice, zoom: f64, height: u32) -> u32 {
    let units_per_plane_unit = zoom * height as f64 * lattice.units_per_pixel;
    let levels = (units_per_plane_unit.ln() / lattice.base.ln()).ceil();
    (levels.max(1.0) as u32).min(lattice.max_levels)
}

/// The view centre in scaled lattice units, and the matrix taking a pixel offset
/// from the centre (x right, y down, before the view rotation) to scaled units
fn place(lattice: &Lattice, center: DVec2, zoom: f64, height: u32) -> (u32, [i64; 2], DMat2) {
    let levels = levels(lattice, zoom, height);
    let scale = lattice.base.powi(levels as i32);
    let scaled = (lattice.to_lattice * center + lattice.offset) * scale;
    let pixel_step = 1.0 / (zoom * height as f64);
    // Same pixel -> plane step as the other shaders: x right, y up
    let pixel_to_units = lattice.to_lattice * DMat2::from_diagonal(DVec2::new(pixel_step, -pixel_step)) * scale;
    (levels, [scaled.x.floor() as i64, scaled.y.floor() as i64], pixel_to_units)
}

/// Fill the lattice uniform slots for the view the uniforms describe, rendered
/// `height` pixels high.
pub fn apply_view(u: &mut FractalUniforms, lattice: &Lattice, height: u32) {
    let center = DVec2::new(
        u.center[0] as f64 + u.center_lo_x as f64,
        u.center[1] as f64 + u.center_lo_y as f64,
    );
    let zoom = u.zoom as f64 + u.zoom_lo as f64;
    let (levels, [x, y], m) = place(lattice, center, zoom, height);

    u.extra = [x as u32, (x >> 32) as u32, y as u32];
    u.color_scheme = (y >> 32) as u32;
    u.ref_escape_iter = levels;
    u.c_real = m.x_axis.x as f32;
    u.c_imag = m.y_axis.x as f32;
    u.center_lo_x = m.x_axis.y as f32;
    u.center_lo_y = m.y_axis.y as f32;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One unit is a fraction of a pixel, but never so small that a pixel offset
    /// would not fit an i32 or the coordinates an i64.
    #[test]
    fn unit_is_a_fraction_of_a_pixel() {
        for lattice in [&SIERPINSKI_TRIANGLE, &SIERPINSKI_CARPET] {
            for (zoom, height) in [(0.3, 64), (1.0, 1080), (3.7e3, 1080), (1.0e9, 4320), (1.0e13, 1080)] {
                let (levels, centre, m) = place(lattice, DVec2::new(0.9, 0.8), zoom, height);
                assert!(levels <= lattice.max_levels);
                // Units per pixel along x
                let units = m.x_axis.length();
                assert!(units >= 1.0, "{units} units per pixel at zoom {zoom}");
                assert!(units * 16384.0 < i32::MAX as f64, "{units} units per pixel at zoom {zoom}");
                assert!(centre.iter().all(|&c| c > 0 && c < i64::MAX / 4));
            }
        }
    }

    /// The triangle's corners land on (0,0), (1,0) and (0,1) of its lattice.
    #[test]
    fn triangle_corners_are_lattice_corners() {
        let m = SIERPINSKI_TRIANGLE.to_lattice;
        let top = m * DVec2::new(0.5, 3f64.sqrt() / 2.0);
        assert!((m * DVec2::X - DVec2::X).length() < 1e-15);
        assert!((top - DVec2::Y).length() < 1e-15, "{top}");
    }
}
