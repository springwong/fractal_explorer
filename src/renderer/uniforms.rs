/// GPU uniform buffer for fractal rendering
/// Must maintain 16-byte alignment for WGSL compatibility
///
/// WGSL struct layout (80 bytes total):
/// offset 0:  center (vec2<f32>) - 8 bytes
/// offset 8:  zoom (f32) - 4 bytes
/// offset 12: aspect_ratio (f32) - 4 bytes
/// offset 16: max_iter (u32) - 4 bytes
/// offset 20: fractal_type (u32) - 4 bytes
/// offset 24: color_scheme (u32) - 4 bytes
/// offset 28: c_real (f32) - 4 bytes (Julia parameter)
/// offset 32: c_imag (f32) - 4 bytes (Julia parameter)
/// offset 36: center_lo_x (f32) - 4 bytes (emulated double: low bits of center.x)
/// offset 40: center_lo_y (f32) - 4 bytes (emulated double: low bits of center.y)
/// offset 44: zoom_lo (f32) - 4 bytes (emulated double: low bits of zoom)
/// offset 48: pixel_step_x (f32) - 4 bytes (per-pixel step in x, computed on CPU in f64)
/// offset 52: pixel_step_y (f32) - 4 bytes (per-pixel step in y, computed on CPU in f64)
/// offset 56: ref_escape_iter (u32) - 4 bytes (iteration where reference orbit escapes)
/// offset 60: rotation (f32) - 4 bytes (view rotation in radians)
/// offset 64: _pad2 ([u32; 3]) - 12 bytes (padding to 80 bytes, 16-byte aligned)
/// offset 76: _pad3 (u32) - 4 bytes
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FractalUniforms {
    /// Center of the complex plane view (real, imaginary)
    pub center: [f32; 2],
    /// Zoom level (higher = more zoomed in)
    pub zoom: f32,
    /// Screen aspect ratio (width / height)
    pub aspect_ratio: f32,
    /// Maximum iteration count for escape-time algorithm
    pub max_iter: u32,
    /// Fractal type selector (0=Mandelbrot, 1=Julia, 2=BurningShip, 3=Tricorn)
    pub fractal_type: u32,
    /// Color scheme selector (0=Smooth, 1=Fire, 2=Ocean, 3=Grayscale)
    pub color_scheme: u32,
    /// Julia set c parameter (real part)
    pub c_real: f32,
    /// Julia set c parameter (imaginary part)
    pub c_imag: f32,
    /// Emulated double: low bits of center.x
    pub center_lo_x: f32,
    /// Emulated double: low bits of center.y
    pub center_lo_y: f32,
    /// Emulated double: low bits of zoom
    pub zoom_lo: f32,
    /// Per-pixel step in x direction (1.0 / (zoom * height)), computed on CPU in f64
    pub pixel_step_x: f32,
    /// Per-pixel step in y direction (-1.0 / (zoom * height)), computed on CPU in f64
    pub pixel_step_y: f32,
    /// Iteration where reference orbit escapes (or max_iter if it doesn't)
    pub ref_escape_iter: u32,
    /// View rotation angle in radians
    pub rotation: f32,
    /// Padding to maintain 80-byte alignment (16-byte boundary)
    pub _pad2: [u32; 3],
    pub _pad3: u32,
}

impl FractalUniforms {
    pub fn new(
        center: [f32; 2],
        zoom: f32,
        aspect_ratio: f32,
        max_iter: u32,
        fractal_type: u32,
        color_scheme: u32,
        c_real: f32,
        c_imag: f32,
        center_lo: [f32; 2],
        zoom_lo: f32,
        pixel_step_x: f32,
        pixel_step_y: f32,
        ref_escape_iter: u32,
        rotation: f32,
    ) -> Self {
        Self {
            center,
            zoom,
            aspect_ratio,
            max_iter,
            fractal_type,
            color_scheme,
            c_real,
            c_imag,
            center_lo_x: center_lo[0],
            center_lo_y: center_lo[1],
            zoom_lo,
            pixel_step_x,
            pixel_step_y,
            ref_escape_iter,
            rotation,
            _pad2: [0; 3],
            _pad3: 0,
        }
    }
}

/// Split an f64 value into two f32 components (hi + lo) for emulated double precision.
pub fn ds_split(value: f64) -> (f32, f32) {
    let hi = value as f32;
    let lo = (value - hi as f64) as f32;
    (hi, lo)
}

/// Compute the reference orbit at center in f64 precision (for perturbation theory).
/// Returns (orbit_data, escape_iter) where orbit_data is a flat Vec of [zx, zy] f32 pairs,
/// and escape_iter is when the reference escapes (or max_iter if it stays bounded).
pub fn compute_reference_orbit(center_x: f64, center_y: f64, max_iter: u32) -> (Vec<f32>, u32) {
    let capacity = (max_iter as usize + 1) * 2;
    let mut orbit = Vec::with_capacity(capacity);
    let mut zx: f64 = 0.0;
    let mut zy: f64 = 0.0;
    let mut escape_iter = max_iter;

    // Store z_0 = (0, 0)
    orbit.push(0.0f32);
    orbit.push(0.0f32);

    for i in 0..max_iter {
        let new_zx = zx * zx - zy * zy + center_x;
        let new_zy = 2.0 * zx * zy + center_y;
        zx = new_zx;
        zy = new_zy;
        orbit.push(zx as f32);
        orbit.push(zy as f32);

        if zx * zx + zy * zy > 1e10 {
            escape_iter = i + 1;
            // Pad remaining entries
            while orbit.len() < capacity {
                orbit.push(0.0f32);
                orbit.push(0.0f32);
            }
            break;
        }
    }

    // Ensure we have exactly the right number of entries
    while orbit.len() < capacity {
        orbit.push(0.0f32);
        orbit.push(0.0f32);
    }

    (orbit, escape_iter)
}

/// Compute the reference orbit for Julia set perturbation theory.
/// Julia iteration: z_{n+1} = z_n^2 + c, starting from z_0 = center (the view center).
/// Returns (orbit_data, escape_iter) as flat Vec of [zx, zy] f32 pairs.
pub fn compute_reference_orbit_julia(
    z0_x: f64, z0_y: f64,
    c_real: f64, c_imag: f64,
    max_iter: u32,
) -> (Vec<f32>, u32) {
    let capacity = (max_iter as usize + 1) * 2;
    let mut orbit = Vec::with_capacity(capacity);
    let mut zx: f64 = z0_x;
    let mut zy: f64 = z0_y;
    let mut escape_iter = max_iter;

    // Store z_0
    orbit.push(zx as f32);
    orbit.push(zy as f32);

    for i in 0..max_iter {
        let new_zx = zx * zx - zy * zy + c_real;
        let new_zy = 2.0 * zx * zy + c_imag;
        zx = new_zx;
        zy = new_zy;
        orbit.push(zx as f32);
        orbit.push(zy as f32);

        if zx * zx + zy * zy > 1e10 {
            escape_iter = i + 1;
            while orbit.len() < capacity {
                orbit.push(0.0f32);
                orbit.push(0.0f32);
            }
            break;
        }
    }

    while orbit.len() < capacity {
        orbit.push(0.0f32);
        orbit.push(0.0f32);
    }

    (orbit, escape_iter)
}

// Compile-time assertion for size (must match WGSL layout)
const _: () = assert!(std::mem::size_of::<FractalUniforms>() == 80);
const _: () = assert!(std::mem::size_of::<FractalUniforms>() % 16 == 0);
