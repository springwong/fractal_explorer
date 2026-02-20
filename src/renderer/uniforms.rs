/// GPU uniform buffer for fractal rendering
/// Must maintain 16-byte alignment for WGSL compatibility
///
/// WGSL struct layout (64 bytes total):
/// offset 0:  center (vec2<f32>) - 8 bytes
/// offset 8:  zoom (f32) - 4 bytes
/// offset 12: aspect_ratio (f32) - 4 bytes
/// offset 16: max_iter (u32) - 4 bytes
/// offset 20: fractal_type (u32) - 4 bytes
/// offset 24: color_scheme (u32) - 4 bytes
/// offset 28: c_real (f32) - 4 bytes (Julia parameter)
/// offset 32: c_imag (f32) - 4 bytes (Julia parameter)
/// offset 36: _padding1 (u32) - 4 bytes (align to 16)
/// offset 40: _padding2 (u32) - 4 bytes
/// offset 44: _padding3 (u32) - 4 bytes
/// offset 48: _padding4 (vec4<u32>) - 16 bytes (vec3 in WGSL takes 16 bytes!)
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
    /// Padding to align to 64 bytes (vec3 in WGSL = 16 bytes)
    pub _padding1: u32,
    pub _padding2: u32,
    pub _padding3: u32,
    pub _padding4: [u32; 4],  // vec3 in WGSL takes 16 bytes due to alignment
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
            _padding1: 0,
            _padding2: 0,
            _padding3: 0,
            _padding4: [0; 4],
        }
    }
}

// Compile-time assertion for size (must match WGSL layout)
const _: () = assert!(std::mem::size_of::<FractalUniforms>() == 64);
const _: () = assert!(std::mem::size_of::<FractalUniforms>() % 16 == 0);
