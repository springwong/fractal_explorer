/// GPU uniform buffer for fractal rendering
/// Must maintain 16-byte alignment for WGSL compatibility
///
/// WGSL struct layout (48 bytes total):
/// offset 0:  center (vec2<f32>) - 8 bytes
/// offset 8:  zoom (f32) - 4 bytes
/// offset 12: aspect_ratio (f32) - 4 bytes
/// offset 16: max_iter (u32) - 4 bytes
/// offset 20: padding - 12 bytes (to align vec3 to 16-byte boundary)
/// offset 32: _padding (vec3<u32>) - 12 bytes (vec3 size)
/// offset 44: padding - 4 bytes (to make total 48 bytes, multiple of 16)
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
    /// Padding before vec3 (12 bytes to align vec3 to offset 32)
    pub _padding1: [u32; 3],
    /// vec3 padding (12 bytes)
    pub _padding2: [u32; 3],
    /// Final padding to make total 48 bytes
    pub _padding3: u32,
}

impl FractalUniforms {
    pub fn new(center: [f32; 2], zoom: f32, aspect_ratio: f32, max_iter: u32) -> Self {
        Self {
            center,
            zoom,
            aspect_ratio,
            max_iter,
            _padding1: [0; 3],
            _padding2: [0; 3],
            _padding3: 0,
        }
    }
}

// Compile-time assertion for size (must match WGSL layout)
const _: () = assert!(std::mem::size_of::<FractalUniforms>() == 48);
const _: () = assert!(std::mem::size_of::<FractalUniforms>() % 16 == 0);
