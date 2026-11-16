use crate::fractals::FractalType;
use glam::DVec2;

/// What to look at: the camera state and iteration budget a frame is rendered for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    /// Centre of the view on the complex plane
    pub center: DVec2,
    /// Zoom level (the view is 1 / zoom plane units high)
    pub zoom: f64,
    /// View rotation in radians
    pub rotation: f64,
    /// Maximum iteration count
    pub max_iter: u32,
}

/// GPU uniform buffer for fractal rendering.
///
/// 80 bytes, 16-byte aligned. The WGSL side of this layout is the `Uniforms` struct in
/// `src/shaders/common.wgsl` — the only other place to touch when the layout changes
/// (the `layout_matches_wgsl` test compares the two).
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FractalUniforms {
    /// Center of the complex plane view (real, imaginary), high part
    pub center: [f32; 2],
    /// Zoom level (higher = more zoomed in), high part
    pub zoom: f32,
    /// Screen aspect ratio (width / height)
    pub aspect_ratio: f32,
    /// Maximum iteration count for escape-time algorithm
    pub max_iter: u32,
    /// Fractal id (`FractalInfo::id`)
    pub fractal_type: u32,
    /// Unused by the palette-LUT shaders; the gasket stores its coarse cell codes here
    pub color_scheme: u32,
    /// Julia / Nova c parameter (real part)
    pub c_real: f32,
    /// Julia / Nova c parameter (imaginary part)
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
    /// Iteration where reference orbit escapes (or max_iter if it doesn't).
    /// Buddhabrot uses it as the per-frame random seed.
    pub ref_escape_iter: u32,
    /// View rotation angle in radians
    pub rotation: f32,
    /// Per-fractal data (Buddhabrot: width, height, accumulated batches; gasket: pixel step bits)
    pub extra: [u32; 3],
    /// Padding to 80 bytes (16-byte boundary)
    pub _pad3: u32,
}

impl FractalUniforms {
    /// Uniforms for rendering `fractal` at `view` into a `width` x `height` target.
    ///
    /// This is the one place the camera's f64 state is turned into GPU values, so the
    /// live view, PNG export and video frames cannot disagree.
    pub fn for_view(fractal: &FractalType, view: &View, width: u32, height: u32) -> Self {
        // Split f64 camera values into hi+lo f32 pairs for emulated double precision
        let (center_x, center_lo_x) = ds_split(view.center.x);
        let (center_y, center_lo_y) = ds_split(view.center.y);
        let (zoom, zoom_lo) = ds_split(view.zoom);
        // Per-pixel step, computed here in f64 for precision
        let pixel_step = 1.0 / (view.zoom * height as f64);
        let c = fractal.c();

        let mut uniforms = Self {
            center: [center_x, center_y],
            zoom,
            aspect_ratio: width as f32 / height as f32,
            max_iter: view.max_iter,
            fractal_type: fractal.id(),
            color_scheme: 0,
            c_real: c.x,
            c_imag: c.y,
            center_lo_x,
            center_lo_y,
            zoom_lo,
            pixel_step_x: pixel_step as f32,
            pixel_step_y: -pixel_step as f32,
            ref_escape_iter: view.max_iter,
            rotation: view.rotation as f32,
            extra: [0; 3],
            _pad3: 0,
        };
        fractal.prepare_uniforms(&mut uniforms, width, height);
        uniforms
    }
}

/// Split an f64 value into two f32 components (hi + lo) for emulated double precision.
pub fn ds_split(value: f64) -> (f32, f32) {
    let hi = value as f32;
    let lo = (value - hi as f64) as f32;
    (hi, lo)
}

/// Compute a reference orbit of z_{n+1} = z_n^2 + c in f64 (for perturbation theory).
///
/// Mandelbrot uses z_0 = 0 and c = view centre; Julia uses z_0 = view centre and its
/// fixed c. Returns (orbit_data, escape_iter): orbit_data holds max_iter + 1 [zx, zy]
/// f32 pairs (zero-padded after an escape), and escape_iter is when the reference
/// escapes (or max_iter if it stays bounded).
pub fn reference_orbit(z0: DVec2, c: DVec2, max_iter: u32) -> (Vec<f32>, u32) {
    let len = (max_iter as usize + 1) * 2;
    let mut orbit = Vec::with_capacity(len);
    let mut z = z0;
    let mut escape_iter = max_iter;

    orbit.extend([z.x as f32, z.y as f32]);
    for i in 0..max_iter {
        z = DVec2::new(z.x * z.x - z.y * z.y + c.x, 2.0 * z.x * z.y + c.y);
        orbit.extend([z.x as f32, z.y as f32]);
        if z.length_squared() > 1e10 {
            escape_iter = i + 1;
            break;
        }
    }
    orbit.resize(len, 0.0);

    (orbit, escape_iter)
}

// Compile-time assertion for size (must match WGSL layout)
const _: () = assert!(std::mem::size_of::<FractalUniforms>() == 80);
const _: () = assert!(std::mem::size_of::<FractalUniforms>() % 16 == 0);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::shader;
    use std::mem::offset_of;
    use wgpu::naga;

    /// The WGSL `Uniforms` struct must have the same size and member offsets as the
    /// Rust struct, member by member.
    #[test]
    fn layout_matches_wgsl() {
        let module = naga::front::wgsl::parse_str(shader::common_source()).expect("common.wgsl parses");
        let (members, span) = module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(name), naga::TypeInner::Struct { members, span }) if name == "Uniforms" => {
                    Some((members, *span))
                }
                _ => None,
            })
            .expect("common.wgsl declares Uniforms");

        let wgsl: Vec<(String, u32)> =
            members.iter().map(|m| (m.name.clone().unwrap_or_default(), m.offset)).collect();
        let rust: Vec<(String, u32)> = [
            ("center", offset_of!(FractalUniforms, center)),
            ("zoom", offset_of!(FractalUniforms, zoom)),
            ("aspect_ratio", offset_of!(FractalUniforms, aspect_ratio)),
            ("max_iter", offset_of!(FractalUniforms, max_iter)),
            ("fractal_type", offset_of!(FractalUniforms, fractal_type)),
            ("color_scheme", offset_of!(FractalUniforms, color_scheme)),
            ("c_real", offset_of!(FractalUniforms, c_real)),
            ("c_imag", offset_of!(FractalUniforms, c_imag)),
            ("center_lo_x", offset_of!(FractalUniforms, center_lo_x)),
            ("center_lo_y", offset_of!(FractalUniforms, center_lo_y)),
            ("zoom_lo", offset_of!(FractalUniforms, zoom_lo)),
            ("pixel_step_x", offset_of!(FractalUniforms, pixel_step_x)),
            ("pixel_step_y", offset_of!(FractalUniforms, pixel_step_y)),
            ("ref_escape_iter", offset_of!(FractalUniforms, ref_escape_iter)),
            ("rotation", offset_of!(FractalUniforms, rotation)),
            ("extra", offset_of!(FractalUniforms, extra)),
            ("_pad3", offset_of!(FractalUniforms, _pad3)),
        ]
        .into_iter()
        .map(|(name, offset)| (name.to_string(), offset as u32))
        .collect();

        assert_eq!(wgsl, rust);
        assert_eq!(span as usize, std::mem::size_of::<FractalUniforms>());
    }

    /// The orbit always has max_iter + 1 entries, and stops advancing once it escapes.
    #[test]
    fn reference_orbit_shape() {
        // c = 0: stays at the origin forever
        let (orbit, escape) = reference_orbit(DVec2::ZERO, DVec2::ZERO, 32);
        assert_eq!((orbit.len(), escape), (66, 32));
        assert!(orbit.iter().all(|v| *v == 0.0));

        // c = 2: 0, 2, 6, 38, 1446, 2090918 -> |z|^2 > 1e10 at step 5
        let (orbit, escape) = reference_orbit(DVec2::ZERO, DVec2::new(2.0, 0.0), 32);
        assert_eq!((orbit.len(), escape), (66, 5));
        assert_eq!(&orbit[..12], &[0.0, 0.0, 2.0, 0.0, 6.0, 0.0, 38.0, 0.0, 1446.0, 0.0, 2090918.0, 0.0]);
        assert!(orbit[12..].iter().all(|v| *v == 0.0));

        // Julia: starts at z_0
        let (orbit, _) = reference_orbit(DVec2::new(0.25, -0.5), DVec2::ZERO, 4);
        assert_eq!(&orbit[..2], &[0.25, -0.5]);
    }
}
