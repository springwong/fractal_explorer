use glam::{DVec2, UVec2, Vec2};

/// Limits on where the camera may go, so the view never ends up in empty space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewBounds {
    /// Lowest allowed view centre (x, y)
    pub min: DVec2,
    /// Highest allowed view centre (x, y)
    pub max: DVec2,
    /// Furthest zoom-out allowed
    pub min_zoom: f64,
    /// Deepest zoom-in allowed (beyond this the precision runs out)
    pub max_zoom: f64,
}

impl ViewBounds {
    /// No limits at all
    pub const UNBOUNDED: Self = Self {
        min: DVec2::splat(f64::NEG_INFINITY),
        max: DVec2::splat(f64::INFINITY),
        min_zoom: 0.0,
        max_zoom: f64::INFINITY,
    };

    /// Bounds with centre limited to the box (x0, y0)..(x1, y1)
    pub fn new(x0: f64, y0: f64, x1: f64, y1: f64, min_zoom: f64, max_zoom: f64) -> Self {
        Self { min: DVec2::new(x0, y0), max: DVec2::new(x1, y1), min_zoom, max_zoom }
    }
}

/// Camera for navigating the complex plane
/// Uses f64 internally for deep zoom precision, converts to f32 for GPU
pub struct Camera {
    /// Center position in complex plane (real, imaginary)
    pub center: DVec2,
    /// Zoom level (higher = more zoomed in)
    pub zoom: f64,
    /// Rotation angle in radians
    pub rotation: f64,
    /// Screen dimensions in pixels
    pub screen_size: UVec2,
    /// Limits applied to centre and zoom
    pub bounds: ViewBounds,
}

impl Camera {
    /// Create a new camera with default Mandelbrot view
    pub fn new(screen_size: UVec2) -> Self {
        Self {
            center: DVec2::new(-0.5, 0.0), // Classic Mandelbrot center
            zoom: 1.0,
            rotation: 0.0,
            screen_size,
            bounds: ViewBounds::UNBOUNDED,
        }
    }

    /// Convert screen coordinates (pixels) to complex plane coordinates.
    ///
    /// Mirrors the shaders: offset from screen centre, normalised by height,
    /// rotated by `rotation`, scaled by 1/zoom, Y flipped.
    pub fn screen_to_complex(&self, screen_pos: Vec2) -> DVec2 {
        let size = self.screen_size.as_dvec2();
        let offset = (screen_pos.as_dvec2() - size / 2.0) / size.y;
        self.center + self.screen_vec_to_complex(offset)
    }

    /// Map a height-normalised screen-space vector to a complex-plane vector
    /// (rotation, zoom and Y flip; no translation).
    fn screen_vec_to_complex(&self, v: DVec2) -> DVec2 {
        let (sin_r, cos_r) = self.rotation.sin_cos();
        let rotated = DVec2::new(v.x * cos_r - v.y * sin_r, v.x * sin_r + v.y * cos_r);
        DVec2::new(rotated.x, -rotated.y) / self.zoom
    }

    /// Zoom in/out centered at a specific screen position
    /// factor > 1.0 zooms in, < 1.0 zooms out
    pub fn zoom_at(&mut self, screen_pos: Vec2, factor: f64) {
        // Get complex coordinate before zoom
        let complex_pos = self.screen_to_complex(screen_pos);

        // Apply zoom, staying inside the allowed range
        self.zoom = (self.zoom * factor).clamp(self.bounds.min_zoom, self.bounds.max_zoom);

        // Get complex coordinate after zoom
        let new_complex = self.screen_to_complex(screen_pos);

        // Adjust center to keep the point under cursor stationary
        self.center += complex_pos - new_complex;
        self.clamp_to_bounds();
    }

    /// Set the limits for the current fractal and pull the view back inside them
    pub fn set_bounds(&mut self, bounds: ViewBounds) {
        self.bounds = bounds;
        self.clamp_to_bounds();
    }

    /// Pull centre and zoom back inside `bounds`
    pub fn clamp_to_bounds(&mut self) {
        self.center = self.center.clamp(self.bounds.min, self.bounds.max);
        self.zoom = self.zoom.clamp(self.bounds.min_zoom, self.bounds.max_zoom);
    }

    /// Pan camera by screen delta (in pixels); the view follows the drag.
    pub fn pan(&mut self, screen_delta: Vec2) {
        let delta = screen_delta.as_dvec2() / self.screen_size.y as f64;
        self.center -= self.screen_vec_to_complex(delta);
        self.clamp_to_bounds();
    }

    /// Update screen size (called on window resize)
    pub fn resize(&mut self, new_size: UVec2) {
        self.screen_size = new_size;
    }

    /// Reset to default Mandelbrot view
    pub fn reset(&mut self) {
        self.center = DVec2::new(-0.5, 0.0);
        self.zoom = 1.0;
        self.rotation = 0.0;
    }

    /// Get current aspect ratio
    pub fn aspect_ratio(&self) -> f32 {
        self.screen_size.x as f32 / self.screen_size.y as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pixel -> plane mapping exactly as the compute shaders write it.
    fn shader_map(cam: &Camera, px: f64, py: f64) -> DVec2 {
        let (w, h) = (cam.screen_size.x as f64, cam.screen_size.y as f64);
        let (ux, uy) = ((px - w / 2.0) / h, (py - h / 2.0) / h);
        let (c, s) = (cam.rotation.cos(), cam.rotation.sin());
        let (rx, ry) = (ux * c - uy * s, ux * s + uy * c);
        DVec2::new(cam.center.x + rx / cam.zoom, cam.center.y - ry / cam.zoom)
    }

    fn cameras() -> Vec<Camera> {
        let mut out = Vec::new();
        for size in [UVec2::new(1920, 1080), UVec2::new(1080, 2340), UVec2::new(800, 800)] {
            for rotation in [0.0, 0.7, -2.1] {
                let mut cam = Camera::new(size);
                cam.center = DVec2::new(-0.3, 0.2);
                cam.zoom = 3.5;
                cam.rotation = rotation;
                out.push(cam);
            }
        }
        out
    }

    fn close(a: DVec2, b: DVec2) -> bool {
        (a - b).length() < 1e-9
    }

    #[test]
    fn screen_to_complex_matches_shader() {
        for cam in cameras() {
            let (w, h) = (cam.screen_size.x as f32, cam.screen_size.y as f32);
            for pos in [Vec2::ZERO, Vec2::new(w, h), Vec2::new(w, 0.0), Vec2::new(w * 0.25, h * 0.75)] {
                let got = cam.screen_to_complex(pos);
                let want = shader_map(&cam, pos.x as f64, pos.y as f64);
                assert!(close(got, want), "{:?} rot {}: {got} vs {want}", cam.screen_size, cam.rotation);
            }
        }
    }

    #[test]
    fn zoom_keeps_point_under_cursor() {
        for mut cam in cameras() {
            let pos = Vec2::new(cam.screen_size.x as f32 * 0.9, cam.screen_size.y as f32 * 0.2);
            let before = shader_map(&cam, pos.x as f64, pos.y as f64);
            cam.zoom_at(pos, 1.7);
            let after = shader_map(&cam, pos.x as f64, pos.y as f64);
            assert!(close(before, after), "{:?} rot {}", cam.screen_size, cam.rotation);
        }
    }

    #[test]
    fn bounds_stop_pan_and_zoom() {
        let mut cam = Camera::new(UVec2::new(1920, 1080));
        cam.set_bounds(ViewBounds::new(-2.0, -1.0, 1.0, 1.0, 0.25, 100.0));

        // Dragging far away stops at the edge of the box
        for _ in 0..200 {
            cam.pan(Vec2::new(-500.0, 300.0));
        }
        assert_eq!(cam.center, DVec2::new(1.0, 1.0));

        // Zooming stops at both limits, and at the limit the view no longer drifts
        let corner = Vec2::new(100.0, 100.0);
        for _ in 0..200 {
            cam.zoom_at(corner, 1.5);
        }
        assert_eq!(cam.zoom, 100.0);
        let settled = cam.center;
        cam.zoom_at(corner, 1.5);
        assert!(close(cam.center, settled));

        for _ in 0..200 {
            cam.zoom_at(corner, 0.5);
        }
        assert_eq!(cam.zoom, 0.25);

        // A view that starts outside is pulled back in
        cam.center = DVec2::new(50.0, -50.0);
        cam.zoom = 1e30;
        cam.set_bounds(cam.bounds);
        assert_eq!((cam.center, cam.zoom), (DVec2::new(1.0, -1.0), 100.0));
    }

    #[test]
    fn pan_drags_content_with_cursor() {
        for mut cam in cameras() {
            let start = Vec2::new(300.0, 400.0);
            let delta = Vec2::new(120.0, -45.0);
            let grabbed = shader_map(&cam, start.x as f64, start.y as f64);
            cam.pan(delta);
            let end = start + delta;
            let now_under = shader_map(&cam, end.x as f64, end.y as f64);
            assert!(close(grabbed, now_under), "{:?} rot {}", cam.screen_size, cam.rotation);
        }
    }
}
