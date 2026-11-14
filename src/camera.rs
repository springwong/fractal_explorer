use glam::{DVec2, UVec2, Vec2};

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
}

impl Camera {
    /// Create a new camera with default Mandelbrot view
    pub fn new(screen_size: UVec2) -> Self {
        Self {
            center: DVec2::new(-0.5, 0.0), // Classic Mandelbrot center
            zoom: 1.0,
            rotation: 0.0,
            screen_size,
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

        // Apply zoom
        self.zoom *= factor;

        // Get complex coordinate after zoom
        let new_complex = self.screen_to_complex(screen_pos);

        // Adjust center to keep the point under cursor stationary
        self.center += complex_pos - new_complex;
    }

    /// Pan camera by screen delta (in pixels); the view follows the drag.
    pub fn pan(&mut self, screen_delta: Vec2) {
        let delta = screen_delta.as_dvec2() / self.screen_size.y as f64;
        self.center -= self.screen_vec_to_complex(delta);
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
