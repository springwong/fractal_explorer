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

    /// Convert screen coordinates (pixels) to complex plane coordinates
    pub fn screen_to_complex(&self, screen_pos: Vec2) -> DVec2 {
        let aspect = self.screen_size.x as f64 / self.screen_size.y as f64;

        // Convert to normalized device coordinates centered at (0, 0)
        let ndc = (screen_pos - self.screen_size.as_vec2() / 2.0) / self.screen_size.y as f32;

        // Scale by zoom and aspect ratio, flip Y axis
        self.center
            + DVec2::new(
                ndc.x as f64 * aspect / self.zoom,
                -ndc.y as f64 / self.zoom,
            )
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

    /// Pan camera by screen delta (in pixels)
    pub fn pan(&mut self, screen_delta: Vec2) {
        let aspect = self.screen_size.x as f64 / self.screen_size.y as f64;

        // Convert screen delta to complex plane delta
        let complex_delta = DVec2::new(
            -screen_delta.x as f64 * aspect / self.zoom / self.screen_size.y as f64,
            screen_delta.y as f64 / self.zoom / self.screen_size.y as f64,
        );

        self.center += complex_delta;
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
