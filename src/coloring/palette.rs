/// A color stop in a gradient palette
#[derive(Clone, Debug)]
pub struct ColorStop {
    /// Position along the gradient (0.0 to 1.0)
    pub position: f32,
    /// RGB color at this position
    pub color: [f32; 3],
}

/// A gradient palette defined by sorted color stops
#[derive(Clone, Debug)]
pub struct Palette {
    pub name: String,
    pub stops: Vec<ColorStop>,
}

impl Palette {
    /// Create a new palette with the given name and stops
    pub fn new(name: &str, stops: Vec<ColorStop>) -> Self {
        let mut palette = Self {
            name: name.to_string(),
            stops,
        };
        palette.sort_stops();
        palette
    }

    /// Sort stops by position
    fn sort_stops(&mut self) {
        self.stops.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
    }

    /// Sample color at position t (0.0 to 1.0) by linear interpolation
    pub fn sample_color(&self, t: f32) -> [f32; 3] {
        let t = t.clamp(0.0, 1.0);

        if self.stops.is_empty() {
            return [0.0, 0.0, 0.0];
        }
        if self.stops.len() == 1 {
            return self.stops[0].color;
        }

        // Find the two stops surrounding t
        if t <= self.stops[0].position {
            return self.stops[0].color;
        }
        if t >= self.stops.last().unwrap().position {
            return self.stops.last().unwrap().color;
        }

        for i in 0..self.stops.len() - 1 {
            let a = &self.stops[i];
            let b = &self.stops[i + 1];
            if t >= a.position && t <= b.position {
                let range = b.position - a.position;
                if range < 1e-6 {
                    return a.color;
                }
                let frac = (t - a.position) / range;
                return [
                    a.color[0] + (b.color[0] - a.color[0]) * frac,
                    a.color[1] + (b.color[1] - a.color[1]) * frac,
                    a.color[2] + (b.color[2] - a.color[2]) * frac,
                ];
            }
        }

        self.stops.last().unwrap().color
    }

    /// Generate a 256-entry RGBA8 lookup table as packed u32s (1024 bytes)
    pub fn generate_lut(&self) -> [u8; 1024] {
        let mut lut = [0u8; 1024];
        for i in 0..256 {
            let t = i as f32 / 255.0;
            let color = self.sample_color(t);
            let offset = i * 4;
            lut[offset] = (color[0].clamp(0.0, 1.0) * 255.0) as u8;
            lut[offset + 1] = (color[1].clamp(0.0, 1.0) * 255.0) as u8;
            lut[offset + 2] = (color[2].clamp(0.0, 1.0) * 255.0) as u8;
            lut[offset + 3] = 255;
        }
        lut
    }

    /// Add a color stop
    pub fn add_stop(&mut self, position: f32, color: [f32; 3]) {
        self.stops.push(ColorStop {
            position: position.clamp(0.0, 1.0),
            color,
        });
        self.sort_stops();
    }

    /// Remove a color stop by index (must keep at least 2 stops)
    pub fn remove_stop(&mut self, index: usize) -> bool {
        if self.stops.len() <= 2 || index >= self.stops.len() {
            return false;
        }
        self.stops.remove(index);
        true
    }

    /// Move a stop to a new position
    pub fn move_stop(&mut self, index: usize, new_position: f32) {
        if index < self.stops.len() {
            self.stops[index].position = new_position.clamp(0.0, 1.0);
            self.sort_stops();
        }
    }
}
