/// Color scheme system for fractal rendering

/// Color scheme selector
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ColorScheme {
    Smooth = 0,
    Fire = 1,
    Ocean = 2,
    Grayscale = 3,
}

impl ColorScheme {
    /// Get the numeric ID for uniform buffer
    pub fn to_id(&self) -> u32 {
        *self as u32
    }

    /// Get the human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            ColorScheme::Smooth => "Smooth Rainbow",
            ColorScheme::Fire => "Fire",
            ColorScheme::Ocean => "Ocean",
            ColorScheme::Grayscale => "Grayscale",
        }
    }

    /// Cycle to the next color scheme
    pub fn next(&self) -> Self {
        match self {
            ColorScheme::Smooth => ColorScheme::Fire,
            ColorScheme::Fire => ColorScheme::Ocean,
            ColorScheme::Ocean => ColorScheme::Grayscale,
            ColorScheme::Grayscale => ColorScheme::Smooth,
        }
    }

    /// All available color schemes
    pub fn all() -> &'static [ColorScheme] {
        &[
            ColorScheme::Smooth,
            ColorScheme::Fire,
            ColorScheme::Ocean,
            ColorScheme::Grayscale,
        ]
    }
}

impl Default for ColorScheme {
    fn default() -> Self {
        ColorScheme::Smooth
    }
}
