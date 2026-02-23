/// Color scheme system for fractal rendering
pub mod palette;
pub mod presets;

pub use palette::Palette;
pub use presets::PresetPalettes;

/// Color scheme selector
#[derive(Clone, Debug)]
pub enum ColorScheme {
    /// A built-in preset palette by index
    Preset(usize),
    /// A user-defined custom palette
    Custom(Palette),
}

impl ColorScheme {
    /// Get the human-readable name
    pub fn name(&self) -> String {
        match self {
            ColorScheme::Preset(idx) => {
                let presets = PresetPalettes::all();
                if *idx < presets.len() {
                    presets[*idx].name.clone()
                } else {
                    "Unknown".to_string()
                }
            }
            ColorScheme::Custom(p) => p.name.clone(),
        }
    }

    /// Cycle to the next preset color scheme
    pub fn next(&self) -> Self {
        let count = PresetPalettes::all().len();
        match self {
            ColorScheme::Preset(idx) => ColorScheme::Preset((*idx + 1) % count),
            ColorScheme::Custom(_) => ColorScheme::Preset(0),
        }
    }

    /// Get the active palette
    pub fn get_palette(&self) -> Palette {
        match self {
            ColorScheme::Preset(idx) => {
                let presets = PresetPalettes::all();
                if *idx < presets.len() {
                    presets[*idx].clone()
                } else {
                    presets[0].clone()
                }
            }
            ColorScheme::Custom(p) => p.clone(),
        }
    }

    /// Get the preset index, if this is a preset
    pub fn preset_index(&self) -> Option<usize> {
        match self {
            ColorScheme::Preset(idx) => Some(*idx),
            ColorScheme::Custom(_) => None,
        }
    }
}

impl Default for ColorScheme {
    fn default() -> Self {
        ColorScheme::Preset(0)
    }
}

impl PartialEq for ColorScheme {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (ColorScheme::Preset(a), ColorScheme::Preset(b)) => a == b,
            _ => false,
        }
    }
}
