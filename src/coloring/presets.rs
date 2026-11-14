use super::palette::{ColorStop, Palette};

/// Built-in palette presets
pub struct PresetPalettes;

impl PresetPalettes {
    /// All built-in presets. Indices 0-3 match the old hardcoded shader color schemes.
    pub fn all() -> Vec<Palette> {
        vec![
            Self::smooth_rainbow(),
            Self::fire(),
            Self::ocean(),
            Self::grayscale(),
            Self::electric(),
            Self::copper(),
            Self::nebula(),
            Self::gold_silver(),
            Self::stained_glass(),
            Self::sakura(),
            Self::twilight(),
            Self::jade(),
            Self::ink_wash(),
        ]
    }

    /// Preset 0: Smooth Rainbow — matches old colorize_smooth (HSV rainbow)
    fn smooth_rainbow() -> Palette {
        // HSV rainbow with sat=0.8, val=0.9
        // Sample the old HSV function at key hue positions
        Palette::new("Smooth Rainbow", vec![
            ColorStop { position: 0.0,    color: [0.9, 0.18, 0.18] },   // red
            ColorStop { position: 0.167,  color: [0.9, 0.9, 0.18] },    // yellow
            ColorStop { position: 0.333,  color: [0.18, 0.9, 0.18] },   // green
            ColorStop { position: 0.5,    color: [0.18, 0.9, 0.9] },    // cyan
            ColorStop { position: 0.667,  color: [0.18, 0.18, 0.9] },   // blue
            ColorStop { position: 0.833,  color: [0.9, 0.18, 0.9] },    // magenta
            ColorStop { position: 1.0,    color: [0.9, 0.18, 0.18] },   // red (wrap)
        ])
    }

    /// Preset 1: Fire — matches old colorize_fire
    fn fire() -> Palette {
        Palette::new("Fire", vec![
            ColorStop { position: 0.0,   color: [0.0, 0.0, 0.0] },
            ColorStop { position: 0.15,  color: [0.3, 0.0, 0.0] },
            ColorStop { position: 0.3,   color: [0.6, 0.0, 0.0] },
            ColorStop { position: 0.5,   color: [1.0, 0.5, 0.0] },
            ColorStop { position: 0.7,   color: [1.0, 1.0, 0.0] },
            ColorStop { position: 1.0,   color: [1.0, 1.0, 1.0] },
        ])
    }

    /// Preset 2: Ocean — matches old colorize_ocean
    fn ocean() -> Palette {
        Palette::new("Ocean", vec![
            ColorStop { position: 0.0,   color: [0.0, 0.0, 0.3] },
            ColorStop { position: 0.2,   color: [0.0, 0.0, 0.44] },
            ColorStop { position: 0.4,   color: [0.0, 0.36, 0.58] },
            ColorStop { position: 0.6,   color: [0.0, 0.72, 0.72] },
            ColorStop { position: 0.8,   color: [0.5, 1.0, 0.86] },
            ColorStop { position: 1.0,   color: [1.0, 1.0, 1.0] },
        ])
    }

    /// Preset 3: Grayscale — matches old colorize_grayscale
    fn grayscale() -> Palette {
        Palette::new("Grayscale", vec![
            ColorStop { position: 0.0, color: [0.0, 0.0, 0.0] },
            ColorStop { position: 1.0, color: [1.0, 1.0, 1.0] },
        ])
    }

    /// Preset 4: Electric — new preset
    fn electric() -> Palette {
        Palette::new("Electric", vec![
            ColorStop { position: 0.0,   color: [0.0, 0.0, 0.1] },
            ColorStop { position: 0.25,  color: [0.1, 0.0, 0.8] },
            ColorStop { position: 0.5,   color: [0.0, 0.8, 1.0] },
            ColorStop { position: 0.75,  color: [1.0, 1.0, 0.2] },
            ColorStop { position: 1.0,   color: [1.0, 1.0, 1.0] },
        ])
    }

    /// Preset 5: Copper — new preset
    fn copper() -> Palette {
        Palette::new("Copper", vec![
            ColorStop { position: 0.0,   color: [0.0, 0.0, 0.0] },
            ColorStop { position: 0.3,   color: [0.4, 0.2, 0.05] },
            ColorStop { position: 0.6,   color: [0.8, 0.5, 0.2] },
            ColorStop { position: 0.85,  color: [1.0, 0.8, 0.5] },
            ColorStop { position: 1.0,   color: [1.0, 0.95, 0.8] },
        ])
    }

    /// Preset 6: Nebula — matches old colorize_nebula (Buddhabrot-tuned)
    fn nebula() -> Palette {
        // Approximate pow curves: r=n^0.4*0.8, g=n^0.7*0.6, b=n^1.2
        Palette::new("Nebula", vec![
            ColorStop { position: 0.0,   color: [0.0, 0.0, 0.0] },
            ColorStop { position: 0.1,   color: [0.505, 0.12, 0.063] },
            ColorStop { position: 0.25,  color: [0.594, 0.226, 0.178] },
            ColorStop { position: 0.5,   color: [0.696, 0.37, 0.435] },
            ColorStop { position: 0.75,  color: [0.757, 0.5, 0.706] },
            ColorStop { position: 1.0,   color: [0.8, 0.6, 1.0] },
        ])
    }

    /// Preset 7: Gold & Silver — navy, bronze, gold and silver (cyclic)
    fn gold_silver() -> Palette {
        Palette::new("Gold & Silver", vec![
            ColorStop { position: 0.0 , color: [0.05, 0.07, 0.16] },
            ColorStop { position: 0.2 , color: [0.55, 0.38, 0.1] },
            ColorStop { position: 0.4 , color: [0.96, 0.78, 0.3] },
            ColorStop { position: 0.55, color: [1.0, 0.95, 0.8] },
            ColorStop { position: 0.75, color: [0.72, 0.76, 0.82] },
            ColorStop { position: 0.9 , color: [0.28, 0.33, 0.45] },
            ColorStop { position: 1.0 , color: [0.05, 0.07, 0.16] },
        ])
    }

    /// Preset 8: Stained Glass — muted jewel tones (cyclic)
    fn stained_glass() -> Palette {
        Palette::new("Stained Glass", vec![
            ColorStop { position: 0.0 , color: [0.62, 0.11, 0.16] },
            ColorStop { position: 0.2 , color: [0.89, 0.6, 0.16] },
            ColorStop { position: 0.4 , color: [0.18, 0.5, 0.34] },
            ColorStop { position: 0.6 , color: [0.13, 0.31, 0.62] },
            ColorStop { position: 0.8 , color: [0.45, 0.22, 0.55] },
            ColorStop { position: 1.0 , color: [0.62, 0.11, 0.16] },
        ])
    }

    /// Preset 9: Sakura — plum, rose, blossom pink and sage (cyclic)
    fn sakura() -> Palette {
        Palette::new("Sakura", vec![
            ColorStop { position: 0.0 , color: [0.36, 0.16, 0.27] },
            ColorStop { position: 0.25, color: [0.8, 0.42, 0.55] },
            ColorStop { position: 0.5 , color: [0.98, 0.76, 0.8] },
            ColorStop { position: 0.7 , color: [1.0, 0.95, 0.9] },
            ColorStop { position: 0.85, color: [0.7, 0.78, 0.62] },
            ColorStop { position: 1.0 , color: [0.36, 0.16, 0.27] },
        ])
    }

    /// Preset 10: Twilight — indigo dusk into rose and peach (cyclic)
    fn twilight() -> Palette {
        Palette::new("Twilight", vec![
            ColorStop { position: 0.0 , color: [0.08, 0.07, 0.25] },
            ColorStop { position: 0.25, color: [0.36, 0.2, 0.56] },
            ColorStop { position: 0.5 , color: [0.85, 0.36, 0.52] },
            ColorStop { position: 0.7 , color: [0.99, 0.69, 0.45] },
            ColorStop { position: 0.85, color: [0.99, 0.9, 0.7] },
            ColorStop { position: 1.0 , color: [0.08, 0.07, 0.25] },
        ])
    }

    /// Preset 11: Jade — deep teal to pale mint with a gold accent (cyclic)
    fn jade() -> Palette {
        Palette::new("Jade", vec![
            ColorStop { position: 0.0 , color: [0.02, 0.16, 0.17] },
            ColorStop { position: 0.3 , color: [0.05, 0.42, 0.38] },
            ColorStop { position: 0.55, color: [0.36, 0.72, 0.58] },
            ColorStop { position: 0.75, color: [0.82, 0.93, 0.8] },
            ColorStop { position: 0.9 , color: [0.85, 0.7, 0.35] },
            ColorStop { position: 1.0 , color: [0.02, 0.16, 0.17] },
        ])
    }

    /// Preset 12: Ink Wash — paper white to ink blue
    fn ink_wash() -> Palette {
        Palette::new("Ink Wash", vec![
            ColorStop { position: 0.0 , color: [0.95, 0.92, 0.84] },
            ColorStop { position: 0.3 , color: [0.66, 0.7, 0.72] },
            ColorStop { position: 0.6 , color: [0.3, 0.38, 0.47] },
            ColorStop { position: 0.85, color: [0.16, 0.22, 0.32] },
            ColorStop { position: 1.0 , color: [0.95, 0.92, 0.84] },
        ])
    }
}
