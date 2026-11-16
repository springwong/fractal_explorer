//! The fractal catalogue.
//!
//! Everything the rest of the program needs to know about a fractal lives in its
//! `FractalInfo`. To add a fractal: add a `FractalType` variant, its `FractalInfo`
//! (one arm in `FractalType::info`), an entry in `FractalType::ALL`, and the WGSL body
//! under `src/shaders/`. The control panel, keyboard shortcuts, render passes and
//! export all work from this table.

mod golden_silver;

use crate::camera::ViewBounds;
use crate::renderer::{reference_orbit, FractalUniforms, Shader};
use glam::{DVec2, Vec2};

/// Pixel density (pixels per plane unit = zoom x output height) from which the
/// deep-zoom shader of a fractal takes over.
///
/// What plain f32 runs out of is room between neighbouring pixels: near |c| = 2 its
/// values are 2.4e-7 apart, so pixels closer than that share coordinates and the
/// picture turns blocky. The limit therefore depends on the output height as well as
/// the zoom: about zoom 3.7e3 on a 1080p view, but only 9e2 for an 8K export.
pub const DEEP_ZOOM_PIXEL_DENSITY: f64 = 4.0e6;

/// Highest iteration count the UI offers; GPU orbit buffers are sized for it.
pub const MAX_ITERATIONS: u32 = 4096;

/// Lowest iteration count the UI offers
pub const MIN_ITERATIONS: u32 = 64;

// Deepest useful zoom: the camera centre is an f64, which perturbation can use fully
const PERTURBATION_ZOOM: f64 = 1e13;

/// How a fractal gets onto the screen
#[derive(Debug)]
pub enum RenderMode {
    /// One compute pass per frame, one invocation per pixel
    PerPixel {
        /// Shader for ordinary zoom levels
        shader: Shader,
        /// Shader from `DEEP_ZOOM_PIXEL_DENSITY` on; `None` if `shader` copes at any zoom
        deep_shader: Option<Shader>,
    },
    /// Progressive Monte Carlo accumulation with its own passes (Buddhabrot)
    Accumulation,
}

/// Static description of one fractal
#[derive(Debug)]
pub struct FractalInfo {
    /// Stable id: index in `FractalType::ALL`, also sent to the GPU
    pub id: u32,
    /// Human-readable name
    pub name: &'static str,
    /// View centre the fractal starts from
    pub default_center: DVec2,
    /// Zoom the fractal starts from
    pub default_zoom: f64,
    /// Where the camera may go: the centre stays inside a box around the interesting
    /// region and zoom stays within what can be shown
    pub bounds: ViewBounds,
    /// How it is rendered
    pub render: RenderMode,
}

static MANDELBROT: FractalInfo = FractalInfo {
    id: 0,
    name: "Mandelbrot Set",
    default_center: DVec2::new(-0.5, 0.0),
    default_zoom: 0.5,
    bounds: ViewBounds::new(-2.5, -1.5, 1.0, 1.5, 0.2, PERTURBATION_ZOOM),
    render: RenderMode::PerPixel {
        shader: Shader::standard(include_str!("../shaders/mandelbrot.wgsl")),
        deep_shader: Some(Shader::rebasing_perturbation(include_str!("../shaders/mandelbrot_f64.wgsl"))),
    },
};

static JULIA: FractalInfo = FractalInfo {
    id: 1,
    name: "Julia Set",
    default_center: DVec2::new(0.0, 0.0),
    default_zoom: 0.5,
    bounds: ViewBounds::new(-2.0, -2.0, 2.0, 2.0, 0.2, PERTURBATION_ZOOM),
    render: RenderMode::PerPixel {
        shader: Shader::standard(include_str!("../shaders/julia.wgsl")),
        deep_shader: Some(Shader::perturbation(include_str!("../shaders/julia_f64.wgsl"))),
    },
};

static BURNING_SHIP: FractalInfo = FractalInfo {
    id: 2,
    name: "Burning Ship",
    // Classic "burning ship" view at the bottom of the main body, zoomed in to show the ship
    default_center: DVec2::new(-1.755, -0.035),
    default_zoom: 150.0,
    bounds: ViewBounds::new(-2.5, -2.0, 1.5, 1.0, 0.2, PERTURBATION_ZOOM),
    render: RenderMode::PerPixel {
        shader: Shader::standard(include_str!("../shaders/burning_ship.wgsl")),
        deep_shader: Some(Shader::rebasing_perturbation(include_str!("../shaders/burning_ship_f64.wgsl"))),
    },
};

static TRICORN: FractalInfo = FractalInfo {
    id: 3,
    name: "Tricorn",
    default_center: DVec2::new(-0.5, 0.0),
    default_zoom: 0.5,
    bounds: ViewBounds::new(-2.5, -2.0, 1.5, 2.0, 0.2, PERTURBATION_ZOOM),
    render: RenderMode::PerPixel {
        shader: Shader::standard(include_str!("../shaders/tricorn.wgsl")),
        deep_shader: Some(Shader::rebasing_perturbation(include_str!("../shaders/tricorn_f64.wgsl"))),
    },
};

static BUDDHABROT: FractalInfo = FractalInfo {
    id: 4,
    name: "Buddhabrot",
    default_center: DVec2::new(-0.5, 0.0),
    default_zoom: 0.5,
    // Samples thin out quadratically with zoom, so deep views stay empty
    bounds: ViewBounds::new(-2.5, -1.5, 1.0, 1.5, 0.2, 200.0),
    render: RenderMode::Accumulation,
};

static NOVA: FractalInfo = FractalInfo {
    id: 5,
    name: "Nova Fractal",
    default_center: DVec2::new(0.0, 0.0),
    default_zoom: 0.5,
    bounds: ViewBounds::new(-3.0, -3.0, 3.0, 3.0, 0.2, PERTURBATION_ZOOM),
    render: RenderMode::PerPixel {
        shader: Shader::standard(include_str!("../shaders/nova.wgsl")),
        deep_shader: Some(Shader::newton_perturbation(include_str!("../shaders/nova_f64.wgsl"))),
    },
};

static GOLDEN_SILVER: FractalInfo = FractalInfo {
    id: 6,
    name: "Golden-Silver Gasket",
    default_center: DVec2::new(0.5, 0.5),
    default_zoom: 0.9,
    // The pattern is drawn out to |x|, |y| < 199 and is black beyond. Let the view
    // roam all of it, with at most a thin black strip showing past the edge.
    // Located in f64 on the CPU (see golden_silver::apply_view); out at the edge
    // f64 leaves about this much zoom
    bounds: ViewBounds::new(-199.0, -199.0, 199.0, 199.0, 0.0022, 1e10).with_edge_margin(0.08),
    render: RenderMode::PerPixel {
        shader: Shader::standard(include_str!("../shaders/golden_silver.wgsl")),
        deep_shader: None,
    },
};

static MULTIBROT: FractalInfo = FractalInfo {
    id: 7,
    name: "Multibrot",
    default_center: DVec2::new(0.0, 0.0),
    default_zoom: 0.6,
    bounds: ViewBounds::new(-2.0, -2.0, 2.0, 2.0, 0.2, PERTURBATION_ZOOM),
    render: RenderMode::PerPixel {
        shader: Shader::standard(include_str!("../shaders/multibrot.wgsl")),
        deep_shader: Some(Shader::rebasing_perturbation(include_str!("../shaders/multibrot_f64.wgsl"))),
    },
};

static CELTIC: FractalInfo = FractalInfo {
    id: 8,
    name: "Celtic",
    default_center: DVec2::new(-0.5, 0.0),
    default_zoom: 0.5,
    bounds: ViewBounds::new(-3.0, -2.0, 2.0, 2.0, 0.2, PERTURBATION_ZOOM),
    render: RenderMode::PerPixel {
        shader: Shader::standard(include_str!("../shaders/celtic.wgsl")),
        deep_shader: Some(Shader::rebasing_perturbation(include_str!("../shaders/celtic_f64.wgsl"))),
    },
};

static BUFFALO: FractalInfo = FractalInfo {
    id: 9,
    name: "Buffalo",
    // The body sits above the real axis
    default_center: DVec2::new(-0.4, 0.5),
    default_zoom: 0.4,
    bounds: ViewBounds::new(-3.0, -2.5, 2.0, 2.5, 0.2, PERTURBATION_ZOOM),
    render: RenderMode::PerPixel {
        shader: Shader::standard(include_str!("../shaders/buffalo.wgsl")),
        deep_shader: Some(Shader::rebasing_perturbation(include_str!("../shaders/buffalo_f64.wgsl"))),
    },
};

static PERPENDICULAR: FractalInfo = FractalInfo {
    id: 10,
    name: "Perpendicular",
    default_center: DVec2::new(-0.5, 0.0),
    default_zoom: 0.5,
    bounds: ViewBounds::new(-3.0, -2.0, 2.0, 2.0, 0.2, PERTURBATION_ZOOM),
    render: RenderMode::PerPixel {
        shader: Shader::standard(include_str!("../shaders/perpendicular.wgsl")),
        deep_shader: Some(Shader::rebasing_perturbation(include_str!("../shaders/perpendicular_f64.wgsl"))),
    },
};

static NEWTON: FractalInfo = FractalInfo {
    id: 11,
    name: "Newton",
    default_center: DVec2::new(0.0, 0.0),
    default_zoom: 0.5,
    bounds: ViewBounds::new(-3.0, -3.0, 3.0, 3.0, 0.2, PERTURBATION_ZOOM),
    render: RenderMode::PerPixel {
        shader: Shader::newton(include_str!("../shaders/newton.wgsl")),
        deep_shader: Some(Shader::newton_perturbation(include_str!("../shaders/newton_f64.wgsl"))),
    },
};

/// Enum for all supported fractal types, with their live parameters
#[derive(Clone, PartialEq, Debug)]
pub enum FractalType {
    Mandelbrot,
    Julia { c: Vec2 },
    BurningShip,
    Tricorn,
    Buddhabrot,
    Nova { c: Vec2 },
    GoldenSilver,
    /// z³ + c
    Multibrot,
    Celtic,
    Buffalo,
    /// Perpendicular Mandelbrot
    Perpendicular,
    /// Newton's method for z³ - 1, coloured by root
    Newton,
}

impl FractalType {
    /// Every fractal with its default parameters, in menu order.
    /// The first nine are selected with keyboard keys 1-9; the fractal picker gives
    /// every entry a letter.
    pub const ALL: [FractalType; 12] = [
        FractalType::Mandelbrot,
        FractalType::Julia { c: Vec2::new(-0.7, 0.27015) },
        FractalType::BurningShip,
        FractalType::Tricorn,
        FractalType::Buddhabrot,
        FractalType::Nova { c: Vec2::new(1.0, 0.0) },
        FractalType::GoldenSilver,
        FractalType::Multibrot,
        FractalType::Celtic,
        FractalType::Buffalo,
        FractalType::Perpendicular,
        FractalType::Newton,
    ];

    /// The static description of this fractal
    pub fn info(&self) -> &'static FractalInfo {
        match self {
            FractalType::Mandelbrot => &MANDELBROT,
            FractalType::Julia { .. } => &JULIA,
            FractalType::BurningShip => &BURNING_SHIP,
            FractalType::Tricorn => &TRICORN,
            FractalType::Buddhabrot => &BUDDHABROT,
            FractalType::Nova { .. } => &NOVA,
            FractalType::GoldenSilver => &GOLDEN_SILVER,
            FractalType::Multibrot => &MULTIBROT,
            FractalType::Celtic => &CELTIC,
            FractalType::Buffalo => &BUFFALO,
            FractalType::Perpendicular => &PERPENDICULAR,
            FractalType::Newton => &NEWTON,
        }
    }

    /// Stable id of the fractal (ignores parameters)
    pub fn id(&self) -> u32 {
        self.info().id
    }

    /// Human-readable name
    pub fn name(&self) -> &'static str {
        self.info().name
    }

    /// The default center for this fractal
    pub fn default_center(&self) -> DVec2 {
        self.info().default_center
    }

    /// The default zoom for this fractal
    pub fn default_zoom(&self) -> f64 {
        self.info().default_zoom
    }

    /// Where the camera may go for this fractal
    pub fn view_bounds(&self) -> ViewBounds {
        self.info().bounds
    }

    /// The complex parameter c of the fractals that have one (zero otherwise)
    pub fn c(&self) -> Vec2 {
        match self {
            FractalType::Julia { c } | FractalType::Nova { c } => *c,
            _ => Vec2::ZERO,
        }
    }

    /// Mutable access to the parameter c, for the fractals that have one
    pub fn c_mut(&mut self) -> Option<&mut Vec2> {
        match self {
            FractalType::Julia { c } | FractalType::Nova { c } => Some(c),
            _ => None,
        }
    }

    /// Whether this fractal type uses accumulation buffer rendering
    pub fn is_buddhabrot(&self) -> bool {
        matches!(self.info().render, RenderMode::Accumulation)
    }

    /// The per-pixel shader to use at `zoom` for an output `height` pixels high, or
    /// `None` for fractals that are not rendered per pixel.
    pub fn shader(&self, zoom: f64, height: u32) -> Option<&'static Shader> {
        match &self.info().render {
            RenderMode::PerPixel { shader, deep_shader } => Some(match deep_shader {
                Some(deep) if is_deep_zoom(zoom, height) => deep,
                _ => shader,
            }),
            RenderMode::Accumulation => None,
        }
    }

    /// Reference orbit for the perturbation shaders, around the view centre.
    /// Returns the orbit as flat [zx, zy] f32 pairs and the index of its last valid
    /// entry; `None` for fractals without a perturbation shader.
    ///
    /// Each map here must be the f64 twin of the fractal's WGSL iteration.
    pub fn reference_orbit(&self, center: DVec2, max_iter: u32) -> Option<(Vec<f32>, u32)> {
        // Well past the shaders' bailout, so a pixel still running when the reference
        // ends is far from it
        const ESCAPE_RADIUS2: f64 = 1e10;
        let c = self.c().as_dvec2();
        Some(match self {
            // z -> z² + c with z_0 = 0, c = centre
            FractalType::Mandelbrot => reference_orbit(DVec2::ZERO, max_iter, ESCAPE_RADIUS2, |z| {
                Some(DVec2::new(z.x * z.x - z.y * z.y + center.x, 2.0 * z.x * z.y + center.y))
            }),
            // z -> (|x| + i|y|)² + c with z_0 = 0, c = centre
            FractalType::BurningShip => reference_orbit(DVec2::ZERO, max_iter, ESCAPE_RADIUS2, |z| {
                Some(DVec2::new(z.x * z.x - z.y * z.y + center.x, 2.0 * (z.x * z.y).abs() + center.y))
            }),
            // z -> conj(z)² + c with z_0 = 0, c = centre
            FractalType::Tricorn => reference_orbit(DVec2::ZERO, max_iter, ESCAPE_RADIUS2, |z| {
                Some(DVec2::new(z.x * z.x - z.y * z.y + center.x, -2.0 * z.x * z.y + center.y))
            }),
            // z -> z³ + c with z_0 = 0, c = centre
            FractalType::Multibrot => reference_orbit(DVec2::ZERO, max_iter, ESCAPE_RADIUS2, |z| {
                let (x2, y2) = (z.x * z.x, z.y * z.y);
                Some(DVec2::new(z.x * (x2 - 3.0 * y2) + center.x, z.y * (3.0 * x2 - y2) + center.y))
            }),
            // z -> |Re(z²)| + i Im(z²) + c with z_0 = 0, c = centre
            FractalType::Celtic => reference_orbit(DVec2::ZERO, max_iter, ESCAPE_RADIUS2, |z| {
                Some(DVec2::new((z.x * z.x - z.y * z.y).abs() + center.x, 2.0 * z.x * z.y + center.y))
            }),
            // z -> |Re(z²)| - i|Im(z²)| + c with z_0 = 0, c = centre
            FractalType::Buffalo => reference_orbit(DVec2::ZERO, max_iter, ESCAPE_RADIUS2, |z| {
                Some(DVec2::new((z.x * z.x - z.y * z.y).abs() + center.x, -2.0 * (z.x * z.y).abs() + center.y))
            }),
            // z -> (|x| - iy)² + c with z_0 = 0, c = centre
            FractalType::Perpendicular => reference_orbit(DVec2::ZERO, max_iter, ESCAPE_RADIUS2, |z| {
                Some(DVec2::new(z.x * z.x - z.y * z.y + center.x, -2.0 * z.x.abs() * z.y + center.y))
            }),
            // z -> z² + c with z_0 = centre, c = the fixed parameter
            FractalType::Julia { .. } => reference_orbit(center, max_iter, ESCAPE_RADIUS2, |z| {
                Some(DVec2::new(z.x * z.x - z.y * z.y + c.x, 2.0 * z.x * z.y + c.y))
            }),
            // z -> z - (z³ - 1) / (3z²) + c = 2z/3 + 1/(3z²) + c with z_0 = centre.
            // Undefined at the pole z = 0; same limits as the shader. Newton is c = 0.
            FractalType::Nova { .. } | FractalType::Newton => reference_orbit(center, max_iter, 1e8, |z| {
                let z2 = DVec2::new(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
                let m = z2.length_squared();
                (9.0 * m >= 1e-20).then(|| z * (2.0 / 3.0) + DVec2::new(z2.x, -z2.y) / (3.0 * m) + c)
            }),
            FractalType::Buddhabrot | FractalType::GoldenSilver => return None,
        })
    }

    /// Fill any fractal-specific uniform slots for the view the uniforms describe,
    /// rendered at `width` x `height`. `FractalUniforms::for_view` calls this.
    pub fn prepare_uniforms(&self, uniforms: &mut FractalUniforms, width: u32, height: u32) {
        if let FractalType::GoldenSilver = self {
            golden_silver::apply_view(uniforms, width, height);
        }
    }
}

/// Whether a view at `zoom`, rendered `height` pixels high, needs the deep-zoom shaders
pub fn is_deep_zoom(zoom: f64, height: u32) -> bool {
    zoom * height as f64 >= DEEP_ZOOM_PIXEL_DENSITY
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::renderer::shader;
    use wgpu::naga;

    fn validate(name: &str, source: &str) {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{name}: parse failed: {}", e.emit_to_string(source)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: validation failed: {e:?}"));
    }

    /// Ids are the positions in `ALL`, so the list and the table cannot drift apart.
    #[test]
    fn ids_match_menu_order() {
        for (i, f) in FractalType::ALL.iter().enumerate() {
            assert_eq!(f.id() as usize, i, "{}", f.name());
        }
    }

    /// Each fractal's default view must sit inside its own bounds.
    #[test]
    fn default_views_are_inside_bounds() {
        for f in &FractalType::ALL {
            let b = f.view_bounds();
            let (c, z) = (f.default_center(), f.default_zoom());
            assert!(c.cmpge(b.min).all() && c.cmple(b.max).all(), "{}: centre", f.name());
            assert!(z >= b.min_zoom && z <= b.max_zoom, "{}: zoom", f.name());
        }
    }

    /// Every assembled shader must parse and validate.
    #[test]
    fn all_shaders_are_valid_wgsl() {
        for f in &FractalType::ALL {
            if let RenderMode::PerPixel { shader, deep_shader } = &f.info().render {
                validate(f.name(), &shader.source());
                if let Some(deep) = deep_shader {
                    validate(&format!("{} (deep)", f.name()), &deep.source());
                }
            }
        }
        validate("buddhabrot", &shader::buddhabrot_accum_source());
        validate("buddhabrot_tonemap", &shader::buddhabrot_tonemap_source());
        validate("fullscreen", include_str!("../shaders/fullscreen.wgsl"));
    }

    /// The switch to the deep-zoom shader follows pixel density: a taller output
    /// needs it at a lower zoom.
    #[test]
    fn deep_zoom_switch_depends_on_output_height() {
        assert!(!is_deep_zoom(3.0e3, 1080));
        assert!(is_deep_zoom(4.0e3, 1080));
        // 8K export of a view that is still fine in f32 on a 1080p screen
        assert!(is_deep_zoom(3.0e3, 4320));
        assert!(!is_deep_zoom(8.0e2, 4320));
    }

    /// Only fractals that can produce a reference orbit may use a perturbation shader.
    #[test]
    fn perturbation_shaders_have_an_orbit() {
        for f in &FractalType::ALL {
            let uses_orbit = f.shader(1.0e9, 1080).is_some_and(|s| s.uses_orbit());
            assert_eq!(uses_orbit, f.reference_orbit(DVec2::ZERO, 16).is_some(), "{}", f.name());
        }
    }
}
