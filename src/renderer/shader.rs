//! WGSL shader assembly.
//!
//! WGSL has no `#include`, so every compute shader is put together here from shared
//! modules plus one fractal-specific body. The uniform block, the palette lookup, the
//! pixel -> plane mapping, the rebasing perturbation loop and Newton's method each exist in exactly one
//! file under `src/shaders/`.

const COMMON: &str = include_str!("../shaders/common.wgsl");
const BINDINGS_STANDARD: &str = include_str!("../shaders/bindings_standard.wgsl");
const BINDINGS_ORBIT: &str = include_str!("../shaders/bindings_orbit.wgsl");
const PALETTE: &str = include_str!("../shaders/palette.wgsl");
const VIEW: &str = include_str!("../shaders/view.wgsl");
const REBASING: &str = include_str!("../shaders/rebasing.wgsl");
const LATTICE: &str = include_str!("../shaders/lattice.wgsl");
const NEWTON_CORE: &str = include_str!("../shaders/newton_core.wgsl");
const NEWTON_PERTURBATION: &str = include_str!("../shaders/newton_perturbation.wgsl");

/// Which shared modules (and which bind group layout) a fractal shader body needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderKind {
    /// Plain f32 maths. Bindings: uniforms, output texture, palette.
    Standard,
    /// Perturbation around a reference orbit computed on the CPU in f64; the body
    /// runs its own loop. Bindings: uniforms, output texture, reference orbit, palette.
    Perturbation,
    /// Perturbation using the shared rebasing loop (`rebasing.wgsl`); the body only
    /// supplies `delta_step`. Same bindings as `Perturbation`.
    RebasingPerturbation,
    /// Newton's method in f32 (`newton_core.wgsl`); the body only picks the colouring.
    /// Same bindings as `Standard`.
    Newton,
    /// Newton's method by perturbation (`newton_perturbation.wgsl`); the body only
    /// picks the colouring. Same bindings as `Perturbation`.
    NewtonPerturbation,
    /// Exact integer pixel coordinates (`lattice.wgsl`) for fractals read off the
    /// digits of a point. Same bindings as `Standard`.
    Lattice,
}

/// A per-pixel fractal compute shader: a fractal-specific body plus the shared
/// modules its kind pulls in.
#[derive(Clone, Copy, Debug)]
pub struct Shader {
    /// Fractal-specific WGSL: the iteration function(s) and `main`
    pub body: &'static str,
    /// Which shared modules and bindings the body is written against
    pub kind: ShaderKind,
}

impl Shader {
    /// Shader using plain f32 maths
    pub const fn standard(body: &'static str) -> Self {
        Self { body, kind: ShaderKind::Standard }
    }

    /// Shader using perturbation theory with a CPU reference orbit
    pub const fn perturbation(body: &'static str) -> Self {
        Self { body, kind: ShaderKind::Perturbation }
    }

    /// Shader supplying `delta_step` for the shared rebasing perturbation loop
    pub const fn rebasing_perturbation(body: &'static str) -> Self {
        Self { body, kind: ShaderKind::RebasingPerturbation }
    }

    /// Shader colouring the outcome of the shared f32 Newton iteration
    pub const fn newton(body: &'static str) -> Self {
        Self { body, kind: ShaderKind::Newton }
    }

    /// Shader colouring the outcome of the shared Newton perturbation loop
    pub const fn newton_perturbation(body: &'static str) -> Self {
        Self { body, kind: ShaderKind::NewtonPerturbation }
    }

    /// Shader working on the integer coordinates of the shared lattice module
    pub const fn lattice(body: &'static str) -> Self {
        Self { body, kind: ShaderKind::Lattice }
    }

    /// Whether the shader reads a reference orbit (and so needs the orbit bind group)
    pub fn uses_orbit(&self) -> bool {
        matches!(
            self.kind,
            ShaderKind::Perturbation | ShaderKind::RebasingPerturbation | ShaderKind::NewtonPerturbation
        )
    }

    /// The complete WGSL source
    pub fn source(&self) -> String {
        let modules: &[&str] = match self.kind {
            ShaderKind::Standard => &[COMMON, BINDINGS_STANDARD, PALETTE, VIEW],
            ShaderKind::Perturbation => &[COMMON, BINDINGS_ORBIT, PALETTE, VIEW],
            ShaderKind::RebasingPerturbation => &[COMMON, BINDINGS_ORBIT, PALETTE, VIEW, REBASING],
            ShaderKind::Newton => &[COMMON, BINDINGS_STANDARD, PALETTE, VIEW, NEWTON_CORE],
            ShaderKind::Lattice => &[COMMON, BINDINGS_STANDARD, PALETTE, VIEW, LATTICE],
            ShaderKind::NewtonPerturbation => {
                &[COMMON, BINDINGS_ORBIT, PALETTE, VIEW, NEWTON_CORE, NEWTON_PERTURBATION]
            }
        };
        assemble(modules, self.body)
    }
}

/// Buddhabrot accumulation pass (declares its own bindings)
pub fn buddhabrot_accum_source() -> String {
    assemble(&[COMMON], include_str!("../shaders/buddhabrot.wgsl"))
}

/// Buddhabrot tonemap pass (declares its own bindings)
pub fn buddhabrot_tonemap_source() -> String {
    assemble(&[COMMON, PALETTE], include_str!("../shaders/buddhabrot_tonemap.wgsl"))
}

/// The shared uniform block on its own (for layout checks)
#[cfg(test)]
pub fn common_source() -> &'static str {
    COMMON
}

fn assemble(modules: &[&str], body: &str) -> String {
    let mut source = String::with_capacity(modules.iter().map(|m| m.len() + 1).sum::<usize>() + body.len());
    for module in modules {
        source.push_str(module);
        source.push('\n');
    }
    source.push_str(body);
    source
}
