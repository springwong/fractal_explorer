//! WGSL shader assembly.
//!
//! WGSL has no `#include`, so every compute shader is put together here from shared
//! modules plus one fractal-specific body. The uniform block, the palette lookup, the
//! pixel -> plane mapping and the double-single arithmetic each exist in exactly one
//! file under `src/shaders/`.

const COMMON: &str = include_str!("../shaders/common.wgsl");
const BINDINGS_STANDARD: &str = include_str!("../shaders/bindings_standard.wgsl");
const BINDINGS_ORBIT: &str = include_str!("../shaders/bindings_orbit.wgsl");
const PALETTE: &str = include_str!("../shaders/palette.wgsl");
const VIEW: &str = include_str!("../shaders/view.wgsl");
const DOUBLE_SINGLE: &str = include_str!("../shaders/double_single.wgsl");

/// Which shared modules (and which bind group layout) a fractal shader body needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderKind {
    /// Plain f32 maths. Bindings: uniforms, output texture, palette.
    Standard,
    /// Double-single maths on the GPU. Same bindings as `Standard`.
    EmulatedDouble,
    /// Perturbation around a reference orbit computed on the CPU in f64.
    /// Bindings: uniforms, output texture, reference orbit, palette.
    Perturbation,
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

    /// Shader using emulated double precision
    pub const fn emulated_double(body: &'static str) -> Self {
        Self { body, kind: ShaderKind::EmulatedDouble }
    }

    /// Shader using perturbation theory with a CPU reference orbit
    pub const fn perturbation(body: &'static str) -> Self {
        Self { body, kind: ShaderKind::Perturbation }
    }

    /// Whether the shader reads a reference orbit (and so needs the orbit bind group)
    pub fn uses_orbit(&self) -> bool {
        self.kind == ShaderKind::Perturbation
    }

    /// The complete WGSL source
    pub fn source(&self) -> String {
        let modules: &[&str] = match self.kind {
            ShaderKind::Standard => &[COMMON, BINDINGS_STANDARD, PALETTE, VIEW],
            ShaderKind::EmulatedDouble => &[COMMON, BINDINGS_STANDARD, PALETTE, VIEW, DOUBLE_SINGLE],
            ShaderKind::Perturbation => &[COMMON, BINDINGS_ORBIT, PALETTE, VIEW],
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
