// Perturbation bindings: output texture + reference orbit + palette.
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
// Reference orbit: flat array of f32 pairs [zx, zy] for iterations 0..=max_iter
@group(0) @binding(2) var<storage, read> ref_orbit: array<f32>;
@group(0) @binding(3) var<storage, read> palette_lut: array<u32>;
