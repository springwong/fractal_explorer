// Shared by every compute shader: the uniform block.
//
// This is the ONLY WGSL copy of the layout. It must match `FractalUniforms` in
// src/renderer/uniforms.rs (80 bytes); a unit test there compares the two.
struct Uniforms {
    center: vec2<f32>,       // offset 0   view centre (high part)
    zoom: f32,               // offset 8   zoom (high part)
    aspect_ratio: f32,       // offset 12
    max_iter: u32,           // offset 16
    fractal_type: u32,       // offset 20
    color_scheme: u32,       // offset 24  (gasket: coarse cell codes)
    c_real: f32,             // offset 28  Julia / Nova c (gasket: reference x)
    c_imag: f32,             // offset 32  Julia / Nova c (gasket: reference y)
    center_lo_x: f32,        // offset 36  low part of center.x (emulated double)
    center_lo_y: f32,        // offset 40  low part of center.y
    zoom_lo: f32,            // offset 44  low part of zoom
    pixel_step_x: f32,       // offset 48  1 / (zoom * height), from the CPU in f64
    pixel_step_y: f32,       // offset 52  -1 / (zoom * height)
    ref_escape_iter: u32,    // offset 56  where the reference orbit escapes (Buddhabrot: frame seed)
    rotation: f32,           // offset 60  view rotation, radians
    extra: vec3<u32>,        // offset 64  per-fractal data (Buddhabrot: width, height, batches)
    _pad3: u32,              // offset 76
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
