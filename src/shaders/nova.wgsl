// Nova fractal: Newton's method with perturbation
// z → z - (z³-1)/(3z²) + c
struct Uniforms {
    center: vec2<f32>,
    zoom: f32,
    aspect_ratio: f32,
    max_iter: u32,
    fractal_type: u32,
    color_scheme: u32,
    c_real: f32,
    c_imag: f32,
    center_lo_x: f32,
    center_lo_y: f32,
    zoom_lo: f32,
    pixel_step_x: f32,
    pixel_step_y: f32,
    ref_escape_iter: u32,
    rotation: f32,
    _pad2: vec3<u32>,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> palette_lut: array<u32>;

/// Nova fractal iteration with convergence-based smooth coloring
fn nova(px: f32, py: f32, cx: f32, cy: f32) -> f32 {
    var zx: f32 = px;
    var zy: f32 = py;
    var iter: u32 = 0u;
    let tolerance: f32 = 1.0e-6;
    let bailout: f32 = 1.0e8;

    while (iter < uniforms.max_iter) {
        let zx2 = zx * zx;
        let zy2 = zy * zy;
        let mag2 = zx2 + zy2;

        // Bail out if z escapes
        if mag2 > bailout {
            break;
        }

        // Compute z³ = (zx + i*zy)³
        // z² = (zx² - zy²) + i*(2*zx*zy)
        // z³ = z² * z = (zx²-zy²)*zx - 2*zx*zy*zy + i*((zx²-zy²)*zy + 2*zx*zy*zx)
        //     = zx³ - 3*zx*zy² + i*(3*zx²*zy - zy³)
        let z3x = zx * zx2 - 3.0 * zx * zy2;
        let z3y = 3.0 * zx2 * zy - zy * zy2;

        // z³ - 1
        let num_x = z3x - 1.0;
        let num_y = z3y;

        // 3z² = 3*(zx² - zy²) + i*(6*zx*zy)
        let den_x = 3.0 * (zx2 - zy2);
        let den_y = 6.0 * zx * zy;

        // Complex division: (z³-1) / (3z²)
        let den_mag2 = den_x * den_x + den_y * den_y;
        if den_mag2 < 1.0e-20 {
            break;
        }
        let div_x = (num_x * den_x + num_y * den_y) / den_mag2;
        let div_y = (num_y * den_x - num_x * den_y) / den_mag2;

        // z_new = z - (z³-1)/(3z²) + c
        let new_zx = zx - div_x + cx;
        let new_zy = zy - div_y + cy;

        // Check convergence: |z_new - z| < tolerance
        let dx = new_zx - zx;
        let dy = new_zy - zy;
        let delta2 = dx * dx + dy * dy;

        zx = new_zx;
        zy = new_zy;
        iter += 1u;

        if delta2 < tolerance {
            // Smooth coloring based on convergence
            let smooth_val = f32(iter) - log2(log2(delta2 + 1.0e-20) / log2(tolerance)) ;
            return max(smooth_val, 0.0);
        }
    }

    // Did not converge — color as "inside" or based on escape
    if iter == uniforms.max_iter {
        return 0.0;
    }
    // Escaped: use iteration count
    return f32(iter);
}

fn sample_palette(t: f32) -> vec4<f32> {
    if t == 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let index = u32(fract(t * 0.05) * 255.0);
    let packed = palette_lut[index];
    let r = f32((packed >> 0u) & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;
    return vec4<f32>(r, g, b, 1.0);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);

    if id.x >= dims.x || id.y >= dims.y {
        return;
    }

    // Convert pixel coordinates to complex plane coordinates
    let uv = (vec2<f32>(id.xy) - vec2<f32>(dims) / 2.0) / f32(dims.y);
    let cos_r = cos(uniforms.rotation);
    let sin_r = sin(uniforms.rotation);
    let rotated = vec2<f32>(uv.x * cos_r - uv.y * sin_r, uv.x * sin_r + uv.y * cos_r);
    let z_init = uniforms.center + (rotated / uniforms.zoom) * vec2<f32>(1.0, -1.0);

    let smooth_val = nova(z_init.x, z_init.y, uniforms.c_real, uniforms.c_imag);

    let color = sample_palette(smooth_val);
    textureStore(output_texture, vec2<i32>(id.xy), color);
}
