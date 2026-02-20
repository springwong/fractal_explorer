/// Common coloring functions for all fractals
/// This file contains reusable colorization schemes

/// Smooth rainbow colorization (default)
fn colorize_smooth(t: f32) -> vec4<f32> {
    if t == 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0); // Inside set = black
    }

    // Map iteration count to hue
    let hue = fract(t * 0.05);
    let sat = 0.8;
    let val = 0.9;

    // HSV to RGB conversion
    let h = hue * 6.0;
    let i = floor(h);
    let f = h - i;
    let p = val * (1.0 - sat);
    let q = val * (1.0 - sat * f);
    let t_val = val * (1.0 - sat * (1.0 - f));

    var rgb: vec3<f32>;
    if i == 0.0 {
        rgb = vec3<f32>(val, t_val, p);
    } else if i == 1.0 {
        rgb = vec3<f32>(q, val, p);
    } else if i == 2.0 {
        rgb = vec3<f32>(p, val, t_val);
    } else if i == 3.0 {
        rgb = vec3<f32>(p, q, val);
    } else if i == 4.0 {
        rgb = vec3<f32>(t_val, p, val);
    } else {
        rgb = vec3<f32>(val, p, q);
    }

    return vec4<f32>(rgb, 1.0);
}

/// Fire colorization (red, orange, yellow, white)
fn colorize_fire(t: f32) -> vec4<f32> {
    if t == 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let normalized = fract(t * 0.03);
    let r = min(1.0, normalized * 2.0);
    let g = max(0.0, min(1.0, (normalized - 0.3) * 2.5));
    let b = max(0.0, min(1.0, (normalized - 0.7) * 3.3));

    return vec4<f32>(r, g, b, 1.0);
}

/// Ocean colorization (blue, cyan, white)
fn colorize_ocean(t: f32) -> vec4<f32> {
    if t == 0.0 {
        return vec4<f32>(0.0, 0.0, 0.1, 1.0); // Deep blue for inside
    }

    let normalized = fract(t * 0.04);
    let r = max(0.0, min(1.0, (normalized - 0.6) * 2.5));
    let g = max(0.0, min(1.0, (normalized - 0.2) * 1.8));
    let b = min(1.0, 0.3 + normalized * 0.7);

    return vec4<f32>(r, g, b, 1.0);
}

/// Grayscale colorization
fn colorize_grayscale(t: f32) -> vec4<f32> {
    if t == 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let intensity = fract(t * 0.05);
    return vec4<f32>(intensity, intensity, intensity, 1.0);
}

/// Main colorization dispatcher
fn colorize(t: f32, scheme: u32) -> vec4<f32> {
    switch scheme {
        case 0u: { return colorize_smooth(t); }
        case 1u: { return colorize_fire(t); }
        case 2u: { return colorize_ocean(t); }
        case 3u: { return colorize_grayscale(t); }
        default: { return colorize_smooth(t); }
    }
}
