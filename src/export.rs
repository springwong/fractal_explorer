use crate::fractals::FractalType;
use crate::renderer::{FractalUniforms, compute_reference_orbit, compute_reference_orbit_julia};
use std::path::Path;

/// Export resolution presets
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExportResolution {
    HD1080p,
    QHD1440p,
    UHD4K,
    UHD8K,
    Custom(u32, u32),
}

impl ExportResolution {
    /// Get width and height in pixels
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            ExportResolution::HD1080p => (1920, 1080),
            ExportResolution::QHD1440p => (2560, 1440),
            ExportResolution::UHD4K => (3840, 2160),
            ExportResolution::UHD8K => (7680, 4320),
            ExportResolution::Custom(w, h) => (*w, *h),
        }
    }

    /// Get a label for filenames
    pub fn label(&self) -> &'static str {
        match self {
            ExportResolution::HD1080p => "1080p",
            ExportResolution::QHD1440p => "1440p",
            ExportResolution::UHD4K => "4k",
            ExportResolution::UHD8K => "8k",
            ExportResolution::Custom(_, _) => "custom",
        }
    }
}

/// Export the current fractal view as a PNG file
pub fn export_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fractal_type: &FractalType,
    uniforms: &FractalUniforms,
    resolution: ExportResolution,
    output_path: &Path,
) -> Result<(), String> {
    let (width, height) = resolution.dimensions();
    let aspect_ratio = width as f32 / height as f32;

    log::info!(
        "Exporting {}x{} PNG to {:?}...",
        width,
        height,
        output_path
    );

    // Recompute pixel_step for export resolution (height differs from screen)
    let zoom_f64 = uniforms.zoom as f64 + uniforms.zoom_lo as f64;
    let export_pixel_step_x = (1.0 / (zoom_f64 * height as f64)) as f32;
    let export_pixel_step_y = (-1.0 / (zoom_f64 * height as f64)) as f32;

    // Create export uniforms with correct aspect ratio for target resolution
    let export_uniforms = FractalUniforms::new(
        uniforms.center,
        uniforms.zoom,
        aspect_ratio,
        uniforms.max_iter,
        uniforms.fractal_type,
        uniforms.color_scheme,
        uniforms.c_real,
        uniforms.c_imag,
        [uniforms.center_lo_x, uniforms.center_lo_y],
        uniforms.zoom_lo,
        export_pixel_step_x,
        export_pixel_step_y,
        uniforms.ref_escape_iter,
        uniforms.rotation,
    );

    // Create uniform buffer
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Export Uniform Buffer"),
        size: std::mem::size_of::<FractalUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&export_uniforms));

    // Create storage texture at target resolution
    let storage_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Export Storage Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });

    let texture_view = storage_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Determine precision mode
    let use_f64 = zoom_f64 >= 5.0e3;
    let ftype = fractal_type.type_id();
    let use_perturbation = use_f64 && (ftype == 0 || ftype == 1);

    // Create orbit buffer for perturbation (Mandelbrot & Julia f64)
    let orbit_buffer = if use_perturbation {
        let center_x = uniforms.center[0] as f64 + uniforms.center_lo_x as f64;
        let center_y = uniforms.center[1] as f64 + uniforms.center_lo_y as f64;
        let orbit_data = if ftype == 0 {
            compute_reference_orbit(center_x, center_y, uniforms.max_iter).0
        } else {
            compute_reference_orbit_julia(
                center_x, center_y,
                uniforms.c_real as f64, uniforms.c_imag as f64,
                uniforms.max_iter,
            ).0
        };
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Export Orbit Buffer"),
            size: (orbit_data.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&orbit_data));
        Some(buf)
    } else {
        None
    };

    // Build bind group layout entries
    let mut layout_entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
    ];
    if use_perturbation {
        layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Export Bind Group Layout"),
        entries: &layout_entries,
    });

    // Build bind group entries
    let mut bg_entries = vec![
        wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
            binding: 1,
            resource: wgpu::BindingResource::TextureView(&texture_view),
        },
    ];
    if let Some(ref ob) = orbit_buffer {
        bg_entries.push(wgpu::BindGroupEntry {
            binding: 2,
            resource: ob.as_entire_binding(),
        });
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Export Bind Group"),
        layout: &bind_group_layout,
        entries: &bg_entries,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Export Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    // Select shader
    let shader_source = if use_f64 {
        fractal_type.shader_source_f64()
    } else {
        fractal_type.shader_source()
    };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Export Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Export Compute Pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // Row bytes must be aligned to 256 for buffer copy
    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
    let buffer_size = (padded_bytes_per_row * height) as u64;

    // Create output buffer for readback
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Export Output Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Create command encoder
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Export Command Encoder"),
    });

    // Dispatch compute shader
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Export Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);

        let workgroup_count_x = (width + 15) / 16;
        let workgroup_count_y = (height + 15) / 16;
        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
    }

    // Copy texture to buffer
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &storage_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    // Submit and wait
    queue.submit(std::iter::once(encoder.finish()));

    // Map the buffer and read pixels
    let buffer_slice = output_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    device.poll(wgpu::Maintain::Wait);
    receiver
        .recv()
        .map_err(|e| format!("Failed to receive map result: {}", e))?
        .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

    // Read pixels and remove row padding
    let data = buffer_slice.get_mapped_range();
    let mut pixels = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
    for row in 0..height {
        let start = (row * padded_bytes_per_row) as usize;
        let end = start + unpadded_bytes_per_row as usize;
        pixels.extend_from_slice(&data[start..end]);
    }
    drop(data);
    output_buffer.unmap();

    // Save as PNG
    let img: image::RgbaImage =
        image::ImageBuffer::from_raw(width, height, pixels)
            .ok_or_else(|| "Failed to create image from pixel data".to_string())?;

    img.save(output_path)
        .map_err(|e| format!("Failed to save PNG: {}", e))?;

    log::info!("Exported PNG: {:?} ({}x{})", output_path, width, height);
    Ok(())
}

/// Generate a timestamped filename for export
pub fn generate_filename(fractal_name: &str, resolution: &ExportResolution) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    // Simple timestamp: YYYYMMDD_HHMMSS approximation using unix time
    format!(
        "fractal_{}_{}_{}s.png",
        fractal_name.to_lowercase().replace(' ', "_"),
        resolution.label(),
        secs
    )
}
