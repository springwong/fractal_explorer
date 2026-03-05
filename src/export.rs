use crate::fractals::FractalType;
use crate::renderer::{FractalUniforms, compute_reference_orbit, compute_reference_orbit_julia};
#[cfg(not(target_arch = "wasm32"))]
use crate::renderer::ds_split;

#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::process::{Command, Stdio};

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

/// Render fractal to raw RGBA pixels at the given resolution (shared by native and wasm export)
fn render_to_pixels(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fractal_type: &FractalType,
    uniforms: &FractalUniforms,
    resolution: ExportResolution,
) -> Result<(Vec<u8>, u32, u32), String> {
    let (width, height) = resolution.dimensions();
    let aspect_ratio = width as f32 / height as f32;

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

    Ok((pixels, width, height))
}

/// Export the current fractal view as a PNG file (native: saves to disk)
#[cfg(not(target_arch = "wasm32"))]
pub fn export_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fractal_type: &FractalType,
    uniforms: &FractalUniforms,
    resolution: ExportResolution,
    output_path: &Path,
) -> Result<(), String> {
    let (width, height) = resolution.dimensions();
    log::info!(
        "Exporting {}x{} PNG to {:?}...",
        width,
        height,
        output_path
    );

    let (pixels, width, height) = render_to_pixels(device, queue, fractal_type, uniforms, resolution)?;

    // Save as PNG
    let img: image::RgbaImage =
        image::ImageBuffer::from_raw(width, height, pixels)
            .ok_or_else(|| "Failed to create image from pixel data".to_string())?;

    img.save(output_path)
        .map_err(|e| format!("Failed to save PNG: {}", e))?;

    log::info!("Exported PNG: {:?} ({}x{})", output_path, width, height);
    Ok(())
}

/// Export the current fractal view as a PNG (wasm: async, triggers browser download)
///
/// On wasm, buffer mapping is asynchronous — we dispatch the GPU work, then
/// `spawn_local` an async task that awaits the map, encodes PNG, and triggers
/// a browser download.  The function returns immediately (the download happens
/// in the background).
#[cfg(target_arch = "wasm32")]
pub fn export_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fractal_type: &FractalType,
    uniforms: &FractalUniforms,
    resolution: ExportResolution,
    filename: &str,
) -> Result<(), String> {
    let (width, height) = resolution.dimensions();
    let aspect_ratio = width as f32 / height as f32;

    let zoom_f64 = uniforms.zoom as f64 + uniforms.zoom_lo as f64;
    let export_pixel_step_x = (1.0 / (zoom_f64 * height as f64)) as f32;
    let export_pixel_step_y = (-1.0 / (zoom_f64 * height as f64)) as f32;

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

    // Create GPU resources
    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Export Uniform Buffer"),
        size: std::mem::size_of::<FractalUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&export_uniforms));

    let storage_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Export Storage Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = storage_texture.create_view(&wgpu::TextureViewDescriptor::default());

    // Determine precision
    let use_f64 = zoom_f64 >= 5.0e3;
    let ftype = fractal_type.type_id();
    let use_perturbation = use_f64 && (ftype == 0 || ftype == 1);

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

    // Build bind group
    let mut layout_entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 },
            count: None,
        },
    ];
    if use_perturbation {
        layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        });
    }

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Export BGL"), entries: &layout_entries,
    });

    let mut bg_entries = vec![
        wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&texture_view) },
    ];
    if let Some(ref ob) = orbit_buffer {
        bg_entries.push(wgpu::BindGroupEntry { binding: 2, resource: ob.as_entire_binding() });
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Export BG"), layout: &bind_group_layout, entries: &bg_entries,
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Export PL"), bind_group_layouts: &[&bind_group_layout], push_constant_ranges: &[],
    });

    let shader_source = if use_f64 { fractal_type.shader_source_f64() } else { fractal_type.shader_source() };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Export Shader"), source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Export Pipeline"), layout: Some(&pipeline_layout),
        module: &shader, entry_point: "main", compilation_options: Default::default(), cache: None,
    });

    // Dispatch and copy to readback buffer
    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
    let buffer_size = (padded_bytes_per_row * height) as u64;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Export Output Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Export Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Export Compute Pass"), timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups((width + 15) / 16, (height + 15) / 16, 1);
    }

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture { texture: &storage_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::ImageCopyBuffer { buffer: &output_buffer, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(padded_bytes_per_row), rows_per_image: Some(height) } },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );

    queue.submit(std::iter::once(encoder.finish()));

    // Async readback — use Arc<AtomicBool> flag set by map_async callback,
    // then poll with setTimeout yields until the mapping is ready.
    let filename = filename.to_string();
    let mapped_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let mapped_flag_cb = mapped_flag.clone();

    output_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
        if let Err(ref e) = result {
            log::error!("map_async failed: {:?}", e);
        }
        mapped_flag_cb.store(result.is_ok(), std::sync::atomic::Ordering::SeqCst);
    });

    wasm_bindgen_futures::spawn_local(async move {
        // Yield to the JS event loop until the map callback fires
        let mut attempts = 0;
        while !mapped_flag.load(std::sync::atomic::Ordering::SeqCst) {
            gloo_timers_sleep(16).await;
            attempts += 1;
            if attempts > 300 { // ~5s timeout
                log::error!("Export timed out waiting for buffer mapping");
                return;
            }
        }

        // Buffer is now mapped — read pixels
        let data = output_buffer.slice(..).get_mapped_range();
        let mut pixels = Vec::with_capacity((width * height * bytes_per_pixel) as usize);
        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + unpadded_bytes_per_row as usize;
            pixels.extend_from_slice(&data[start..end]);
        }
        drop(data);
        output_buffer.unmap();

        // Encode PNG
        let img: image::RgbaImage = match image::ImageBuffer::from_raw(width, height, pixels) {
            Some(img) => img,
            None => {
                log::error!("Failed to create image from pixel data");
                return;
            }
        };
        let mut png_data: Vec<u8> = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut png_data);
        use image::ImageEncoder;
        if let Err(e) = encoder.write_image(&img, width, height, image::ExtendedColorType::Rgba8) {
            log::error!("Failed to encode PNG: {}", e);
            return;
        }

        match trigger_browser_download(&filename, &png_data, "image/png") {
            Ok(()) => log::info!("PNG download triggered: {} ({}x{})", filename, width, height),
            Err(e) => log::error!("Download failed: {}", e),
        }
    });

    log::info!("Export started (async)...");
    Ok(())
}

/// Simple async sleep for wasm using a JS setTimeout promise
#[cfg(target_arch = "wasm32")]
async fn gloo_timers_sleep(ms: u32) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        let _ = web_sys::window().unwrap().set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms as i32);
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
}

/// Trigger a file download in the browser via Blob + anchor click
#[cfg(target_arch = "wasm32")]
fn trigger_browser_download(filename: &str, data: &[u8], mime_type: &str) -> Result<(), String> {
    use wasm_bindgen::JsCast;
    use web_sys::BlobPropertyBag;

    let window = web_sys::window().ok_or("No window")?;
    let document = window.document().ok_or("No document")?;

    // Create Blob from data
    let uint8_array = js_sys::Uint8Array::from(data);
    let array = js_sys::Array::new();
    array.push(&uint8_array.buffer());

    let blob_opts = BlobPropertyBag::new();
    blob_opts.set_type(mime_type);
    let blob = web_sys::Blob::new_with_buffer_source_sequence_and_options(&array, &blob_opts)
        .map_err(|e| format!("Failed to create blob: {:?}", e))?;

    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|e| format!("Failed to create object URL: {:?}", e))?;

    // Create anchor element, set href and download, click it
    let anchor: web_sys::HtmlAnchorElement = document
        .create_element("a")
        .map_err(|e| format!("Failed to create anchor: {:?}", e))?
        .dyn_into()
        .map_err(|_| "Failed to cast to HtmlAnchorElement")?;

    anchor.set_href(&url);
    anchor.set_download(filename);
    anchor.click();

    // Clean up
    let _ = web_sys::Url::revoke_object_url(&url);

    Ok(())
}

/// Video recording settings
#[derive(Clone, Debug)]
pub struct VideoSettings {
    pub resolution: ExportResolution,
    pub fps: u32,
    pub duration_secs: f64,
    pub target_zoom: f64,
}

impl Default for VideoSettings {
    fn default() -> Self {
        Self {
            resolution: ExportResolution::HD1080p,
            fps: 30,
            duration_secs: 5.0,
            target_zoom: 1e6,
        }
    }
}

/// Check if ffmpeg is available on the system
#[cfg(not(target_arch = "wasm32"))]
fn check_ffmpeg() -> Result<(), String> {
    Command::new("ffmpeg")
        .arg("-version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|_| "ffmpeg not found. Please install ffmpeg to record videos.".to_string())?;
    Ok(())
}

/// Record a zoom animation video by rendering frames offline and piping to ffmpeg
#[cfg(not(target_arch = "wasm32"))]
pub fn record_video(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fractal_type: &FractalType,
    base_uniforms: &FractalUniforms,
    settings: &VideoSettings,
    center: glam::DVec2,
    start_zoom: f64,
    palette_lut: &[u8; 1024],
) -> Result<String, String> {
    check_ffmpeg()?;

    let (width, height) = settings.resolution.dimensions();
    let total_frames = (settings.fps as f64 * settings.duration_secs) as u32;
    if total_frames == 0 {
        return Err("Duration too short for any frames".to_string());
    }

    let output_filename = generate_video_filename(fractal_type.name(), &settings.resolution);
    let output_dir = Path::new("export");
    std::fs::create_dir_all(output_dir).map_err(|e| format!("Failed to create export dir: {}", e))?;
    let output_path = output_dir.join(&output_filename);

    log::info!(
        "Recording video: {}x{} @ {} fps, {} frames, zoom {:.2e} -> {:.2e}",
        width, height, settings.fps, total_frames, start_zoom, settings.target_zoom
    );

    // --- Create GPU resources (reused across all frames) ---
    let aspect_ratio = width as f32 / height as f32;

    let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Video Uniform Buffer"),
        size: std::mem::size_of::<FractalUniforms>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let palette_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Video Palette Buffer"),
        size: 1024,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&palette_buffer, 0, palette_lut);

    let storage_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Video Storage Texture"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let texture_view = storage_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = width * bytes_per_pixel;
    let padded_bytes_per_row = (unpadded_bytes_per_row + 255) & !255;
    let buffer_size = (padded_bytes_per_row * height) as u64;

    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Video Output Buffer"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Determine precision and create orbit buffer if needed
    let ftype = fractal_type.type_id();
    let max_zoom = start_zoom.max(settings.target_zoom);
    let ever_needs_perturbation = max_zoom >= 5.0e3 && (ftype == 0 || ftype == 1);

    let orbit_buffer = if ever_needs_perturbation {
        Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Video Orbit Buffer"),
            size: ((base_uniforms.max_iter as usize + 1) * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }))
    } else {
        None
    };

    // We need two pipelines: standard (3 bindings) and perturbation (4 bindings)
    // Standard: uniform + texture + palette
    let standard_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Video Standard BGL"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            },
        ],
    });

    let standard_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Video Standard BG"),
        layout: &standard_layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&texture_view) },
            wgpu::BindGroupEntry { binding: 2, resource: palette_buffer.as_entire_binding() },
        ],
    });

    let standard_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Video Standard PL"),
        bind_group_layouts: &[&standard_layout],
        push_constant_ranges: &[],
    });

    // Perturbation layout (4 bindings: uniform + texture + orbit + palette)
    let (perturbation_bind_group, perturbation_pipeline_layout) = if let Some(ref ob) = orbit_buffer {
        let perturbation_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Video Perturbation BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Video Perturbation BG"),
            layout: &perturbation_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&texture_view) },
                wgpu::BindGroupEntry { binding: 2, resource: ob.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: palette_buffer.as_entire_binding() },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Video Perturbation PL"),
            bind_group_layouts: &[&perturbation_layout],
            push_constant_ranges: &[],
        });

        (Some(bg), Some(pl))
    } else {
        (None, None)
    };

    // Create compute pipelines (standard f32, f64, and perturbation as needed)
    let shader_f32 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Video Shader f32"),
        source: wgpu::ShaderSource::Wgsl(fractal_type.shader_source().into()),
    });
    let pipeline_f32 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Video Pipeline f32"),
        layout: Some(&standard_pipeline_layout),
        module: &shader_f32,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    });

    // f64 pipeline (perturbation for Mandelbrot/Julia, standard layout for others)
    let shader_f64 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Video Shader f64"),
        source: wgpu::ShaderSource::Wgsl(fractal_type.shader_source_f64().into()),
    });
    let pipeline_f64 = if ever_needs_perturbation {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Video Pipeline f64 perturbation"),
            layout: Some(perturbation_pipeline_layout.as_ref().unwrap()),
            module: &shader_f64,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
    } else {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Video Pipeline f64"),
            layout: Some(&standard_pipeline_layout),
            module: &shader_f64,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        })
    };

    // Spawn ffmpeg
    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgba",
            "-s", &format!("{}x{}", width, height),
            "-r", &settings.fps.to_string(),
            "-i", "-",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "medium",
            "-crf", "18",
            output_path.to_str().unwrap_or("export/output.mp4"),
        ])
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn ffmpeg: {}", e))?;

    let mut ffmpeg_stdin = ffmpeg.stdin.take().ok_or("Failed to open ffmpeg stdin")?;

    // Zoom interpolation parameters
    let zoom_ratio = settings.target_zoom / start_zoom;
    let params = fractal_type.params();
    let workgroup_count_x = (width + 15) / 16;
    let workgroup_count_y = (height + 15) / 16;

    // Render each frame
    for frame in 0..total_frames {
        let t = if total_frames > 1 {
            frame as f64 / (total_frames - 1) as f64
        } else {
            0.0
        };
        let zoom = start_zoom * zoom_ratio.powf(t);
        let use_f64 = zoom >= 5.0e3;
        let use_perturbation = use_f64 && (ftype == 0 || ftype == 1);

        // Compute hi/lo splits
        let (center_x_hi, center_x_lo) = ds_split(center.x);
        let (center_y_hi, center_y_lo) = ds_split(center.y);
        let (zoom_hi, zoom_lo) = ds_split(zoom);
        let pixel_step_x = (1.0 / (zoom * height as f64)) as f32;
        let pixel_step_y = (-1.0 / (zoom * height as f64)) as f32;

        // Compute reference orbit if needed
        let ref_escape_iter = if use_perturbation {
            let (orbit_data, escape_iter) = if ftype == 0 {
                compute_reference_orbit(center.x, center.y, base_uniforms.max_iter)
            } else {
                compute_reference_orbit_julia(
                    center.x, center.y,
                    params.c_real as f64, params.c_imag as f64,
                    base_uniforms.max_iter,
                )
            };
            if let Some(ref ob) = orbit_buffer {
                queue.write_buffer(ob, 0, bytemuck::cast_slice(&orbit_data));
            }
            escape_iter
        } else {
            base_uniforms.max_iter
        };

        // Build uniforms for this frame
        let frame_uniforms = FractalUniforms::new(
            [center_x_hi, center_y_hi],
            zoom_hi,
            aspect_ratio,
            base_uniforms.max_iter,
            ftype,
            0,
            params.c_real,
            params.c_imag,
            [center_x_lo, center_y_lo],
            zoom_lo,
            pixel_step_x,
            pixel_step_y,
            ref_escape_iter,
            base_uniforms.rotation,
        );
        queue.write_buffer(&uniform_buffer, 0, bytemuck::bytes_of(&frame_uniforms));

        // Dispatch compute shader
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Video Frame Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Video Compute Pass"),
                timestamp_writes: None,
            });

            if use_perturbation {
                pass.set_pipeline(&pipeline_f64);
                pass.set_bind_group(0, perturbation_bind_group.as_ref().unwrap(), &[]);
            } else if use_f64 {
                pass.set_pipeline(&pipeline_f64);
                pass.set_bind_group(0, &standard_bind_group, &[]);
            } else {
                pass.set_pipeline(&pipeline_f32);
                pass.set_bind_group(0, &standard_bind_group, &[]);
            }

            pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        }

        // Copy texture to output buffer
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
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Read back pixels
        let buffer_slice = output_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|e| format!("Frame {}: failed to receive map result: {}", frame, e))?
            .map_err(|e| format!("Frame {}: failed to map buffer: {:?}", frame, e))?;

        // Extract unpadded pixel data and write to ffmpeg
        {
            let data = buffer_slice.get_mapped_range();
            for row in 0..height {
                let start = (row * padded_bytes_per_row) as usize;
                let end = start + unpadded_bytes_per_row as usize;
                if ffmpeg_stdin.write_all(&data[start..end]).is_err() {
                    drop(data);
                    output_buffer.unmap();
                    return Err("ffmpeg process terminated unexpectedly".to_string());
                }
            }
        }
        output_buffer.unmap();

        if (frame + 1) % 10 == 0 || frame == 0 || frame == total_frames - 1 {
            log::info!("Recording frame {}/{} (zoom: {:.2e})", frame + 1, total_frames, zoom);
        }
    }

    // Close stdin and wait for ffmpeg to finish
    drop(ffmpeg_stdin);
    let ffmpeg_output = ffmpeg.wait_with_output()
        .map_err(|e| format!("Failed to wait for ffmpeg: {}", e))?;

    if !ffmpeg_output.status.success() {
        let stderr = String::from_utf8_lossy(&ffmpeg_output.stderr);
        return Err(format!("ffmpeg exited with error: {}", stderr));
    }

    log::info!("Video saved: {:?}", output_path);
    Ok(output_filename)
}

/// Video recording stub for wasm (not supported)
#[cfg(target_arch = "wasm32")]
pub fn record_video(
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    _fractal_type: &FractalType,
    _base_uniforms: &FractalUniforms,
    _settings: &VideoSettings,
    _center: glam::DVec2,
    _start_zoom: f64,
    _palette_lut: &[u8; 1024],
) -> Result<String, String> {
    Err("Video recording is not supported in the browser".to_string())
}

/// Generate a timestamped filename for video export
#[cfg(not(target_arch = "wasm32"))]
pub fn generate_video_filename(fractal_name: &str, resolution: &ExportResolution) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    format!(
        "fractal_{}_{}_{}s.mp4",
        fractal_name.to_lowercase().replace(' ', "_"),
        resolution.label(),
        secs
    )
}

/// Generate a timestamped filename for export
pub fn generate_filename(fractal_name: &str, resolution: &ExportResolution) -> String {
    let secs = current_timestamp_secs();
    format!(
        "fractal_{}_{}_{}s.png",
        fractal_name.to_lowercase().replace(' ', "_"),
        resolution.label(),
        secs
    )
}

/// Get current timestamp in seconds (platform-conditional)
#[cfg(not(target_arch = "wasm32"))]
fn current_timestamp_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(target_arch = "wasm32")]
fn current_timestamp_secs() -> u64 {
    (js_sys::Date::now() / 1000.0) as u64
}
