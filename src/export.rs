use crate::fractals::FractalType;
use crate::renderer::fractal_pass::{create_palette_buffer, PALETTE_LUT_BYTES};
use crate::renderer::{FractalPipelines, FractalTarget, View};

#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::process::{Command, Stdio};

/// Directory exported files are written to (native)
#[cfg(not(target_arch = "wasm32"))]
const EXPORT_DIR: &str = "export";

/// Palette LUT as uploaded to the GPU
pub type PaletteLut = [u8; PALETTE_LUT_BYTES as usize];

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

/// An off-screen fractal render target with a CPU readback buffer.
/// Used for PNG export and for every frame of a video.
struct Offscreen {
    pipelines: FractalPipelines,
    target: FractalTarget,
    /// Destination of the texture copy; rows are padded to wgpu's row alignment
    readback: wgpu::Buffer,
    padded_bytes_per_row: u32,
    // Bound into `target`'s bind groups; kept alongside for clarity of ownership
    _palette_buffer: wgpu::Buffer,
}

impl Offscreen {
    const BYTES_PER_PIXEL: u32 = 4;

    fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32, palette_lut: &PaletteLut) -> Self {
        let pipelines = FractalPipelines::new(device);
        let palette_buffer = create_palette_buffer(device);
        queue.write_buffer(&palette_buffer, 0, palette_lut);
        let target = FractalTarget::new(
            device,
            &pipelines,
            &palette_buffer,
            "Export",
            width,
            height,
            wgpu::TextureUsages::COPY_SRC,
        );

        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = (width * Self::BYTES_PER_PIXEL).div_ceil(align) * align;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Export Readback Buffer"),
            size: padded_bytes_per_row as u64 * height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self { pipelines, target, readback, padded_bytes_per_row, _palette_buffer: palette_buffer }
    }

    /// Render `fractal` at `view` and copy the result into the readback buffer
    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        fractal: &FractalType,
        view: &View,
    ) -> Result<(), String> {
        let (width, height) = (self.target.width(), self.target.height());
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Export Command Encoder"),
        });

        self.target.render(device, queue, &mut encoder, &mut self.pipelines, fractal, view)?;

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.target.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.readback,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Tightly packed RGBA pixels from the mapped readback buffer (drops row padding)
    fn mapped_pixels(&self) -> Vec<u8> {
        let (width, height) = (self.target.width(), self.target.height());
        let row_bytes = (width * Self::BYTES_PER_PIXEL) as usize;
        let data = self.readback.slice(..).get_mapped_range();
        let mut pixels = Vec::with_capacity(row_bytes * height as usize);
        for row in data.chunks(self.padded_bytes_per_row as usize) {
            pixels.extend_from_slice(&row[..row_bytes]);
        }
        pixels
    }

    /// Wait for the last `render` and return its pixels (RGBA, row-major)
    #[cfg(not(target_arch = "wasm32"))]
    fn read_pixels(&self, device: &wgpu::Device) -> Result<Vec<u8>, String> {
        let (sender, receiver) = std::sync::mpsc::channel();
        self.readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|e| format!("Failed to receive map result: {}", e))?
            .map_err(|e| format!("Failed to map buffer: {:?}", e))?;

        let pixels = self.mapped_pixels();
        self.readback.unmap();
        Ok(pixels)
    }

    /// Wait for the last `render` and return its pixels (RGBA, row-major).
    /// Buffer mapping is asynchronous in the browser, so this yields to the JS event
    /// loop until the map callback has fired.
    #[cfg(target_arch = "wasm32")]
    async fn read_pixels(&self) -> Result<Vec<u8>, String> {
        use std::sync::{Arc, Mutex};

        // wgpu wants a Send callback even on the single-threaded web target
        let mapped: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> = Arc::new(Mutex::new(None));
        let mapped_cb = mapped.clone();
        self.readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            *mapped_cb.lock().unwrap() = Some(result);
        });

        // ~5 s timeout
        for _ in 0..300 {
            let result = mapped.lock().unwrap().take();
            if let Some(result) = result {
                result.map_err(|e| format!("Failed to map buffer: {:?}", e))?;
                let pixels = self.mapped_pixels();
                self.readback.unmap();
                return Ok(pixels);
            }
            sleep_ms(16).await;
        }
        Err("Timed out waiting for buffer mapping".to_string())
    }
}

/// Encode tightly packed RGBA pixels as a PNG file in memory
fn encode_png(pixels: &[u8], width: u32, height: u32) -> Result<Vec<u8>, String> {
    use image::ImageEncoder;
    let mut png = Vec::new();
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(pixels, width, height, image::ExtendedColorType::Rgba8)
        .map_err(|e| format!("Failed to encode PNG: {}", e))?;
    Ok(png)
}

/// Export the fractal at `view` as a PNG named `filename`.
///
/// Native: renders, waits for the result and writes `export/<filename>`.
/// Browser: readback is asynchronous there, so this starts the render and returns;
/// the download is triggered from a background task once the pixels are available.
pub fn export_png(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fractal: &FractalType,
    view: &View,
    resolution: ExportResolution,
    filename: &str,
    palette_lut: &PaletteLut,
) -> Result<(), String> {
    let (width, height) = resolution.dimensions();
    log::info!("Exporting {}x{} PNG: {}", width, height, filename);

    let mut offscreen = Offscreen::new(device, queue, width, height, palette_lut);
    offscreen.render(device, queue, fractal, view)?;

    #[cfg(not(target_arch = "wasm32"))]
    {
        let png = encode_png(&offscreen.read_pixels(device)?, width, height)?;
        std::fs::create_dir_all(EXPORT_DIR).map_err(|e| format!("Failed to create export dir: {}", e))?;
        let path = Path::new(EXPORT_DIR).join(filename);
        std::fs::write(&path, png).map_err(|e| format!("Failed to save PNG: {}", e))?;
        log::info!("Exported PNG: {:?} ({}x{})", path, width, height);
    }

    #[cfg(target_arch = "wasm32")]
    {
        let filename = filename.to_string();
        wasm_bindgen_futures::spawn_local(async move {
            let result = match offscreen.read_pixels().await {
                Ok(pixels) => encode_png(&pixels, width, height)
                    .and_then(|png| trigger_browser_download(&filename, &png, "image/png")),
                Err(e) => Err(e),
            };
            match result {
                Ok(()) => log::info!("PNG download triggered: {} ({}x{})", filename, width, height),
                Err(e) => log::error!("Export failed: {}", e),
            }
        });
        log::info!("Export started (async)...");
    }

    Ok(())
}

/// Simple async sleep for wasm using a JS setTimeout promise
#[cfg(target_arch = "wasm32")]
async fn sleep_ms(ms: u32) {
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

/// Record a zoom animation video by rendering frames offline and piping to ffmpeg.
/// The zoom runs from `start.zoom` to `settings.target_zoom`; everything else stays
/// as in `start`. Returns the name of the file written under `export/`.
#[cfg(not(target_arch = "wasm32"))]
pub fn record_video(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    fractal: &FractalType,
    start: &View,
    settings: &VideoSettings,
    palette_lut: &PaletteLut,
) -> Result<String, String> {
    check_ffmpeg()?;

    let (width, height) = settings.resolution.dimensions();
    let total_frames = (settings.fps as f64 * settings.duration_secs) as u32;
    if total_frames == 0 {
        return Err("Duration too short for any frames".to_string());
    }

    let output_filename = generate_video_filename(fractal.name(), &settings.resolution);
    let output_dir = Path::new(EXPORT_DIR);
    std::fs::create_dir_all(output_dir).map_err(|e| format!("Failed to create export dir: {}", e))?;
    let output_path = output_dir.join(&output_filename);

    log::info!(
        "Recording video: {}x{} @ {} fps, {} frames, zoom {:.2e} -> {:.2e}",
        width, height, settings.fps, total_frames, start.zoom, settings.target_zoom
    );

    // GPU resources are reused across all frames
    let mut offscreen = Offscreen::new(device, queue, width, height, palette_lut);

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
        ])
        .arg(&output_path)
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to spawn ffmpeg: {}", e))?;

    let mut ffmpeg_stdin = ffmpeg.stdin.take().ok_or("Failed to open ffmpeg stdin")?;

    // Exponential zoom interpolation
    let zoom_ratio = settings.target_zoom / start.zoom;

    for frame in 0..total_frames {
        let t = if total_frames > 1 {
            frame as f64 / (total_frames - 1) as f64
        } else {
            0.0
        };
        let view = View { zoom: start.zoom * zoom_ratio.powf(t), ..*start };

        offscreen
            .render(device, queue, fractal, &view)
            .map_err(|e| format!("Frame {}: {}", frame, e))?;
        let pixels = offscreen
            .read_pixels(device)
            .map_err(|e| format!("Frame {}: {}", frame, e))?;
        ffmpeg_stdin
            .write_all(&pixels)
            .map_err(|_| "ffmpeg process terminated unexpectedly".to_string())?;

        if (frame + 1) % 10 == 0 || frame == 0 || frame == total_frames - 1 {
            log::info!("Recording frame {}/{} (zoom: {:.2e})", frame + 1, total_frames, view.zoom);
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
    _fractal: &FractalType,
    _start: &View,
    _settings: &VideoSettings,
    _palette_lut: &PaletteLut,
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

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::coloring::ColorScheme;
    use glam::DVec2;

    /// A device without a window, or `None` on machines with no usable GPU.
    fn headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
    }

    /// Every per-pixel fractal renders off-screen, at ordinary and at deep zoom,
    /// without GPU validation errors and with something other than a flat image.
    #[test]
    fn every_fractal_renders_offscreen() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let lut = ColorScheme::default().get_palette().generate_lut();
        let (width, height) = (96, 64);
        // Deep views chosen on the boundary of each set so the picture has structure
        let deep_centers = [
            DVec2::new(-0.743643887037151, 0.131825904205330),
            // Repelling fixed point of the default Julia set, which lies on the set
            DVec2::new(1.4842927426042225, -0.13723051797712932),
            DVec2::new(-1.4679944402079477, -0.12282724805880647),
            DVec2::new(-1.3663993401074568, -0.008939771987894548),
            DVec2::ZERO,
            DVec2::new(-1.5, -0.8154231018505872),
            // The gasket repeats at every scale around the origin
            DVec2::new(2.0e-6, 1.0e-6),
        ];

        for (fractal, deep_center) in FractalType::ALL.iter().zip(deep_centers) {
            let views = [
                View { center: fractal.default_center(), zoom: fractal.default_zoom(), rotation: 0.3, max_iter: 256 },
                View { center: deep_center, zoom: 1.0e5, rotation: 0.3, max_iter: 256 },
            ];
            for (i, view) in views.iter().enumerate() {
                device.push_error_scope(wgpu::ErrorFilter::Validation);
                let mut offscreen = Offscreen::new(&device, &queue, width, height, &lut);
                let result = offscreen.render(&device, &queue, fractal, view);
                let error = pollster::block_on(device.pop_error_scope());
                assert!(error.is_none(), "{}: {error:?}", fractal.name());

                if fractal.is_buddhabrot() {
                    assert!(result.is_err(), "Buddhabrot has no per-pixel shader to export with");
                    continue;
                }
                result.unwrap();
                let pixels = offscreen.read_pixels(&device).unwrap();
                assert_eq!(pixels.len(), (width * height * 4) as usize);
                assert!(
                    pixels.chunks(4).any(|p| p != &pixels[..4]),
                    "{} view {i}: image is a single flat colour",
                    fractal.name()
                );
                if let Ok(dir) = std::env::var("FRACTAL_DUMP_DIR") {
                    std::fs::write(format!("{dir}/{}_{i}.rgba", fractal.id()), &pixels).unwrap();
                }
            }
        }
    }
}
