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
            // Repelling fixed point of the default Nova map
            DVec2::new(-0.051901701367768276, 0.5652358516771708),
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

    // ---- Accuracy against a CPU f64 reference ---------------------------------------

    /// Smooth escape time as the shaders define it (bailout radius² 256), in f64.
    fn escape_time(z0: DVec2, n: u32, step: impl Fn(DVec2) -> DVec2) -> f64 {
        let mut z = z0;
        for i in 0..n {
            let m = z.length_squared();
            if m > 256.0 {
                return i as f64 + 1.0 - (m.log2() / 2.0).log2();
            }
            z = step(z);
        }
        0.0
    }

    /// The Nova shader's value in f64.
    fn nova_value(z0: DVec2, c: DVec2, n: u32) -> f64 {
        let mut z = z0;
        let mut i = 0;
        while i < n {
            if z.length_squared() > 1.0e8 {
                break;
            }
            let z2 = DVec2::new(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y);
            let m = z2.length_squared();
            if 9.0 * m < 1.0e-20 {
                break;
            }
            let next = z * (2.0 / 3.0) + DVec2::new(z2.x, -z2.y) / (3.0 * m) + c;
            let d2 = (next - z).length_squared();
            z = next;
            i += 1;
            if d2 < 1.0e-6 {
                return (i as f64 - ((d2 + 1.0e-20).log2() / 1.0e-6f64.log2()).log2()).max(0.0);
            }
        }
        if i == n { 0.0 } else { i as f64 }
    }

    /// The f64 twin of each per-pixel fractal: its value at plane point `p`
    fn cpu_value(fractal: &FractalType, p: DVec2, n: u32) -> f64 {
        let c = fractal.c().as_dvec2();
        match fractal {
            FractalType::Mandelbrot => {
                escape_time(DVec2::ZERO, n, |z| DVec2::new(z.x * z.x - z.y * z.y + p.x, 2.0 * z.x * z.y + p.y))
            }
            FractalType::BurningShip => {
                escape_time(DVec2::ZERO, n, |z| DVec2::new(z.x * z.x - z.y * z.y + p.x, 2.0 * (z.x * z.y).abs() + p.y))
            }
            FractalType::Tricorn => {
                escape_time(DVec2::ZERO, n, |z| DVec2::new(z.x * z.x - z.y * z.y + p.x, -2.0 * z.x * z.y + p.y))
            }
            FractalType::Julia { .. } => {
                escape_time(p, n, |z| DVec2::new(z.x * z.x - z.y * z.y + c.x, 2.0 * z.x * z.y + c.y))
            }
            FractalType::Nova { .. } => nova_value(p, c, n),
            FractalType::Buddhabrot | FractalType::GoldenSilver => unreachable!(),
        }
    }

    /// A point on an edge between two classes of points, to f64 precision: scans the
    /// box `lo..hi` for two neighbouring grid points that differ, then bisects.
    fn find_edge<T: PartialEq>(class: impl Fn(DVec2) -> T, lo: DVec2, hi: DVec2) -> DVec2 {
        const GRID: u32 = 23;
        let at = |i: u32, j: u32| lo + (hi - lo) * DVec2::new(i as f64, j as f64) / GRID as f64;
        let (mut a, mut b) = (0..GRID)
            .flat_map(|j| (0..GRID).map(move |i| (i, j)))
            .map(|(i, j)| (at(i, j), at(i + 1, j)))
            .find(|(a, b)| class(*a) != class(*b))
            .expect("the box contains an edge");
        let class_a = class(a);
        for _ in 0..80 {
            let mid = (a + b) / 2.0;
            if class(mid) == class_a {
                a = mid;
            } else {
                b = mid;
            }
        }
        a
    }

    /// How a GPU render compares with the CPU reference
    struct Accuracy {
        /// % of pixels on the wrong side of inside / outside
        wrong_side: f64,
        /// % of outside pixels more than one iteration away from the reference
        wrong_value: f64,
        /// Number of different palette entries in the picture
        levels: usize,
    }

    /// Render `fractal` and compare each pixel with the f64 reference. A pixel only
    /// counts as wrong if it matches neither its own reference value nor that of any
    /// neighbour: right at a fractal's edge, values legitimately change within a pixel.
    fn measure(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        fractal: &FractalType,
        center: DVec2,
        zoom: f64,
        max_iter: u32,
    ) -> Accuracy {
        let (w, h) = (160u32, 120u32);
        // Palette index in red; green tells outside (255) from inside (0)
        let mut lut = [0u8; 1024];
        for i in 0..256 {
            lut[i * 4..i * 4 + 4].copy_from_slice(&[i as u8, 255, 0, 255]);
        }
        let view = View { center, zoom, rotation: 0.0, max_iter };
        let mut offscreen = Offscreen::new(device, queue, w, h, &lut);
        offscreen.render(device, queue, fractal, &view).unwrap();
        let gpu = offscreen.read_pixels(device).unwrap();

        let reference: Vec<f64> = (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f64, (i / w) as f64);
                let offset = DVec2::new(x - w as f64 / 2.0, -(y - h as f64 / 2.0)) / (h as f64 * zoom);
                cpu_value(fractal, center + offset, max_iter)
            })
            .collect();

        let (mut wrong_side, mut wrong_value) = (0, 0);
        let mut levels = std::collections::HashSet::new();
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let pixel = &gpu[((y as u32 * w + x as u32) * 4) as usize..][..4];
                let gpu_inside = pixel[1] == 0;
                levels.insert((pixel[0], pixel[1]));
                let (mut side_ok, mut value_ok) = (false, false);
                for (dx, dy) in (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dx, dy))) {
                    let (nx, ny) = (x + dx, y + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let t = reference[(ny as u32 * w + nx as u32) as usize];
                    if (t == 0.0) != gpu_inside {
                        continue;
                    }
                    side_ok = true;
                    // One iteration is 0.05 of the palette = 12.75 entries
                    let want = ((t * 0.05).fract() * 255.0) as i32;
                    let distance = (want - pixel[0] as i32).rem_euclid(255);
                    value_ok |= gpu_inside || distance.min(255 - distance) <= 13;
                }
                if !side_ok {
                    wrong_side += 1;
                } else if !value_ok {
                    wrong_value += 1;
                }
            }
        }
        let percent = |count: i32| count as f64 * 100.0 / (w * h) as f64;
        Accuracy { wrong_side: percent(wrong_side), wrong_value: percent(wrong_value), levels: levels.len() }
    }

    /// Deep zoom must show the same picture an f64 CPU render does, for every fractal
    /// with a deep-zoom shader and all the way down. This is what catches a shader that
    /// silently degrades to f32 (blocky, then one flat colour) or drifts off the true c.
    #[test]
    fn deep_zoom_matches_f64_reference() {
        let Some((device, queue)) = headless_device() else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let n = 400;
        let julia = FractalType::ALL[1].clone();
        let nova = FractalType::ALL[5].clone();
        let (julia_c, nova_c) = (julia.c().as_dvec2(), nova.c().as_dvec2());
        let cmul = |a: DVec2, b: DVec2| DVec2::new(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        let cdiv = |a: DVec2, b: DVec2| DVec2::new(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / b.length_squared();
        // Newton's method for a complex root of `f`, given f and f'
        let root = |start: DVec2, f: &dyn Fn(DVec2) -> (DVec2, DVec2)| {
            (0..100).fold(start, |z, _| {
                let (value, slope) = f(z);
                z - cdiv(value, slope)
            })
        };

        // Each centre sits on the fractal itself, so there is structure at every zoom.
        // Escape-time sets: a point on the edge of the interior.
        let inside = |f: FractalType| move |p: DVec2| cpu_value(&f, p, n) == 0.0;
        let box_around = |c: DVec2, r: f64| (c - r, c + r);
        let edge_of = |f: &FractalType, (lo, hi): (DVec2, DVec2)| find_edge(inside(f.clone()), lo, hi);
        // Julia: the repelling fixed point of z² + c (z² - z + c = 0) lies on the set
        let julia_point = root(DVec2::new(1.5, -0.1), &|z| (cmul(z, z) - z + julia_c, 2.0 * z - DVec2::X));
        // Nova: a repelling fixed point of 2z/3 + 1/(3z²) + c (z³ - 3cz² - 1 = 0)
        let nova_point = root(DVec2::new(-0.1, 0.6), &|z| {
            let z2 = cmul(z, z);
            (cmul(z2, z) - 3.0 * cmul(nova_c, z2) - DVec2::X, 3.0 * z2 - 6.0 * cmul(nova_c, z))
        });
        let cases = [
            (FractalType::Mandelbrot, edge_of(&FractalType::Mandelbrot, box_around(DVec2::new(-0.75, 0.2), 0.2))),
            // Around the period-3 minibrot on the needle
            (FractalType::Mandelbrot, edge_of(&FractalType::Mandelbrot, box_around(DVec2::new(-1.7549, 0.0), 0.02))),
            (julia, julia_point),
            (FractalType::BurningShip, edge_of(&FractalType::BurningShip, box_around(DVec2::new(-0.5, -0.5), 1.5))),
            // The small ship on the needle
            (FractalType::BurningShip, edge_of(&FractalType::BurningShip, box_around(DVec2::new(-1.755, -0.03), 0.04))),
            (FractalType::Tricorn, edge_of(&FractalType::Tricorn, box_around(DVec2::ZERO, 2.0))),
            (nova, nova_point),
        ];

        for (fractal, center) in &cases {
            for zoom in [1.0e2, 1.0e5, 1.0e8, 1.0e11] {
                let a = measure(&device, &queue, fractal, *center, zoom, n);
                let label = format!(
                    "{} at ({}, {}) zoom {zoom:.0e}: wrong side {:.2}%, wrong value {:.2}%, {} levels",
                    fractal.name(), center.x, center.y, a.wrong_side, a.wrong_value, a.levels
                );
                println!("{label}");
                assert!(a.wrong_side < 5.0, "{label}");
                if matches!(fractal, FractalType::BurningShip) {
                    // The Burning Ship's fold makes slow orbits depend on the last bit of
                    // every operation: f32 and f64 disagree on them at any zoom, even on
                    // identical inputs. So check the picture keeps its detail instead.
                    assert!(a.levels > 100, "{label}");
                } else {
                    assert!(a.wrong_value < 5.0, "{label}");
                }
            }
        }
    }
}
