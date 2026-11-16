use std::sync::Arc;
use winit::window::Window;

/// Shown when the browser has no WebGPU at all
#[cfg(target_arch = "wasm32")]
const NO_WEBGPU_MESSAGE: &str = "This browser does not support WebGPU, which Fractal Explorer needs to draw. \
Try a recent version of Chrome or Edge, or Safari 18+. In Firefox, WebGPU may need to be enabled in the settings. \
WebGPU is also only available on pages loaded over HTTPS.";

/// Shown when WebGPU (or a native backend) exists but offers no usable adapter
const NO_ADAPTER_MESSAGE: &str = "No suitable GPU adapter was found. \
Your graphics hardware or driver may be unsupported or blocked, or hardware acceleration may be turned off.";

/// GPU context holding all wgpu resources
pub struct GpuContext {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface_config: wgpu::SurfaceConfiguration,
}

impl GpuContext {
    /// Initialize wgpu with a window.
    ///
    /// Fails with a message fit to show the user when there is no usable GPU
    /// (e.g. a browser without WebGPU) instead of panicking.
    pub async fn new(window: Arc<Window>) -> Result<Self, String> {
        // wgpu's WebGPU backend calls straight into `navigator.gpu` and throws when it
        // is missing, so check before touching it.
        #[cfg(target_arch = "wasm32")]
        if !webgpu_available() {
            return Err(NO_WEBGPU_MESSAGE.to_string());
        }

        // Create instance
        #[cfg(target_arch = "wasm32")]
        let backends = wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL;
        #[cfg(not(target_arch = "wasm32"))]
        let backends = wgpu::Backends::all();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        // Create surface
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| format!("Could not create a drawing surface: {e}"))?;

        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| NO_ADAPTER_MESSAGE.to_string())?;

        // Request device and queue
        // On wasm, use conservative limits to avoid requesting unrecognized WebGPU limits
        #[cfg(target_arch = "wasm32")]
        let required_limits = wgpu::Limits::downlevel_webgl2_defaults();
        #[cfg(not(target_arch = "wasm32"))]
        let required_limits = wgpu::Limits::default();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Main Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .map_err(|e| format!("Could not initialise the GPU device: {e}"))?;

        // Configure surface
        let surface_caps = surface.get_capabilities(&adapter);
        let (Some(&first_format), Some(&alpha_mode)) = (surface_caps.formats.first(), surface_caps.alpha_modes.first())
        else {
            return Err("The GPU adapter cannot draw to this window.".to_string());
        };
        let surface_format = surface_caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(first_format);

        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo, // VSync
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &surface_config);

        log::info!("GPU initialized successfully");
        log::info!("  Adapter: {:?}", adapter.get_info());
        log::info!("  Surface format: {:?}", surface_format);

        Ok(Self { surface, device, queue, surface_config })
    }

    /// Resize surface
    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width > 0 && new_height > 0 {
            self.surface_config.width = new_width;
            self.surface_config.height = new_height;
            self.surface.configure(&self.device, &self.surface_config);
            log::debug!("Surface resized to {}x{}", new_width, new_height);
        }
    }

    /// Get current surface texture
    pub fn get_current_texture(&self) -> Result<wgpu::SurfaceTexture, wgpu::SurfaceError> {
        self.surface.get_current_texture()
    }
}

/// Whether the browser exposes WebGPU (`navigator.gpu`)
#[cfg(target_arch = "wasm32")]
fn webgpu_available() -> bool {
    use wasm_bindgen::JsValue;
    let get = |target: &JsValue, key: &str| js_sys::Reflect::get(target, &JsValue::from_str(key)).ok();
    get(&js_sys::global(), "navigator")
        .and_then(|navigator| get(&navigator, "gpu"))
        .is_some_and(|gpu| !gpu.is_undefined() && !gpu.is_null())
}
