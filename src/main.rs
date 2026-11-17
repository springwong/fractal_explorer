mod camera;
mod coloring;
mod export;
mod fractals;
mod renderer;
mod ui;

use camera::Camera;
use coloring::ColorScheme;
use fractals::{FractalType, MAX_ITERATIONS, MIN_ITERATIONS};
use glam::{UVec2, Vec2};
use renderer::{ComputePipeline, GpuContext, RenderPipeline, View};
use export::{ExportResolution, VideoSettings};
use ui::{fractal_for_letter, ControlPanel, FractalPicker, PanelAction, PaletteEditor, RecordingState, UiContext};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

/// Which viewport the mouse is over in linked mode
#[derive(Clone, Copy, PartialEq)]
enum ActiveViewport {
    Left,
    Right,
    None,
}

/// Everything the fractal compute pass depends on. While this is unchanged the
/// previous frame's texture is still correct and the compute pass can be skipped.
#[derive(Clone, PartialEq)]
struct ViewKey {
    fractal: FractalType,
    view: View,
    size: (u32, u32),
    /// Julia view and c when linked mode is on
    linked: Option<(View, Vec2)>,
}

/// Portable instant type for FPS tracking
#[cfg(not(target_arch = "wasm32"))]
type FrameInstant = std::time::Instant;

#[cfg(target_arch = "wasm32")]
#[derive(Clone, Copy)]
struct FrameInstant(f64);

#[cfg(target_arch = "wasm32")]
impl FrameInstant {
    fn now() -> Self {
        Self(js_sys::Date::now())
    }
    fn elapsed(&self) -> std::time::Duration {
        let ms = js_sys::Date::now() - self.0;
        std::time::Duration::from_millis(ms.max(0.0) as u64)
    }
}

/// Everything that exists only once the GPU is up
struct GpuState {
    gpu: GpuContext,
    compute: ComputePipeline,
    render: RenderPipeline,
    ui: UiContext,
}

/// Events sent to the event loop from outside winit
enum AppEvent {
    /// Asynchronous GPU initialisation finished (the browser cannot block on it)
    GpuReady(Result<Box<GpuState>, String>),
}

/// Main application state
struct App {
    window: Option<Arc<Window>>,
    /// Hands results of async work back to the event loop
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    proxy: EventLoopProxy<AppEvent>,
    /// GPU initialisation failed; the error is on screen and nothing more is drawn
    gpu_failed: bool,
    gpu: Option<GpuContext>,
    camera: Camera,
    compute: Option<ComputePipeline>,
    render: Option<RenderPipeline>,
    ui: Option<UiContext>,

    // Input state
    mouse_pos: Vec2,
    mouse_pressed: bool,
    last_mouse_pos: Option<Vec2>,

    // Touch state (for mobile)
    touches: std::collections::HashMap<u64, Vec2>,
    last_pinch_distance: Option<f32>,

    // Fractal parameters
    max_iter: u32,
    current_fractal: FractalType,
    current_color: ColorScheme,

    /// Overlay for choosing a fractal by picture (Space)
    fractal_picker: FractalPicker,
    /// Fractal chosen in the picker with the mouse, applied at the start of the next frame
    pending_fractal: Option<FractalType>,

    // Palette state
    palette_editor: PaletteEditor,
    palette_dirty: bool,

    // Export state
    pending_export: Option<ExportResolution>,
    pending_record: Option<VideoSettings>,
    pending_resize_canvas: bool,
    resize_countdown: u32, // frames to wait before auto-resize on wasm
    recording_state: RecordingState,
    /// View state the fractal texture was last computed for (None = must recompute)
    rendered_view: Option<ViewKey>,

    // Buddhabrot state
    buddhabrot_seed: u32,
    buddhabrot_dirty: bool,
    prev_fractal_id: u32,

    // FPS tracking
    last_frame_time: FrameInstant,
    frame_count: u32,
    fps: f32,

    // Linked mode state
    linked_mode: bool,
    julia_camera: Camera,
    linked_julia_c: Vec2,
    saved_fractal: Option<FractalType>,
    active_viewport: ActiveViewport,
    pending_toggle_linked: bool,
}

impl App {
    fn new(proxy: EventLoopProxy<AppEvent>) -> Self {
        Self {
            window: None,
            proxy,
            gpu_failed: false,
            gpu: None,
            camera: Camera::new(UVec2::new(1920, 1080)),
            compute: None,
            render: None,
            ui: None,
            mouse_pos: Vec2::ZERO,
            mouse_pressed: false,
            last_mouse_pos: None,
            touches: std::collections::HashMap::new(),
            last_pinch_distance: None,
            max_iter: 256,
            current_fractal: FractalType::Mandelbrot,
            current_color: ColorScheme::default(),
            fractal_picker: FractalPicker::default(),
            pending_fractal: None,
            palette_editor: PaletteEditor::new(),
            palette_dirty: true, // Upload initial palette on first frame
            pending_export: None,
            pending_record: None,
            pending_resize_canvas: false,
            resize_countdown: 0,
            recording_state: RecordingState::default(),
            rendered_view: None,
            buddhabrot_seed: 0,
            buddhabrot_dirty: true,
            prev_fractal_id: 0,
            last_frame_time: FrameInstant::now(),
            frame_count: 0,
            fps: 0.0,
            linked_mode: false,
            julia_camera: Camera::new(UVec2::new(1920, 1080)),
            linked_julia_c: Vec2::new(-0.7, 0.27015),
            saved_fractal: None,
            active_viewport: ActiveViewport::None,
            pending_toggle_linked: false,
        }
    }

    /// Initialize GPU and pipelines after window creation (async)
    async fn init_gpu(window: Arc<Window>) -> Result<GpuState, String> {
        let size = window.inner_size();
        // Fallback to reasonable default if canvas hasn't been laid out yet
        let w = if size.width > 0 { size.width } else { 1920 };
        let h = if size.height > 0 { size.height } else { 1080 };
        let gpu = GpuContext::new(window.clone()).await?;
        let compute = ComputePipeline::new(&gpu.device, w, h);
        let render = RenderPipeline::new(&gpu.device, gpu.surface_config.format, &compute.main.view, &compute.julia.view);
        let ui = UiContext::new(&gpu.device, gpu.surface_config.format, &window);
        Ok(GpuState { gpu, compute, render, ui })
    }

    /// Take over the result of GPU initialisation. On failure the user is told why
    /// instead of being left with a blank window.
    fn on_gpu_ready(&mut self, event_loop: &ActiveEventLoop, result: Result<Box<GpuState>, String>) {
        let state = match result {
            Ok(state) => *state,
            Err(message) => {
                log::error!("GPU initialisation failed: {message}");
                show_fatal_error(&message);
                self.gpu_failed = true;
                // Nothing will ever be drawn: stop spinning the event loop.
                // A native window has nowhere to show the message, so close it.
                event_loop.set_control_flow(ControlFlow::Wait);
                #[cfg(not(target_arch = "wasm32"))]
                event_loop.exit();
                return;
            }
        };
        let GpuState { mut gpu, compute, render, mut ui } = state;

        // Hand the fractal picker its preview images
        let thumbnails = compute
            .thumbnail_views()
            .map(|view| ui.egui_renderer.register_native_texture(&gpu.device, view, wgpu::FilterMode::Linear))
            .collect();
        self.fractal_picker.set_thumbnails(thumbnails);

        // The surface may have been configured before the canvas was laid out
        // (browser). Reconfigure with the actual size.
        if let Some(size) = self.window.as_ref().map(|w| w.inner_size()) {
            if size.width > 0 && size.height > 0 {
                gpu.resize(size.width, size.height);
                self.camera.resize(UVec2::new(size.width, size.height));
                self.julia_camera.resize(UVec2::new(size.width, size.height));
            }
        }

        log::info!(
            "Application initialized successfully, size: {}x{}",
            gpu.surface_config.width,
            gpu.surface_config.height
        );
        self.ui = Some(ui);
        self.gpu = Some(gpu);
        self.compute = Some(compute);
        self.render = Some(render);

        // Auto-resize after ~1s to fix mobile viewport height
        #[cfg(target_arch = "wasm32")]
        {
            self.resize_countdown = 60;
        }
        if let Some(ref window) = self.window {
            window.request_redraw();
        }
    }

    /// Switch to `fractal`, starting from its default view
    fn select_fractal(&mut self, fractal: FractalType) {
        self.current_fractal = fractal;
        self.camera.center = self.current_fractal.default_center();
        self.camera.zoom = self.current_fractal.default_zoom();
        log::info!("Switched to: {}", self.current_fractal.name());
    }

    /// Switch to a fractal chosen in the fractal picker. The picker offers every
    /// fractal, so this leaves the linked split view if it is on.
    fn pick_fractal(&mut self, fractal: FractalType) {
        if self.linked_mode {
            self.toggle_linked_mode();
        }
        self.select_fractal(fractal);
    }

    /// Move the Julia / Nova parameter c by `delta`
    fn nudge_c(&mut self, delta: Vec2) {
        let name = self.current_fractal.name();
        if let Some(c) = self.current_fractal.c_mut() {
            *c += delta;
            log::info!("{} c = ({:.4}, {:.4})", name, c.x, c.y);
        }
    }

    fn toggle_linked_mode(&mut self) {
        self.linked_mode = !self.linked_mode;

        if self.linked_mode {
            // Entering linked mode: save current state, switch to Mandelbrot
            self.saved_fractal = Some(self.current_fractal.clone());
            self.current_fractal = FractalType::Mandelbrot;
            self.camera.center = FractalType::Mandelbrot.default_center();
            self.camera.zoom = FractalType::Mandelbrot.default_zoom();
            self.camera.rotation = 0.0;

            // Initialize Julia camera
            let julia_default = FractalType::Julia { c: Vec2::ZERO };
            self.julia_camera.center = julia_default.default_center();
            self.julia_camera.zoom = julia_default.default_zoom();
            self.julia_camera.rotation = 0.0;

            log::info!("Linked mode enabled: Mandelbrot (left) + Julia (right)");
        } else {
            // Exiting linked mode: restore previous fractal
            if let Some(saved) = self.saved_fractal.take() {
                self.current_fractal = saved;
                self.camera.center = self.current_fractal.default_center();
                self.camera.zoom = self.current_fractal.default_zoom();
                self.camera.rotation = 0.0;
            }
            log::info!("Linked mode disabled");
        }
    }

    fn render_frame(&mut self) {
        // Handle deferred linked mode toggle before borrowing fields
        if self.pending_toggle_linked {
            self.pending_toggle_linked = false;
            self.toggle_linked_mode();
        }
        if let Some(fractal) = self.pending_fractal.take() {
            self.pick_fractal(fractal);
        }

        // Keep the view inside the current fractal's bounds
        self.camera.set_bounds(self.current_fractal.view_bounds());
        if self.linked_mode {
            self.julia_camera.set_bounds(FractalType::Julia { c: Vec2::ZERO }.view_bounds());
        }

        // Handle pending canvas resize (mobile fix button)
        if self.pending_resize_canvas {
            self.pending_resize_canvas = false;
            if let Some(ref window) = self.window {
                #[cfg(target_arch = "wasm32")]
                {
                    use winit::platform::web::WindowExtWebSys;
                    let canvas = window.canvas().unwrap();
                    let dpr = web_sys::window().map(|w| w.device_pixel_ratio()).unwrap_or(1.0);
                    let pixel_w = (canvas.client_width() as f64 * dpr) as u32;
                    let pixel_h = (canvas.client_height() as f64 * dpr) as u32;
                    if pixel_w > 0 && pixel_h > 0 {
                        canvas.set_width(pixel_w);
                        canvas.set_height(pixel_h);
                        self.resize(winit::dpi::PhysicalSize::new(pixel_w, pixel_h));
                        log::info!("Canvas resized to {}x{} (client: {}x{})", pixel_w, pixel_h, canvas.client_width(), canvas.client_height());
                    }
                }
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let size = window.inner_size();
                    if size.width > 0 && size.height > 0 {
                        self.resize(size);
                    }
                }
            }
        }

        // GPU init is async on wasm: nothing to draw until `AppEvent::GpuReady` arrives
        if self.gpu.is_none() {
            return;
        }

        // On wasm, auto-resize after GPU init. The countdown starts when
        // GPU becomes ready and fires the resize after ~60 frames (~1s),
        // giving the mobile browser time to settle its viewport.
        #[cfg(target_arch = "wasm32")]
        {
            if self.resize_countdown > 0 {
                self.resize_countdown -= 1;
                if self.resize_countdown == 0 {
                    self.pending_resize_canvas = true;
                }
            }
        }

        let Some(ref window) = self.window else { return };
        let Some(ref gpu) = self.gpu else { return };
        let Some(ref mut compute) = self.compute else { return };
        let Some(ref render) = self.render else { return };
        let Some(ref mut ui) = self.ui else { return };

        // Begin egui frame
        let raw_input = ui.begin_frame(window);
        let mut panel_action = PanelAction::None;
        let mut editor_palette_result: Option<coloring::Palette> = None;
        let full_output = ui.egui_ctx.run(raw_input, |ctx| {
            // Show control panel
            let using_f64 = compute.using_f64;
            panel_action = ControlPanel::show(
                ctx,
                &mut self.current_fractal,
                &mut self.current_color,
                &mut self.max_iter,
                self.fps,
                self.camera.center,
                self.camera.zoom,
                self.camera.rotation,
                using_f64,
                self.linked_mode,
                self.linked_julia_c,
                &mut self.recording_state,
            );

            // Show palette editor if open
            editor_palette_result = self.palette_editor.show(ctx);

            // Fractal picker, over everything else
            if let Some(fractal) = self.fractal_picker.show(ctx, &self.current_fractal) {
                self.pending_fractal = Some(fractal);
            }
        });

        // Handle panel actions
        match panel_action {
            PanelAction::Export(resolution) => {
                self.pending_export = Some(resolution);
            }
            PanelAction::OpenPaletteEditor => {
                self.palette_editor.open(&self.current_color);
            }
            PanelAction::SelectPreset(idx) => {
                self.current_color = ColorScheme::Preset(idx);
                self.palette_dirty = true;
            }
            PanelAction::ToggleLinkedMode => {
                self.pending_toggle_linked = true;
            }
            PanelAction::OpenFractalPicker => {
                self.fractal_picker.open();
            }
            PanelAction::StartRecording(settings) => {
                self.pending_record = Some(settings);
            }
            PanelAction::ResizeCanvas => {
                self.pending_resize_canvas = true;
            }
            PanelAction::None => {}
        }

        // Handle palette editor result
        if let Some(palette) = editor_palette_result {
            self.current_color = ColorScheme::Custom(palette);
            self.palette_dirty = true;
        }

        let is_buddhabrot = self.current_fractal.is_buddhabrot();

        // Detect fractal type change
        let current_id = self.current_fractal.id();
        if current_id != self.prev_fractal_id {
            if is_buddhabrot {
                self.buddhabrot_dirty = true;
            }
            // Start every fractal from its own default view (the keyboard shortcuts
            // already did; the panel used to keep the previous fractal's position and
            // zoom, which usually lands deep inside or far outside the new one).
            // Linked mode manages its own cameras.
            if !self.linked_mode {
                self.camera.center = self.current_fractal.default_center();
                self.camera.zoom = self.current_fractal.default_zoom();
            }
            self.prev_fractal_id = current_id;
        }

        let view = View {
            center: self.camera.center,
            zoom: self.camera.zoom,
            rotation: self.camera.rotation,
            max_iter: self.max_iter,
        };
        let julia_view = View {
            center: self.julia_camera.center,
            zoom: self.julia_camera.zoom,
            rotation: self.julia_camera.rotation,
            max_iter: self.max_iter,
        };

        // Skip the fractal compute pass when nothing it depends on has changed.
        // Buddhabrot accumulates progressively, so it always runs.
        let view_key = ViewKey {
            fractal: self.current_fractal.clone(),
            view,
            size: (gpu.surface_config.width, gpu.surface_config.height),
            linked: self.linked_mode.then_some((julia_view, self.linked_julia_c)),
        };
        let needs_compute = is_buddhabrot
            || self.palette_dirty
            || self.rendered_view.as_ref() != Some(&view_key);

        // Upload palette LUT if dirty
        let palette_changed = self.palette_dirty;
        if self.palette_dirty {
            let palette = self.current_color.get_palette();
            let lut = palette.generate_lut();
            compute.upload_palette(&gpu.queue, &lut);
            self.palette_dirty = false;
        }

        // Handle Buddhabrot dirty state (clear accumulation buffer on view change)
        if is_buddhabrot && self.buddhabrot_dirty {
            compute.clear_accum_buffer(&gpu.queue);
            self.buddhabrot_seed = 0;
            self.buddhabrot_dirty = false;
        }

        // Get surface texture
        let surface_texture = match gpu.get_current_texture() {
            Ok(texture) => texture,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                // Reconfigure surface on next frame
                return;
            }
            Err(e) => {
                log::error!("Failed to get surface texture: {:?}", e);
                return;
            }
        };

        let surface_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Create command encoder
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame Command Encoder"),
            });

        // Compute the fractal. Buddhabrot uses its own two-pass accumulation pipeline;
        // everything else is one per-pixel pass that picks its precision from the zoom.
        if is_buddhabrot && !self.linked_mode {
            compute.render_buddhabrot(&gpu.device, &gpu.queue, &mut encoder, &view, self.buddhabrot_seed);
            // Increment seed for next frame's random samples
            self.buddhabrot_seed = self.buddhabrot_seed.wrapping_add(1);
        } else if needs_compute {
            compute.render_main(&gpu.device, &gpu.queue, &mut encoder, &self.current_fractal, &view);
        }

        // In linked mode, also compute the Julia side
        if self.linked_mode && needs_compute {
            compute.render_julia(&gpu.device, &gpu.queue, &mut encoder, self.linked_julia_c, &julia_view);
        }

        // Keep the fractal picker's previews current while it is on screen
        let previews_stale = self.fractal_picker.take_stale();
        if self.fractal_picker.is_open() {
            compute.render_thumbnails(&gpu.device, &gpu.queue, &mut encoder, previews_stale || palette_changed);
        }

        // Render fractal to surface
        if self.linked_mode {
            render.render_split(
                &mut encoder,
                &surface_view,
                gpu.surface_config.width,
                gpu.surface_config.height,
            );
        } else {
            render.render(&mut encoder, &surface_view);
        }

        // Prepare egui for rendering
        let screen_descriptor = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [gpu.surface_config.width, gpu.surface_config.height],
            pixels_per_point: window.scale_factor() as f32,
        };

        // Update egui textures
        for (id, image_delta) in &full_output.textures_delta.set {
            ui.egui_renderer.update_texture(
                &gpu.device,
                &gpu.queue,
                *id,
                image_delta,
            );
        }

        // Tessellate egui primitives
        let primitives = ui.tessellate(&full_output);

        // Update egui buffers
        ui.egui_renderer.update_buffers(
            &gpu.device,
            &gpu.queue,
            &mut encoder,
            &primitives,
            &screen_descriptor,
        );

        // Render egui
        {
            let mut egui_rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Egui Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &surface_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Load existing fractal content
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            }).forget_lifetime();

            ui.egui_renderer.render(&mut egui_rpass, &primitives, &screen_descriptor);
        }

        // Submit commands
        gpu.queue.submit(std::iter::once(encoder.finish()));
        self.rendered_view = Some(view_key);

        // Cleanup egui resources
        ui.finish(window, full_output);

        surface_texture.present();

        // Process pending video recording
        if let Some(settings) = self.pending_record.take() {
            self.recording_state.is_recording = true;
            let lut = self.current_color.get_palette().generate_lut();
            match export::record_video(&gpu.device, &gpu.queue, &self.current_fractal, &view, &settings, &lut) {
                Ok(filename) => log::info!("Video saved: {}", filename),
                Err(e) => log::error!("Video recording failed: {}", e),
            }
            self.recording_state.is_recording = false;
        }

        // Process pending export
        if let Some(resolution) = self.pending_export.take() {
            let filename = export::generate_filename(self.current_fractal.name(), &resolution);
            let lut = self.current_color.get_palette().generate_lut();
            let result =
                export::export_png(&gpu.device, &gpu.queue, &self.current_fractal, &view, resolution, &filename, &lut);
            if let Err(e) = result {
                log::error!("Export failed: {}", e);
            }
        }

        // Update FPS counter
        self.frame_count += 1;
        let elapsed = self.last_frame_time.elapsed();
        if elapsed.as_secs_f32() >= 1.0 {
            self.fps = self.frame_count as f32 / elapsed.as_secs_f32();
            log::info!(
                "FPS: {:.1} | Center: ({:.6}, {:.6}) | Zoom: {:.2e} | Max Iter: {}",
                self.fps,
                self.camera.center.x,
                self.camera.center.y,
                self.camera.zoom,
                self.max_iter
            );
            self.frame_count = 0;
            self.last_frame_time = FrameInstant::now();
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        // Textures are recreated below, so the fractal must be recomputed
        self.rendered_view = None;

        // Update cameras
        self.camera
            .resize(UVec2::new(new_size.width, new_size.height));
        self.julia_camera
            .resize(UVec2::new(new_size.width, new_size.height));

        // Resize GPU surface
        if let Some(ref mut gpu) = self.gpu {
            gpu.resize(new_size.width, new_size.height);

            // Recreate compute pipeline textures
            if let Some(ref mut compute) = self.compute {
                compute.resize(&gpu.device, new_size.width, new_size.height);

                // Update render pipeline bind groups
                if let Some(ref mut render) = self.render {
                    render.update_texture(&gpu.device, &compute.main.view);
                    render.update_julia_texture(&gpu.device, &compute.julia.view);
                }
            }
        }

        log::info!("Window resized to {}x{}", new_size.width, new_size.height);
    }

    /// Determine which viewport the mouse is in (for linked mode)
    fn get_active_viewport(&self) -> ActiveViewport {
        if !self.linked_mode {
            return ActiveViewport::Left; // Single view, treat as left
        }
        let Some(ref gpu) = self.gpu else { return ActiveViewport::None };
        let half_width = gpu.surface_config.width as f32 / 2.0;
        if self.mouse_pos.x < half_width {
            ActiveViewport::Left
        } else {
            ActiveViewport::Right
        }
    }

    /// Get the camera for the active viewport in linked mode, converting mouse to half-screen coords
    fn linked_screen_pos(&self) -> Vec2 {
        let Some(ref gpu) = self.gpu else { return self.mouse_pos };
        let half_width = gpu.surface_config.width as f32 / 2.0;
        if self.mouse_pos.x < half_width {
            // Left side: x range [0, half_width) maps to screen_to_complex with half-width screen
            self.mouse_pos
        } else {
            // Right side: x range [half_width, width) maps to [0, half_width)
            Vec2::new(self.mouse_pos.x - half_width, self.mouse_pos.y)
        }
    }

    fn handle_mouse_input(&mut self, button: MouseButton, state: ElementState) {
        match button {
            MouseButton::Left => {
                self.mouse_pressed = state == ElementState::Pressed;
                if self.mouse_pressed {
                    self.last_mouse_pos = Some(self.mouse_pos);
                    self.active_viewport = self.get_active_viewport();
                } else {
                    self.last_mouse_pos = None;
                    self.active_viewport = ActiveViewport::None;
                }
            }
            MouseButton::Right if state == ElementState::Pressed => {
                // Right-click sets Julia parameter to current mouse position
                if let FractalType::Julia { ref mut c } = self.current_fractal {
                    let complex_pos = self.camera.screen_to_complex(self.mouse_pos);
                    *c = Vec2::new(complex_pos.x as f32, complex_pos.y as f32);
                    log::info!(
                        "Julia parameter updated: c = ({:.6}, {:.6})",
                        c.x,
                        c.y
                    );
                }
            }
            _ => {}
        }
    }

    fn handle_cursor_moved(&mut self, position: Vec2) {
        self.mouse_pos = position;

        if self.linked_mode {
            // In linked mode, update julia_c when hovering over the left (Mandelbrot) side
            let viewport = self.get_active_viewport();
            if viewport == ActiveViewport::Left && !self.mouse_pressed {
                // Map mouse position to complex coordinates on the Mandelbrot side
                let complex_pos = self.camera.screen_to_complex(self.mouse_pos);
                self.linked_julia_c = Vec2::new(complex_pos.x as f32, complex_pos.y as f32);
            }

            // Pan the appropriate camera when dragging
            if self.mouse_pressed {
                if let Some(last_pos) = self.last_mouse_pos {
                    let delta = position - last_pos;
                    match self.active_viewport {
                        ActiveViewport::Left => self.camera.pan(delta),
                        ActiveViewport::Right => self.julia_camera.pan(delta),
                        ActiveViewport::None => {}
                    }
                    self.last_mouse_pos = Some(position);
                }
            }
        } else {
            // Normal mode: pan camera if mouse is pressed
            if self.mouse_pressed {
                if let Some(last_pos) = self.last_mouse_pos {
                    let delta = position - last_pos;
                    self.camera.pan(delta);
                    self.last_mouse_pos = Some(position);
                    // Mark Buddhabrot dirty on camera change
                    if self.current_fractal.is_buddhabrot() {
                        self.buddhabrot_dirty = true;
                    }
                }
            }
        }
    }

    fn handle_mouse_wheel(&mut self, delta: MouseScrollDelta) {
        let zoom_factor = match delta {
            MouseScrollDelta::LineDelta(_x, y) => {
                if y > 0.0 {
                    1.2_f64 // Zoom in
                } else {
                    1.0 / 1.2 // Zoom out
                }
            }
            MouseScrollDelta::PixelDelta(pos) => {
                let y = pos.y as f64;
                if y > 0.0 {
                    1.0 + y.abs() * 0.01
                } else {
                    1.0 / (1.0 + y.abs() * 0.01)
                }
            }
        };

        if self.linked_mode {
            let viewport = self.get_active_viewport();
            match viewport {
                ActiveViewport::Left => {
                    self.camera.zoom_at(self.mouse_pos, zoom_factor);
                }
                ActiveViewport::Right => {
                    let adjusted_pos = self.linked_screen_pos();
                    self.julia_camera.zoom_at(adjusted_pos, zoom_factor);
                }
                ActiveViewport::None => {}
            }
        } else {
            self.camera.zoom_at(self.mouse_pos, zoom_factor);
            // Mark Buddhabrot dirty on camera change
            if self.current_fractal.is_buddhabrot() {
                self.buddhabrot_dirty = true;
            }
        }
    }

    fn handle_keyboard(&mut self, event: KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }

        // Space opens and closes the fractal picker
        if event.logical_key == Key::Named(NamedKey::Space) {
            if !event.repeat {
                self.fractal_picker.toggle();
            }
            return;
        }
        // While the picker is open the keyboard belongs to it: a card's letter
        // chooses that fractal, and no other shortcut acts on the hidden view
        if self.fractal_picker.is_open() {
            if let Key::Character(ref c) = event.logical_key {
                if let Some(fractal) = fractal_for_letter(c) {
                    self.fractal_picker.close();
                    self.pick_fractal(fractal);
                }
            }
            return;
        }

        match event.logical_key {
            // Switch fractal: key n selects the n-th entry of FractalType::ALL
            Key::Character(ref c) if fractal_for_key(c).is_some() => {
                if let Some(fractal) = fractal_for_key(c) {
                    self.select_fractal(fractal);
                }
            }
            // Reset view
            Key::Character(ref c) if c == "r" || c == "R" => {
                self.camera.center = self.current_fractal.default_center();
                self.camera.zoom = self.current_fractal.default_zoom();
                self.camera.rotation = 0.0;
                self.max_iter = 256;
                log::info!("View reset for: {}", self.current_fractal.name());
            }
            // Increase iterations (+/=)
            Key::Character(ref c) if c == "=" || c == "+" => {
                self.max_iter = (self.max_iter + 64).min(MAX_ITERATIONS);
                log::info!("Max iterations: {}", self.max_iter);
            }
            // Decrease iterations (-)
            Key::Character(ref c) if c == "-" || c == "_" => {
                self.max_iter = self.max_iter.saturating_sub(64).max(MIN_ITERATIONS);
                log::info!("Max iterations: {}", self.max_iter);
            }
            // Pan with arrow keys
            Key::Named(NamedKey::ArrowUp) => {
                self.camera.pan(Vec2::new(0.0, 80.0));
                if self.current_fractal.is_buddhabrot() {
                    self.buddhabrot_dirty = true;
                }
            }
            Key::Named(NamedKey::ArrowDown) => {
                self.camera.pan(Vec2::new(0.0, -80.0));
                if self.current_fractal.is_buddhabrot() {
                    self.buddhabrot_dirty = true;
                }
            }
            // Cycle color scheme
            Key::Character(ref c) if c == "c" || c == "C" => {
                self.current_color = self.current_color.next();
                self.palette_dirty = true;
                log::info!("Color scheme: {}", self.current_color.name());
            }
            // Pan down with S
            Key::Character(ref c) if c == "s" || c == "S" => {
                self.camera.pan(Vec2::new(0.0, -80.0));
                if self.current_fractal.is_buddhabrot() {
                    self.buddhabrot_dirty = true;
                }
            }
            // Screenshot at 1080p
            Key::Character(ref c) if c == "p" || c == "P" => {
                self.pending_export = Some(ExportResolution::HD1080p);
                log::info!("Screenshot requested (1080p)");
            }
            // Rotate view: E = anticlockwise
            Key::Character(ref c) if c == "e" || c == "E" => {
                self.camera.rotation -= 0.05;
                log::info!("Rotation: {:.1}°", self.camera.rotation.to_degrees());
                if self.current_fractal.is_buddhabrot() {
                    self.buddhabrot_dirty = true;
                }
            }
            // Pan with WASD
            Key::Character(ref c) if c == "w" || c == "W" => {
                self.camera.pan(Vec2::new(0.0, 80.0));
                if self.current_fractal.is_buddhabrot() {
                    self.buddhabrot_dirty = true;
                }
            }
            Key::Character(ref c) if c == "a" || c == "A" => {
                self.camera.pan(Vec2::new(80.0, 0.0));
                if self.current_fractal.is_buddhabrot() {
                    self.buddhabrot_dirty = true;
                }
            }
            Key::Named(NamedKey::ArrowLeft) => {
                self.camera.pan(Vec2::new(80.0, 0.0));
                if self.current_fractal.is_buddhabrot() {
                    self.buddhabrot_dirty = true;
                }
            }
            Key::Named(NamedKey::ArrowRight) => {
                self.camera.pan(Vec2::new(-80.0, 0.0));
                if self.current_fractal.is_buddhabrot() {
                    self.buddhabrot_dirty = true;
                }
            }
            Key::Character(ref c) if c == "d" || c == "D" => {
                self.camera.pan(Vec2::new(-80.0, 0.0));
                if self.current_fractal.is_buddhabrot() {
                    self.buddhabrot_dirty = true;
                }
            }
            // Rotate view: Q = clockwise
            Key::Character(ref c) if c == "q" || c == "Q" => {
                self.camera.rotation += 0.05;
                log::info!("Rotation: {:.1}°", self.camera.rotation.to_degrees());
                if self.current_fractal.is_buddhabrot() {
                    self.buddhabrot_dirty = true;
                }
            }
            // Zoom in/out with T/G (centered on screen)
            Key::Character(ref c) if c == "t" || c == "T" => {
                let screen_center = self.camera.screen_size.as_vec2() / 2.0;
                self.camera.zoom_at(screen_center, 1.5);
                if self.current_fractal.is_buddhabrot() {
                    self.buddhabrot_dirty = true;
                }
            }
            Key::Character(ref c) if c == "g" || c == "G" => {
                let screen_center = self.camera.screen_size.as_vec2() / 2.0;
                self.camera.zoom_at(screen_center, 1.0 / 1.5);
                if self.current_fractal.is_buddhabrot() {
                    self.buddhabrot_dirty = true;
                }
            }
            // Zoom out with Z
            Key::Character(ref c) if c == "z" || c == "Z" => {
                let screen_center = self.camera.screen_size.as_vec2() / 2.0;
                self.camera.zoom_at(screen_center, 1.0 / 1.5);
                if self.current_fractal.is_buddhabrot() {
                    self.buddhabrot_dirty = true;
                }
            }
            // Julia/Nova c parameter: J/L for c_real, I/K for c_imag
            Key::Character(ref c) if c == "j" || c == "J" => self.nudge_c(Vec2::new(-0.01, 0.0)),
            Key::Character(ref c) if c == "i" || c == "I" => self.nudge_c(Vec2::new(0.0, 0.01)),
            Key::Character(ref c) if c == "k" || c == "K" => self.nudge_c(Vec2::new(0.0, -0.01)),
            Key::Character(ref c) if c == "l" || c == "L" => {
                // With a c parameter to adjust, L moves it; otherwise it toggles linked mode
                if !self.linked_mode && self.current_fractal.c_mut().is_some() {
                    self.nudge_c(Vec2::new(0.01, 0.0));
                } else {
                    self.toggle_linked_mode();
                }
            }
            _ => {}
        }
    }

    fn is_touch_over_ui(&self, pos: Vec2) -> bool {
        if let Some(ref ui) = self.ui {
            // Check if touch position is inside any egui area by querying layer rects
            let egui_pos = egui::pos2(pos.x, pos.y);
            ui.egui_ctx.available_rect(); // ensure ctx is up to date
            // Use the panel width: if the touch is within the panel region, it's over UI
            let panel_rect = ui.egui_ctx.memory(|mem| {
                mem.area_rect(egui::Id::new("control_panel"))
            });
            if let Some(rect) = panel_rect {
                rect.contains(egui_pos)
            } else {
                false
            }
        } else {
            false
        }
    }

    fn handle_touch(&mut self, touch: winit::event::Touch) {
        use winit::event::TouchPhase;

        let pos = Vec2::new(touch.location.x as f32, touch.location.y as f32);

        match touch.phase {
            TouchPhase::Started => {
                self.touches.insert(touch.id, pos);
            }
            TouchPhase::Moved => {
                let prev_pos = self.touches.get(&touch.id).copied();
                self.touches.insert(touch.id, pos);

                if self.touches.len() == 1 {
                    // Single finger: pan
                    if let Some(prev) = prev_pos {
                        let delta = pos - prev;
                        self.camera.pan(delta);
                        if self.current_fractal.is_buddhabrot() {
                            self.buddhabrot_dirty = true;
                        }
                    }
                } else if self.touches.len() == 2 {
                    // Two fingers: pinch to zoom
                    let positions: Vec<Vec2> = self.touches.values().copied().collect();
                    let distance = (positions[0] - positions[1]).length();

                    if let Some(last_distance) = self.last_pinch_distance {
                        if last_distance > 0.0 {
                            let zoom_factor = (distance / last_distance) as f64;
                            let midpoint = (positions[0] + positions[1]) / 2.0;
                            self.camera.zoom_at(midpoint, zoom_factor);
                            if self.current_fractal.is_buddhabrot() {
                                self.buddhabrot_dirty = true;
                            }
                        }
                    }
                    self.last_pinch_distance = Some(distance);
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.touches.remove(&touch.id);
                if self.touches.len() < 2 {
                    self.last_pinch_distance = None;
                }
            }
        }
    }
}

impl ApplicationHandler<AppEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Create window
        let window_attributes = WindowAttributes::default()
            .with_title("Fractal Explorer - Mandelbrot Set")
            .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080));

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        // On wasm, attach canvas to document body and set pixel dimensions.
        // Use screen.availHeight as a better estimate on mobile (not affected
        // by browser chrome), with window.innerHeight as fallback.
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::WindowExtWebSys;
            let canvas = window.canvas().expect("Failed to get canvas");
            // Set CSS style to fill viewport
            let style = canvas.style();
            let _ = style.set_property("width", "100%");
            let _ = style.set_property("height", "100dvh");
            let _ = style.set_property("display", "block");
            web_sys::window()
                .and_then(|win| win.document())
                .and_then(|doc| doc.body())
                .map(|body| body.append_child(&canvas).ok());

            // Set canvas pixel buffer. Use screen dimensions on mobile for a
            // better initial estimate (screen.availWidth/Height aren't affected
            // by the browser chrome). The per-frame check will correct later.
            if let Some(win) = web_sys::window() {
                let dpr = win.device_pixel_ratio();
                let screen = win.screen().ok();
                let css_w = screen.as_ref()
                    .and_then(|s| s.avail_width().ok())
                    .map(|v| v as f64)
                    .unwrap_or_else(|| win.inner_width().ok().and_then(|v| v.as_f64()).unwrap_or(1920.0));
                let css_h = screen.as_ref()
                    .and_then(|s| s.avail_height().ok())
                    .map(|v| v as f64)
                    .unwrap_or_else(|| win.inner_height().ok().and_then(|v| v.as_f64()).unwrap_or(1080.0));
                let pixel_w = (css_w * dpr) as u32;
                let pixel_h = (css_h * dpr) as u32;
                canvas.set_width(pixel_w);
                canvas.set_height(pixel_h);
                let _ = window.request_inner_size(winit::dpi::PhysicalSize::new(pixel_w, pixel_h));
            }
        }

        let size = window.inner_size();
        // On wasm, inner_size may still be 0 before layout; use a fallback
        let (sw, sh) = if size.width == 0 || size.height == 0 {
            (1920, 1080)
        } else {
            (size.width, size.height)
        };

        // Initialize camera
        self.camera.resize(UVec2::new(sw, sh));

        self.window = Some(window.clone());

        // Initialize GPU. Native can simply block on it; the browser cannot, so there
        // the result comes back to the event loop as an `AppEvent::GpuReady`.
        #[cfg(not(target_arch = "wasm32"))]
        {
            let result = pollster::block_on(Self::init_gpu(window));
            self.user_event(event_loop, AppEvent::GpuReady(result.map(Box::new)));
        }

        #[cfg(target_arch = "wasm32")]
        {
            let proxy = self.proxy.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let result = App::init_gpu(window).await;
                // Fails only if the event loop is already gone
                let _ = proxy.send_event(AppEvent::GpuReady(result.map(Box::new)));
            });
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::GpuReady(result) => self.on_gpu_ready(event_loop, result),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: winit::window::WindowId, event: WindowEvent) {
        // Let egui handle the event first
        let egui_consumed = if let (Some(ref window), Some(ref mut ui)) = (&self.window, &mut self.ui) {
            ui.handle_event(window, &event)
        } else {
            false
        };

        // Handle RedrawRequested regardless
        if let WindowEvent::RedrawRequested = event {
            self.render_frame();
            if let Some(ref window) = self.window {
                window.request_redraw();
            }
            return;
        }

        // Always handle touch events for mobile pan/zoom, even if egui consumed them
        if let WindowEvent::Touch(touch) = &event {
            let touch_pos = Vec2::new(touch.location.x as f32, touch.location.y as f32);
            if !self.is_touch_over_ui(touch_pos) {
                self.handle_touch(*touch);
            }
        }

        // If egui consumed the event, don't pass to app
        if egui_consumed {
            return;
        }

        match event {
            // Escape closes the fractal picker first; only then does it exit
            WindowEvent::KeyboardInput {
                event: KeyEvent { logical_key: Key::Named(NamedKey::Escape), state: ElementState::Pressed, .. },
                ..
            } if self.fractal_picker.is_open() => {
                self.fractal_picker.close();
            }

            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                log::info!("Exiting...");
                event_loop.exit();
            }

            WindowEvent::Resized(new_size) => {
                self.resize(new_size);
            }

            WindowEvent::MouseInput { button, state, .. } => {
                self.handle_mouse_input(button, state);
            }

            WindowEvent::CursorMoved { position, .. } => {
                self.handle_cursor_moved(Vec2::new(position.x as f32, position.y as f32));
            }

            WindowEvent::MouseWheel { delta, .. } => {
                self.handle_mouse_wheel(delta);
            }

            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard(event);
            }

            // Touch is handled above (before egui consumed check)

            WindowEvent::RedrawRequested => {
                self.render_frame();
                if let Some(ref window) = self.window {
                    window.request_redraw();
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if self.gpu_failed {
            return;
        }
        if let Some(ref window) = self.window {
            window.request_redraw();
        }
    }
}

/// The fractal a number key selects ("1" = first entry of `FractalType::ALL`)
fn fractal_for_key(key: &str) -> Option<FractalType> {
    let n: usize = key.parse().ok()?;
    FractalType::ALL.get(n.checked_sub(1)?).cloned()
}

/// Tell the user about an error the app cannot recover from.
///
/// Browser: fills and reveals the `#fatal-error` overlay from index.html — otherwise
/// all the user would see is a black canvas. Native: prints to stderr.
fn show_fatal_error(message: &str) {
    #[cfg(target_arch = "wasm32")]
    {
        let overlay = web_sys::window()
            .and_then(|window| window.document())
            .and_then(|document| document.get_element_by_id("fatal-error"));
        match overlay {
            Some(overlay) => {
                if let Some(text) = overlay.query_selector(".message").ok().flatten() {
                    text.set_text_content(Some(message));
                }
                let _ = overlay.remove_attribute("hidden");
            }
            None => log::error!("No #fatal-error element to show: {message}"),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("Fractal Explorer cannot start: {message}");
}

fn main() {
    // Initialize logger (platform-specific)
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::Builder::from_default_env()
            .filter_level(log::LevelFilter::Warn)
            .filter_module("fractal_explorer", log::LevelFilter::Info)
            .init();
    }

    #[cfg(target_arch = "wasm32")]
    {
        // A panic leaves a dead canvas behind: log it, then say so on the page
        std::panic::set_hook(Box::new(|info| {
            console_error_panic_hook::hook(info);
            show_fatal_error("Fractal Explorer hit an unexpected error and stopped. Reload the page to try again.");
        }));
        console_log::init_with_level(log::Level::Info)
            .expect("Failed to initialize console_log");
    }

    log::info!("Starting Fractal Explorer...");

    // Create event loop
    let event_loop = EventLoop::<AppEvent>::with_user_event().build().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let app = App::new(event_loop.create_proxy());

    // Create app and run event loop (platform-specific)
    #[cfg(not(target_arch = "wasm32"))]
    {
        let mut app = app;
        event_loop.run_app(&mut app).unwrap();
    }

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::EventLoopExtWebSys;
        event_loop.spawn_app(app);
    }
}
