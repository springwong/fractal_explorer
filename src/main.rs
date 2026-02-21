mod camera;
mod coloring;
mod export;
mod fractals;
mod renderer;
mod ui;

use camera::Camera;
use coloring::ColorScheme;
use fractals::FractalType;
use glam::{UVec2, Vec2};
use renderer::{ComputePipeline, FractalUniforms, GpuContext, RenderPipeline, ds_split, compute_reference_orbit, compute_reference_orbit_julia};
use export::ExportResolution;
use ui::{ControlPanel, PanelAction, UiContext};
use std::sync::Arc;
use winit::{
    application::ApplicationHandler,
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

/// Main application state
struct App<'window> {
    window: Option<Arc<Window>>,
    gpu: Option<GpuContext<'window>>,
    camera: Camera,
    compute: Option<ComputePipeline>,
    render: Option<RenderPipeline>,
    ui: Option<UiContext>,

    // Input state
    mouse_pos: Vec2,
    mouse_pressed: bool,
    last_mouse_pos: Option<Vec2>,

    // Fractal parameters
    max_iter: u32,
    current_fractal: FractalType,
    current_color: ColorScheme,

    // Export state
    pending_export: Option<ExportResolution>,
    last_uniforms: FractalUniforms,

    // Buddhabrot state
    buddhabrot_seed: u32,
    buddhabrot_dirty: bool,
    prev_fractal_type_id: u32,

    // FPS tracking
    last_frame_time: std::time::Instant,
    frame_count: u32,
    fps: f32,
}

impl<'window> App<'window> {
    fn new() -> Self {
        Self {
            window: None,
            gpu: None,
            camera: Camera::new(UVec2::new(1920, 1080)),
            compute: None,
            render: None,
            ui: None,
            mouse_pos: Vec2::ZERO,
            mouse_pressed: false,
            last_mouse_pos: None,
            max_iter: 256,
            current_fractal: FractalType::Mandelbrot,
            current_color: ColorScheme::default(),
            pending_export: None,
            last_uniforms: FractalUniforms::new([0.0; 2], 1.0, 1.0, 256, 0, 0, 0.0, 0.0, [0.0; 2], 0.0, 0.0, 0.0, 0, 0.0),
            buddhabrot_seed: 0,
            buddhabrot_dirty: true,
            prev_fractal_type_id: 0,
            last_frame_time: std::time::Instant::now(),
            frame_count: 0,
            fps: 0.0,
        }
    }

    fn render_frame(&mut self) {
        let Some(ref window) = self.window else { return };
        let Some(ref gpu) = self.gpu else { return };
        let Some(ref mut compute) = self.compute else { return };
        let Some(ref render) = self.render else { return };
        let Some(ref mut ui) = self.ui else { return };

        // Begin egui frame
        let raw_input = ui.begin_frame(window);
        let mut panel_action = PanelAction::None;
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
            );
        });

        // Handle panel export action
        if let PanelAction::Export(resolution) = panel_action {
            self.pending_export = Some(resolution);
        }

        // Get fractal parameters
        let params = self.current_fractal.params();
        let is_buddhabrot = self.current_fractal.is_buddhabrot();

        // Detect fractal type change
        let current_type_id = self.current_fractal.type_id();
        if current_type_id != self.prev_fractal_type_id {
            if is_buddhabrot {
                self.buddhabrot_dirty = true;
            }
            self.prev_fractal_type_id = current_type_id;
        }

        // Split f64 camera values into hi+lo f32 pairs for emulated double precision
        let (center_x_hi, center_x_lo) = ds_split(self.camera.center.x);
        let (center_y_hi, center_y_lo) = ds_split(self.camera.center.y);
        let (zoom_hi, zoom_lo) = ds_split(self.camera.zoom);

        // Compute per-pixel step on CPU in f64 for precision
        let screen_height = gpu.surface_config.height as f64;
        let screen_width = gpu.surface_config.width as f64;
        let pixel_step_x;
        let pixel_step_y;

        if is_buddhabrot {
            // For Buddhabrot, repurpose pixel_step_x/y to pass screen dimensions
            // The shader recovers dimensions via: width = u32(abs(1.0 / pixel_step_x))
            pixel_step_x = 1.0 / screen_width as f32;
            pixel_step_y = 1.0 / screen_height as f32;
        } else {
            pixel_step_x = (1.0 / (self.camera.zoom * screen_height)) as f32;
            pixel_step_y = (-1.0 / (self.camera.zoom * screen_height)) as f32;
        }

        // Compute reference orbit for perturbation (Mandelbrot & Julia, when using f64)
        let use_f64 = self.camera.zoom >= 5.0e3;
        let fractal_id = self.current_fractal.type_id();
        let ref_escape_iter = if is_buddhabrot {
            // For Buddhabrot, use ref_escape_iter as frame seed
            self.buddhabrot_seed
        } else if use_f64 && fractal_id == 0 {
            // Mandelbrot: z_{n+1} = z_n^2 + c, z_0 = 0, c = center
            let (orbit_data, escape_iter) = compute_reference_orbit(
                self.camera.center.x,
                self.camera.center.y,
                self.max_iter,
            );
            compute.upload_orbit(&gpu.device, &gpu.queue, &orbit_data);
            escape_iter
        } else if use_f64 && fractal_id == 1 {
            // Julia: z_{n+1} = z_n^2 + c, z_0 = center, c = fixed param
            let (orbit_data, escape_iter) = compute_reference_orbit_julia(
                self.camera.center.x,
                self.camera.center.y,
                params.c_real as f64,
                params.c_imag as f64,
                self.max_iter,
            );
            compute.upload_orbit(&gpu.device, &gpu.queue, &orbit_data);
            escape_iter
        } else {
            self.max_iter
        };

        // Update uniforms
        let uniforms = FractalUniforms::new(
            [center_x_hi, center_y_hi],
            zoom_hi,
            self.camera.aspect_ratio(),
            self.max_iter,
            self.current_fractal.type_id(),
            self.current_color.to_id(),
            params.c_real,
            params.c_imag,
            [center_x_lo, center_y_lo],
            zoom_lo,
            pixel_step_x,
            pixel_step_y,
            ref_escape_iter,
            self.camera.rotation as f32,
        );
        compute.update_uniforms(&gpu.queue, &uniforms);
        self.last_uniforms = uniforms;

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

        // Dispatch compute shader with current fractal type
        // Buddhabrot uses a separate two-pass accumulation pipeline
        if is_buddhabrot {
            compute.dispatch_buddhabrot(
                &gpu.device,
                &mut encoder,
                gpu.surface_config.width,
                gpu.surface_config.height,
            );
            // Increment seed for next frame's random samples
            self.buddhabrot_seed = self.buddhabrot_seed.wrapping_add(1);
        } else {
            // Auto-switch to f64 emulated precision at high zoom levels
            compute.dispatch(
                &gpu.device,
                &mut encoder,
                gpu.surface_config.width,
                gpu.surface_config.height,
                &self.current_fractal,
                self.camera.zoom,
            );
        }

        // Render fractal to surface
        render.render(&mut encoder, &surface_view);

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

        // Cleanup egui resources
        ui.finish(window, full_output);

        surface_texture.present();

        // Process pending export
        if let Some(resolution) = self.pending_export.take() {
            let filename = export::generate_filename(
                self.current_fractal.name(),
                &resolution,
            );
            let path = std::path::Path::new("export").join(&filename);
            match export::export_png(
                &gpu.device,
                &gpu.queue,
                &self.current_fractal,
                &self.last_uniforms,
                resolution,
                &path,
            ) {
                Ok(()) => log::info!("Export saved: {}", filename),
                Err(e) => log::error!("Export failed: {}", e),
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
            self.last_frame_time = std::time::Instant::now();
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        // Update camera
        self.camera
            .resize(UVec2::new(new_size.width, new_size.height));

        // Resize GPU surface
        if let Some(ref mut gpu) = self.gpu {
            gpu.resize(new_size.width, new_size.height);

            // Recreate compute pipeline textures
            if let Some(ref mut compute) = self.compute {
                compute.resize(&gpu.device, new_size.width, new_size.height);

                // Update render pipeline bind group
                if let Some(ref mut render) = self.render {
                    render.update_texture(&gpu.device, &compute.texture_view);
                }
            }
        }

        log::info!("Window resized to {}x{}", new_size.width, new_size.height);
    }

    fn handle_mouse_input(&mut self, button: MouseButton, state: ElementState) {
        match button {
            MouseButton::Left => {
                self.mouse_pressed = state == ElementState::Pressed;
                if self.mouse_pressed {
                    self.last_mouse_pos = Some(self.mouse_pos);
                } else {
                    self.last_mouse_pos = None;
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

        // Pan camera if mouse is pressed
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

        self.camera.zoom_at(self.mouse_pos, zoom_factor);
        // Mark Buddhabrot dirty on camera change
        if self.current_fractal.is_buddhabrot() {
            self.buddhabrot_dirty = true;
        }
    }

    fn handle_keyboard(&mut self, event: KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }

        match event.logical_key {
            // Switch to Mandelbrot
            Key::Character(ref c) if c == "1" => {
                self.current_fractal = FractalType::Mandelbrot;
                self.camera.center = self.current_fractal.default_center();
                self.camera.zoom = self.current_fractal.default_zoom();
                log::info!("Switched to: {}", self.current_fractal.name());
            }
            // Switch to Julia
            Key::Character(ref c) if c == "2" => {
                self.current_fractal = FractalType::Julia {
                    c: Vec2::new(-0.7, 0.27015),
                };
                self.camera.center = self.current_fractal.default_center();
                self.camera.zoom = self.current_fractal.default_zoom();
                log::info!("Switched to: {}", self.current_fractal.name());
            }
            // Switch to Burning Ship
            Key::Character(ref c) if c == "3" => {
                self.current_fractal = FractalType::BurningShip;
                self.camera.center = self.current_fractal.default_center();
                self.camera.zoom = self.current_fractal.default_zoom();
                log::info!("Switched to: {}", self.current_fractal.name());
            }
            // Switch to Tricorn
            Key::Character(ref c) if c == "4" => {
                self.current_fractal = FractalType::Tricorn;
                self.camera.center = self.current_fractal.default_center();
                self.camera.zoom = self.current_fractal.default_zoom();
                log::info!("Switched to: {}", self.current_fractal.name());
            }
            // Switch to Buddhabrot
            Key::Character(ref c) if c == "5" => {
                self.current_fractal = FractalType::Buddhabrot;
                self.camera.center = self.current_fractal.default_center();
                self.camera.zoom = self.current_fractal.default_zoom();
                log::info!("Switched to: {}", self.current_fractal.name());
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
                self.max_iter = (self.max_iter + 64).min(4096);
                log::info!("Max iterations: {}", self.max_iter);
            }
            // Decrease iterations (-)
            Key::Character(ref c) if c == "-" || c == "_" => {
                self.max_iter = self.max_iter.saturating_sub(64).max(64);
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
            // Julia c parameter: J/L for c_real, I/K for c_imag
            Key::Character(ref c) if c == "j" || c == "J" => {
                if let FractalType::Julia { ref mut c } = self.current_fractal {
                    c.x -= 0.01;
                    log::info!("Julia c = ({:.4}, {:.4})", c.x, c.y);
                }
            }
            Key::Character(ref c) if c == "l" || c == "L" => {
                if let FractalType::Julia { ref mut c } = self.current_fractal {
                    c.x += 0.01;
                    log::info!("Julia c = ({:.4}, {:.4})", c.x, c.y);
                }
            }
            Key::Character(ref c) if c == "i" || c == "I" => {
                if let FractalType::Julia { ref mut c } = self.current_fractal {
                    c.y += 0.01;
                    log::info!("Julia c = ({:.4}, {:.4})", c.x, c.y);
                }
            }
            Key::Character(ref c) if c == "k" || c == "K" => {
                if let FractalType::Julia { ref mut c } = self.current_fractal {
                    c.y -= 0.01;
                    log::info!("Julia c = ({:.4}, {:.4})", c.x, c.y);
                }
            }
            _ => {}
        }
    }
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Create window
        let window_attributes = WindowAttributes::default()
            .with_title("Fractal Explorer - Mandelbrot Set")
            .with_inner_size(winit::dpi::PhysicalSize::new(1920, 1080));

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        let size = window.inner_size();

        // Initialize camera
        self.camera.resize(UVec2::new(size.width, size.height));

        // Initialize GPU
        let gpu = pollster::block_on(GpuContext::new(window.clone()));

        // Create compute pipeline
        let compute = ComputePipeline::new(&gpu.device, size.width, size.height);

        // Create render pipeline
        let render = RenderPipeline::new(&gpu.device, gpu.surface_config.format, &compute.texture_view);

        // Initialize egui
        let ui = UiContext::new(&gpu.device, gpu.surface_config.format, &window);

        self.ui = Some(ui);
        self.window = Some(window);
        self.gpu = Some(gpu);
        self.compute = Some(compute);
        self.render = Some(render);

        log::info!("Application initialized successfully");
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

        // If egui consumed the event, don't pass to app
        if egui_consumed {
            return;
        }

        match event {
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
        if let Some(ref window) = self.window {
            window.request_redraw();
        }
    }
}

fn main() {
    // Initialize logger
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn) // Only warnings and errors
        .filter_module("fractal_explorer", log::LevelFilter::Info) // Our app's info logs
        .init();

    log::info!("Starting Fractal Explorer...");

    // Create event loop
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    // Create app
    let mut app = App::new();

    // Run event loop
    event_loop.run_app(&mut app).unwrap();
}
