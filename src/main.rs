mod camera;
mod renderer;

use camera::Camera;
use glam::{UVec2, Vec2};
use renderer::{ComputePipeline, FractalUniforms, GpuContext, RenderPipeline};
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

    // Input state
    mouse_pos: Vec2,
    mouse_pressed: bool,
    last_mouse_pos: Option<Vec2>,

    // Fractal parameters
    max_iter: u32,

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
            mouse_pos: Vec2::ZERO,
            mouse_pressed: false,
            last_mouse_pos: None,
            max_iter: 256,
            last_frame_time: std::time::Instant::now(),
            frame_count: 0,
            fps: 0.0,
        }
    }

    fn render_frame(&mut self) {
        let Some(ref gpu) = self.gpu else { return };
        let Some(ref compute) = self.compute else { return };
        let Some(ref render) = self.render else { return };

        // Update uniforms
        let uniforms = FractalUniforms::new(
            [self.camera.center.x as f32, self.camera.center.y as f32],
            self.camera.zoom as f32,
            self.camera.aspect_ratio(),
            self.max_iter,
        );
        compute.update_uniforms(&gpu.queue, &uniforms);

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

        // Dispatch compute shader
        compute.dispatch(&mut encoder, gpu.surface_config.width, gpu.surface_config.height);

        // Render to surface
        render.render(&mut encoder, &surface_view);

        // Submit commands
        gpu.queue.submit(std::iter::once(encoder.finish()));
        surface_texture.present();

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
        if button == MouseButton::Left {
            self.mouse_pressed = state == ElementState::Pressed;
            if self.mouse_pressed {
                self.last_mouse_pos = Some(self.mouse_pos);
            } else {
                self.last_mouse_pos = None;
            }
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
    }

    fn handle_keyboard(&mut self, event: KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }

        match event.logical_key {
            // Reset view
            Key::Character(ref c) if c == "r" || c == "R" => {
                self.camera.reset();
                self.max_iter = 256;
                log::info!("View reset");
            }
            // Increase iterations
            Key::Named(NamedKey::ArrowUp) => {
                self.max_iter = (self.max_iter + 64).min(4096);
                log::info!("Max iterations: {}", self.max_iter);
            }
            // Decrease iterations
            Key::Named(NamedKey::ArrowDown) => {
                self.max_iter = self.max_iter.saturating_sub(64).max(64);
                log::info!("Max iterations: {}", self.max_iter);
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

        self.window = Some(window);
        self.gpu = Some(gpu);
        self.compute = Some(compute);
        self.render = Some(render);

        log::info!("Application initialized successfully");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: winit::window::WindowId, event: WindowEvent) {
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
