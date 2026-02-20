/// UI module for egui integration
mod control_panel;

pub use control_panel::ControlPanel;

use egui_winit::State as EguiState;

/// UI rendering state
pub struct UiContext {
    pub egui_ctx: egui::Context,
    pub egui_state: EguiState,
    pub egui_renderer: egui_wgpu::Renderer,  // Made public for direct access
}

impl UiContext {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        window: &winit::window::Window,
    ) -> Self {
        let egui_ctx = egui::Context::default();

        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            None,
            None,
            None,
        );

        let egui_renderer = egui_wgpu::Renderer::new(
            device,
            surface_format,
            None,
            1,
            false,
        );

        Self {
            egui_ctx,
            egui_state,
            egui_renderer,
        }
    }

    /// Handle window event, returns true if event was consumed
    pub fn handle_event(&mut self, window: &winit::window::Window, event: &winit::event::WindowEvent) -> bool {
        let response = self.egui_state.on_window_event(window, event);
        response.consumed
    }

    /// Begin a new UI frame
    pub fn begin_frame(&mut self, window: &winit::window::Window) -> egui::RawInput {
        self.egui_state.take_egui_input(window)
    }

    /// Tessellate egui output into primitives
    pub fn tessellate(&self, full_output: &egui::FullOutput) -> Vec<egui::ClippedPrimitive> {
        self.egui_ctx.tessellate(full_output.shapes.clone(), full_output.pixels_per_point)
    }

    /// Cleanup after rendering
    pub fn finish(&mut self, window: &winit::window::Window, full_output: egui::FullOutput) {
        // Free textures
        for id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }

        // Handle platform output
        self.egui_state.handle_platform_output(window, full_output.platform_output);
    }
}
