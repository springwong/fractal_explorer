use crate::coloring::ColorScheme;
use crate::fractals::FractalType;
use glam::Vec2;

/// Control panel UI state and rendering
pub struct ControlPanel;

impl ControlPanel {
    /// Show the control panel UI
    pub fn show(
        ctx: &egui::Context,
        fractal: &mut FractalType,
        color_scheme: &mut ColorScheme,
        max_iter: &mut u32,
        fps: f32,
        center: glam::DVec2,
        zoom: f64,
    ) -> bool {
        let mut changed = false;

        egui::SidePanel::left("control_panel")
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("🔭 Fractal Explorer");
                ui.separator();

                // FPS display
                ui.horizontal(|ui| {
                    ui.label("FPS:");
                    ui.colored_label(
                        if fps > 55.0 {
                            egui::Color32::GREEN
                        } else if fps > 30.0 {
                            egui::Color32::YELLOW
                        } else {
                            egui::Color32::RED
                        },
                        format!("{:.1}", fps),
                    );
                });

                ui.separator();

                // Fractal type selector
                ui.heading("Fractal Type");
                let old_fractal = fractal.clone();

                if ui.radio_value(&mut *fractal, FractalType::Mandelbrot, "Mandelbrot Set").clicked() {
                    changed = true;
                }
                if ui.radio_value(&mut *fractal, FractalType::Julia { c: Vec2::new(-0.7, 0.27015) }, "Julia Set").clicked() {
                    changed = true;
                }
                if ui.radio_value(&mut *fractal, FractalType::BurningShip, "Burning Ship").clicked() {
                    changed = true;
                }
                if ui.radio_value(&mut *fractal, FractalType::Tricorn, "Tricorn").clicked() {
                    changed = true;
                }

                // Julia parameters
                if let FractalType::Julia { ref mut c } = fractal {
                    ui.separator();
                    ui.label("Julia Parameters:");
                    ui.add(egui::Slider::new(&mut c.x, -2.0..=2.0).text("c (real)"));
                    ui.add(egui::Slider::new(&mut c.y, -2.0..=2.0).text("c (imag)"));
                    ui.label("💡 Tip: Right-click to set c");
                }

                ui.separator();

                // Iterations
                ui.heading("Rendering");
                ui.add(
                    egui::Slider::new(max_iter, 64..=4096)
                        .text("Max Iterations")
                        .logarithmic(true),
                );

                ui.separator();

                // Color scheme
                ui.heading("Color Scheme");
                for scheme in ColorScheme::all() {
                    if ui.radio_value(&mut *color_scheme, *scheme, scheme.name()).clicked() {
                        changed = true;
                    }
                }

                ui.separator();

                // Camera info
                ui.heading("Camera");
                ui.label(format!("Center: ({:.6}, {:.6})", center.x, center.y));
                ui.label(format!("Zoom: {:.2e}", zoom));

                ui.separator();

                // Keyboard shortcuts
                ui.heading("Keyboard Shortcuts");
                ui.label("1-4: Switch fractal type");
                ui.label("C: Cycle color scheme");
                ui.label("R: Reset view");
                ui.label("↑/↓: Adjust iterations");
                ui.label("Esc: Exit");

                ui.separator();

                ui.small("Phase 2 Implementation");
            });

        changed
    }
}
