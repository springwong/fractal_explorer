use crate::coloring::{ColorScheme, PresetPalettes};
use crate::export::ExportResolution;
use crate::fractals::FractalType;
use glam::Vec2;

/// Action requested by the control panel
#[derive(Clone, Debug)]
pub enum PanelAction {
    None,
    Export(ExportResolution),
    OpenPaletteEditor,
    SelectPreset(usize),
}

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
        rotation: f64,
        using_f64: bool,
    ) -> PanelAction {
        let mut action = PanelAction::None;

        egui::SidePanel::left("control_panel")
            .default_width(280.0)
            .show(ctx, |ui| {
                ui.heading("Fractal Explorer");
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

                ui.radio_value(&mut *fractal, FractalType::Mandelbrot, "Mandelbrot Set");
                ui.radio_value(&mut *fractal, FractalType::Julia { c: Vec2::new(-0.7, 0.27015) }, "Julia Set");
                ui.radio_value(&mut *fractal, FractalType::BurningShip, "Burning Ship");
                ui.radio_value(&mut *fractal, FractalType::Tricorn, "Tricorn");
                ui.radio_value(&mut *fractal, FractalType::Buddhabrot, "Buddhabrot");
                ui.radio_value(&mut *fractal, FractalType::Nova { c: Vec2::new(1.0, 0.0) }, "Nova Fractal");

                // Julia parameters
                if let FractalType::Julia { ref mut c } = fractal {
                    ui.separator();
                    ui.label("Julia Parameters:");
                    ui.add(egui::Slider::new(&mut c.x, -2.0..=2.0).text("c (real)"));
                    ui.add(egui::Slider::new(&mut c.y, -2.0..=2.0).text("c (imag)"));
                    ui.label("Tip: Right-click to set c");
                }

                // Nova parameters
                if let FractalType::Nova { ref mut c } = fractal {
                    ui.separator();
                    ui.label("Nova Parameters:");
                    ui.add(egui::Slider::new(&mut c.x, -2.0..=2.0).text("c (real)"));
                    ui.add(egui::Slider::new(&mut c.y, -2.0..=2.0).text("c (imag)"));
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

                // Color scheme - preset list
                ui.heading("Color Scheme");
                let presets = PresetPalettes::all();
                let current_preset = color_scheme.preset_index();
                for (i, preset) in presets.iter().enumerate() {
                    let selected = current_preset == Some(i);
                    if ui.selectable_label(selected, &preset.name).clicked() && !selected {
                        action = PanelAction::SelectPreset(i);
                    }
                }

                // Show "Custom" if a custom palette is active
                if let ColorScheme::Custom(ref p) = color_scheme {
                    let _ = ui.selectable_label(true, format!("* {}", p.name));
                }

                if ui.button("Edit Palette...").clicked() {
                    action = PanelAction::OpenPaletteEditor;
                }

                ui.separator();

                // Camera info
                ui.heading("Camera");
                ui.label(format!("Center: ({:.6}, {:.6})", center.x, center.y));
                ui.label(format!("Zoom: {:.2e}", zoom));
                ui.label(format!("Rotation: {:.1}", rotation.to_degrees()));
                ui.horizontal(|ui| {
                    ui.label("Precision:");
                    if using_f64 {
                        ui.colored_label(egui::Color32::from_rgb(255, 200, 50), "f64 (perturbation)");
                    } else {
                        ui.colored_label(egui::Color32::GREEN, "f32");
                    }
                });

                ui.separator();

                // Export section
                ui.heading("Export");
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Save PNG (1080p)").clicked() {
                        action = PanelAction::Export(ExportResolution::HD1080p);
                    }
                    if ui.button("Save 4K PNG").clicked() {
                        action = PanelAction::Export(ExportResolution::UHD4K);
                    }
                    if ui.button("Save 8K PNG").clicked() {
                        action = PanelAction::Export(ExportResolution::UHD8K);
                    }
                });
                ui.small("Keyboard: P = screenshot");

                ui.separator();

                // Keyboard shortcuts
                ui.heading("Keyboard Shortcuts");
                ui.label("1-6: Switch fractal type");
                ui.label("C: Cycle color scheme");
                ui.label("R: Reset view");
                ui.label("Q/E: Rotate left/right");
                ui.label("T/G: Zoom in/out");
                ui.label("J/L: Julia/Nova c real -/+");
                ui.label("I/K: Julia/Nova c imag +/-");
                ui.label("P: Save screenshot (1080p)");
                ui.label("Esc: Exit");

                ui.separator();

                ui.small("Phase 3 Implementation");
            });

        action
    }
}
