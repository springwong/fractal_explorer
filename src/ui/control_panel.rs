use crate::coloring::{ColorScheme, PresetPalettes};
use crate::export::{ExportResolution, VideoSettings};
use crate::fractals::FractalType;
use glam::Vec2;

/// Action requested by the control panel
#[derive(Clone, Debug)]
pub enum PanelAction {
    None,
    ResizeCanvas,
    Export(ExportResolution),
    OpenPaletteEditor,
    SelectPreset(usize),
    ToggleLinkedMode,
    StartRecording(VideoSettings),
}

/// Persistent state for the video recording UI section
pub struct RecordingState {
    pub target_zoom_exp: f64,  // log10 of target zoom
    pub duration_secs: f64,
    pub fps: u32,
    pub resolution: ExportResolution,
    pub is_recording: bool,
}

impl Default for RecordingState {
    fn default() -> Self {
        Self {
            target_zoom_exp: 6.0, // 1e6
            duration_secs: 5.0,
            fps: 30,
            resolution: ExportResolution::HD1080p,
            is_recording: false,
        }
    }
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
        linked_mode: bool,
        linked_julia_c: Vec2,
        recording_state: &mut RecordingState,
    ) -> PanelAction {
        let mut action = PanelAction::None;

        egui::SidePanel::left("control_panel")
            .default_width(280.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.heading("Fractal Explorer");
                    if ui.button("⟳").on_hover_text("Fix canvas size").clicked() {
                        action = PanelAction::ResizeCanvas;
                    }
                });
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

                // Linked mode toggle
                ui.heading("Linked View");
                if ui.checkbox(&mut { linked_mode }, "Mandelbrot / Julia Split").changed() {
                    action = PanelAction::ToggleLinkedMode;
                }

                if linked_mode {
                    ui.label(format!("Julia c: ({:.4}, {:.4})", linked_julia_c.x, linked_julia_c.y));
                    ui.small("Hover Mandelbrot to update Julia c");
                    ui.separator();
                }

                // Fractal type selector (disabled in linked mode)
                if !linked_mode {
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

                // Record Video section
                ui.heading("Record Video");
                if recording_state.is_recording {
                    ui.colored_label(egui::Color32::YELLOW, "Recording in progress...");
                    ui.small("App will freeze until done.");
                } else {
                    ui.add(
                        egui::Slider::new(&mut recording_state.target_zoom_exp, 2.0..=12.0)
                            .text("Target Zoom")
                            .custom_formatter(|v, _| format!("1e{:.0}", v))
                    );

                    ui.add(
                        egui::Slider::new(&mut recording_state.duration_secs, 2.0..=60.0)
                            .text("Duration (s)")
                    );

                    egui::ComboBox::from_label("FPS")
                        .selected_text(format!("{}", recording_state.fps))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut recording_state.fps, 30, "30");
                            ui.selectable_value(&mut recording_state.fps, 60, "60");
                        });

                    egui::ComboBox::from_label("Resolution")
                        .selected_text(match recording_state.resolution {
                            ExportResolution::HD1080p => "1080p",
                            ExportResolution::UHD4K => "4K",
                            _ => "1080p",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut recording_state.resolution, ExportResolution::HD1080p, "1080p");
                            ui.selectable_value(&mut recording_state.resolution, ExportResolution::UHD4K, "4K");
                        });

                    if ui.button("Start Recording").clicked() {
                        let settings = VideoSettings {
                            resolution: recording_state.resolution,
                            fps: recording_state.fps,
                            duration_secs: recording_state.duration_secs,
                            target_zoom: 10.0f64.powf(recording_state.target_zoom_exp),
                        };
                        action = PanelAction::StartRecording(settings);
                    }
                    ui.small("Requires ffmpeg installed");
                }

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
                ui.label("L: Toggle linked view");
                ui.label("Esc: Exit");

                ui.separator();

                ui.small("Phase 3 Implementation");
                }); // ScrollArea
            });

        action
    }
}
