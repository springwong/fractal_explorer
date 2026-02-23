use crate::coloring::{ColorScheme, Palette, PresetPalettes};

/// Palette editor window state
pub struct PaletteEditor {
    /// Whether the editor window is open
    is_open: bool,
    /// The palette currently being edited
    editing_palette: Palette,
    /// Index of the currently selected stop
    selected_stop: Option<usize>,
}

impl PaletteEditor {
    pub fn new() -> Self {
        Self {
            is_open: false,
            editing_palette: PresetPalettes::all()[0].clone(),
            selected_stop: None,
        }
    }

    /// Open the editor with the current color scheme's palette
    pub fn open(&mut self, current: &ColorScheme) {
        self.editing_palette = current.get_palette();
        self.editing_palette.name = "Custom".to_string();
        self.selected_stop = if self.editing_palette.stops.is_empty() { None } else { Some(0) };
        self.is_open = true;
    }

    /// Show the palette editor window. Returns Some(Palette) when user clicks Apply.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<Palette> {
        if !self.is_open {
            return None;
        }

        let mut result = None;
        let mut open = self.is_open;

        egui::Window::new("Palette Editor")
            .open(&mut open)
            .default_width(400.0)
            .resizable(true)
            .show(ctx, |ui| {
                // Preset buttons
                ui.label("Presets:");
                ui.horizontal_wrapped(|ui| {
                    for (i, preset) in PresetPalettes::all().iter().enumerate() {
                        if ui.button(&preset.name).clicked() {
                            self.editing_palette = preset.clone();
                            self.editing_palette.name = "Custom".to_string();
                            self.selected_stop = Some(0);
                        }
                        if i < 6 { ui.separator(); }
                    }
                });

                ui.separator();

                // Gradient preview bar
                ui.label("Preview:");
                let (response, painter) = ui.allocate_painter(
                    egui::vec2(ui.available_width(), 30.0),
                    egui::Sense::click(),
                );
                let rect = response.rect;

                // Draw gradient as segments
                let segments = 256;
                let seg_width = rect.width() / segments as f32;
                for i in 0..segments {
                    let t = i as f32 / (segments - 1) as f32;
                    let color = self.editing_palette.sample_color(t);
                    let x = rect.left() + i as f32 * seg_width;
                    painter.rect_filled(
                        egui::Rect::from_min_size(
                            egui::pos2(x, rect.top()),
                            egui::vec2(seg_width + 1.0, rect.height()),
                        ),
                        0.0,
                        egui::Color32::from_rgb(
                            (color[0] * 255.0) as u8,
                            (color[1] * 255.0) as u8,
                            (color[2] * 255.0) as u8,
                        ),
                    );
                }

                // Draw stop markers
                for (i, stop) in self.editing_palette.stops.iter().enumerate() {
                    let x = rect.left() + stop.position * rect.width();
                    let is_selected = self.selected_stop == Some(i);
                    let radius = if is_selected { 7.0 } else { 5.0 };
                    let stroke_color = if is_selected {
                        egui::Color32::WHITE
                    } else {
                        egui::Color32::GRAY
                    };

                    painter.circle(
                        egui::pos2(x, rect.bottom()),
                        radius,
                        egui::Color32::from_rgb(
                            (stop.color[0] * 255.0) as u8,
                            (stop.color[1] * 255.0) as u8,
                            (stop.color[2] * 255.0) as u8,
                        ),
                        egui::Stroke::new(2.0, stroke_color),
                    );
                }

                // Click on gradient bar to select nearest stop
                if response.clicked() {
                    if let Some(pos) = response.interact_pointer_pos() {
                        let t = ((pos.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
                        // Find nearest stop
                        let mut nearest = 0;
                        let mut nearest_dist = f32::MAX;
                        for (i, stop) in self.editing_palette.stops.iter().enumerate() {
                            let dist = (stop.position - t).abs();
                            if dist < nearest_dist {
                                nearest_dist = dist;
                                nearest = i;
                            }
                        }
                        self.selected_stop = Some(nearest);
                    }
                }

                ui.separator();

                // Stop list
                ui.label("Color Stops:");
                let mut stop_to_remove: Option<usize> = None;
                let num_stops = self.editing_palette.stops.len();

                egui::ScrollArea::vertical().max_height(200.0).show(ui, |ui| {
                    for i in 0..num_stops {
                        let is_selected = self.selected_stop == Some(i);
                        ui.horizontal(|ui| {
                            // Selection indicator
                            if ui.selectable_label(is_selected, format!("#{}", i + 1)).clicked() {
                                self.selected_stop = Some(i);
                            }

                            // Position slider
                            let mut pos = self.editing_palette.stops[i].position;
                            if ui.add(egui::Slider::new(&mut pos, 0.0..=1.0).text("pos").fixed_decimals(3)).changed() {
                                self.editing_palette.stops[i].position = pos;
                            }

                            // Color picker
                            let mut color = self.editing_palette.stops[i].color;
                            let mut rgba = egui::Rgba::from_rgb(color[0], color[1], color[2]);
                            if egui::color_picker::color_edit_button_rgba(ui, &mut rgba, egui::color_picker::Alpha::Opaque).changed() {
                                color = [rgba.r(), rgba.g(), rgba.b()];
                                self.editing_palette.stops[i].color = color;
                            }

                            // Remove button (only if more than 2 stops)
                            if num_stops > 2 {
                                if ui.small_button("X").clicked() {
                                    stop_to_remove = Some(i);
                                }
                            }
                        });
                    }
                });

                // Handle stop removal
                if let Some(idx) = stop_to_remove {
                    self.editing_palette.remove_stop(idx);
                    if let Some(sel) = self.selected_stop {
                        if sel >= self.editing_palette.stops.len() {
                            self.selected_stop = Some(self.editing_palette.stops.len() - 1);
                        }
                    }
                }

                ui.separator();

                // Add / Apply buttons
                ui.horizontal(|ui| {
                    if ui.button("Add Stop").clicked() {
                        // Add a stop at midpoint of selected stop and next, or at 0.5
                        let pos = if let Some(sel) = self.selected_stop {
                            if sel + 1 < self.editing_palette.stops.len() {
                                (self.editing_palette.stops[sel].position
                                    + self.editing_palette.stops[sel + 1].position)
                                    / 2.0
                            } else if sel > 0 {
                                (self.editing_palette.stops[sel - 1].position
                                    + self.editing_palette.stops[sel].position)
                                    / 2.0
                            } else {
                                0.5
                            }
                        } else {
                            0.5
                        };
                        let color = self.editing_palette.sample_color(pos);
                        self.editing_palette.add_stop(pos, color);
                        // Select the new stop
                        for (i, stop) in self.editing_palette.stops.iter().enumerate() {
                            if (stop.position - pos).abs() < 0.001 {
                                self.selected_stop = Some(i);
                                break;
                            }
                        }
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Apply").clicked() {
                            // Sort stops before applying
                            self.editing_palette.stops.sort_by(|a, b| {
                                a.position.partial_cmp(&b.position).unwrap()
                            });
                            result = Some(self.editing_palette.clone());
                        }
                    });
                });
            });

        self.is_open = open;
        result
    }
}
