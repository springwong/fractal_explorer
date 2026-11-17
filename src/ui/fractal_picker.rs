//! The fractal picker: a full-window overlay showing every fractal as a big picture
//! card with a letter on it. Opened with Space; a fractal is chosen by clicking its
//! card or pressing its letter.

use crate::fractals::FractalType;
use egui::{Align2, Color32, FontId, Pos2, Rect, Rounding, Sense, Stroke, Vec2};

/// Card picture size in points (same 16:10 shape as the preview textures)
const IMAGE_SIZE: Vec2 = Vec2::new(240.0, 150.0);
/// Height of the name strip under the picture
const NAME_HEIGHT: f32 = 40.0;
/// Space between cards, and around the whole grid
const GAP: f32 = 18.0;
/// Height of the title area above the grid
const HEADER_HEIGHT: f32 = 76.0;
/// Most cards in one row
const MAX_COLUMNS: usize = 4;
/// Diameter of the letter badge
const BADGE_SIZE: f32 = 56.0;

/// Bright badge colours, cycled through card by card
const BADGE_COLORS: [Color32; 7] = [
    Color32::from_rgb(239, 71, 71),
    Color32::from_rgb(255, 159, 28),
    Color32::from_rgb(255, 209, 37),
    Color32::from_rgb(76, 201, 104),
    Color32::from_rgb(56, 176, 240),
    Color32::from_rgb(150, 110, 245),
    Color32::from_rgb(244, 114, 182),
];

/// The letter that selects entry `index` of `FractalType::ALL` ('A' for the first)
pub fn letter_for_index(index: usize) -> Option<char> {
    (index < 26).then(|| (b'A' + index as u8) as char)
}

/// The fractal a letter key selects in the picker ("a" or "A" = first entry of
/// `FractalType::ALL`)
pub fn fractal_for_letter(key: &str) -> Option<FractalType> {
    let mut chars = key.chars();
    let (c, None) = (chars.next()?, chars.next()) else { return None };
    let c = c.to_ascii_uppercase();
    if !c.is_ascii_uppercase() {
        return None;
    }
    FractalType::ALL.get((c as u8 - b'A') as usize).cloned()
}

/// State of the fractal picker overlay
#[derive(Default)]
pub struct FractalPicker {
    open: bool,
    /// The previews must be redrawn before they are next shown
    stale: bool,
    /// Preview image of each entry of `FractalType::ALL`
    thumbnails: Vec<egui::TextureId>,
}

impl FractalPicker {
    /// Set the preview image of each entry of `FractalType::ALL`
    pub fn set_thumbnails(&mut self, thumbnails: Vec<egui::TextureId>) {
        self.thumbnails = thumbnails;
    }

    /// Whether the picker is on screen
    pub fn is_open(&self) -> bool {
        self.open
    }

    /// Show the picker
    pub fn open(&mut self) {
        self.open = true;
        self.stale = true;
    }

    /// Hide the picker without choosing anything
    pub fn close(&mut self) {
        self.open = false;
    }

    /// Show the picker if it is hidden, hide it if it is shown
    pub fn toggle(&mut self) {
        if self.open {
            self.close();
        } else {
            self.open();
        }
    }

    /// Whether the previews need redrawing (true once after each `open`)
    pub fn take_stale(&mut self) -> bool {
        std::mem::take(&mut self.stale)
    }

    /// Draw the picker if it is open. Returns the fractal whose card was clicked;
    /// the picker closes itself on a choice and on a click outside the cards.
    pub fn show(&mut self, ctx: &egui::Context, current: &FractalType) -> Option<FractalType> {
        if !self.open {
            return None;
        }
        let screen = ctx.screen_rect();
        let count = FractalType::ALL.len();

        // As many columns as fit, then shrink everything if it is still too big
        let card = Vec2::new(IMAGE_SIZE.x, IMAGE_SIZE.y + NAME_HEIGHT);
        let fit = ((screen.width() - GAP) / (card.x + GAP)).floor() as usize;
        let columns = fit.clamp(1, MAX_COLUMNS).min(count);
        let rows = count.div_ceil(columns);
        let full = Vec2::new(
            columns as f32 * (card.x + GAP) + GAP,
            HEADER_HEIGHT + rows as f32 * (card.y + GAP) + GAP,
        );
        let scale = (screen.width() / full.x).min(screen.height() / full.y).min(1.0);
        let (card, gap) = (card * scale, GAP * scale);
        let panel = Rect::from_center_size(screen.center(), full * scale);

        let mut chosen = None;
        let mut dismissed = false;
        egui::Area::new(egui::Id::new("fractal_picker"))
            .order(egui::Order::Foreground)
            .fixed_pos(screen.min)
            .show(ctx, |ui| {
                // Dim everything behind; a click there closes the picker
                let backdrop = ui.allocate_rect(screen, Sense::click());
                dismissed = backdrop.clicked();
                let painter = ui.painter();
                painter.rect_filled(screen, Rounding::ZERO, Color32::from_black_alpha(190));

                painter.text(
                    Pos2::new(panel.center().x, panel.top() + gap + 16.0 * scale),
                    Align2::CENTER_CENTER,
                    "Pick a fractal",
                    FontId::proportional(34.0 * scale),
                    Color32::WHITE,
                );
                painter.text(
                    Pos2::new(panel.center().x, panel.top() + gap + 46.0 * scale),
                    Align2::CENTER_CENTER,
                    "Press its letter or click its picture   ·   Space to go back",
                    FontId::proportional(16.0 * scale),
                    Color32::from_gray(190),
                );

                for (index, fractal) in FractalType::ALL.iter().enumerate() {
                    let (row, column) = (index / columns, index % columns);
                    // Centre a last row that is not full
                    let in_row = if row + 1 == rows { count - row * columns } else { columns };
                    let row_width = in_row as f32 * (card.x + gap) - gap;
                    let min = Pos2::new(
                        panel.center().x - row_width / 2.0 + column as f32 * (card.x + gap),
                        panel.top() + (HEADER_HEIGHT * scale) + gap + row as f32 * (card.y + gap),
                    );
                    let rect = Rect::from_min_size(min, card);
                    let response = ui
                        .interact(rect, ui.id().with(index), Sense::click())
                        .on_hover_cursor(egui::CursorIcon::PointingHand);
                    if response.clicked() {
                        chosen = Some(fractal.clone());
                    }
                    let is_current = fractal.id() == current.id();
                    self.paint_card(ui.painter(), rect, index, fractal, scale, response.hovered(), is_current);
                }
            });

        if chosen.is_some() || dismissed {
            self.close();
        }
        chosen
    }

    /// Paint one card: picture, letter badge, name
    #[allow(clippy::too_many_arguments)]
    fn paint_card(
        &self,
        painter: &egui::Painter,
        rect: Rect,
        index: usize,
        fractal: &FractalType,
        scale: f32,
        hovered: bool,
        is_current: bool,
    ) {
        let color = BADGE_COLORS[index % BADGE_COLORS.len()];
        let rounding = Rounding::same(14.0 * scale);
        // A hovered card grows a little
        let rect = if hovered { rect.expand(5.0 * scale) } else { rect };
        let image_rect = Rect::from_min_size(rect.min, Vec2::new(rect.width(), rect.height() - NAME_HEIGHT * scale));

        painter.rect_filled(rect, rounding, Color32::from_gray(28));
        if let Some(&texture) = self.thumbnails.get(index) {
            let uv = Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0));
            let top = Rounding { sw: 0.0, se: 0.0, ..rounding };
            painter.add(egui::epaint::RectShape {
                rect: image_rect,
                rounding: top,
                fill: Color32::WHITE,
                stroke: Stroke::NONE,
                blur_width: 0.0,
                fill_texture_id: texture,
                uv,
            });
        }

        // Name strip
        painter.text(
            Pos2::new(rect.center().x, (image_rect.bottom() + rect.bottom()) / 2.0),
            Align2::CENTER_CENTER,
            fractal.name(),
            FontId::proportional(20.0 * scale),
            Color32::WHITE,
        );

        // Outline: the card's own colour when hovered, white for the fractal on screen
        let outline = if hovered {
            Stroke::new(5.0 * scale, color)
        } else if is_current {
            Stroke::new(4.0 * scale, Color32::WHITE)
        } else {
            Stroke::new(1.5 * scale, Color32::from_gray(90))
        };
        painter.rect_stroke(rect, rounding, outline);

        // Letter badge, overlapping the top-left corner
        if let Some(letter) = letter_for_index(index) {
            let radius = BADGE_SIZE * scale / 2.0;
            let center = rect.min + Vec2::splat(radius * 0.75);
            painter.circle_filled(center + Vec2::splat(2.5 * scale), radius, Color32::from_black_alpha(140));
            painter.circle(center, radius, color, Stroke::new(4.0 * scale, Color32::WHITE));
            let font = FontId::proportional(38.0 * scale);
            let text_pos = center + Vec2::new(0.0, 1.0 * scale);
            // Dark outline so the letter reads on every badge colour
            for offset in [Vec2::new(1.5, 1.5), Vec2::new(-1.0, 1.5), Vec2::new(1.5, -1.0)] {
                painter.text(
                    text_pos + offset * scale,
                    Align2::CENTER_CENTER,
                    letter,
                    font.clone(),
                    Color32::from_black_alpha(150),
                );
            }
            painter.text(text_pos, Align2::CENTER_CENTER, letter, font, Color32::WHITE);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every fractal has its own letter, and the letter selects that fractal.
    #[test]
    fn letters_select_their_fractal() {
        for (index, fractal) in FractalType::ALL.iter().enumerate() {
            let letter = letter_for_index(index).expect("more fractals than letters");
            let upper = letter.to_string();
            let lower = letter.to_ascii_lowercase().to_string();
            assert_eq!(fractal_for_letter(&upper).map(|f| f.id()), Some(fractal.id()));
            assert_eq!(fractal_for_letter(&lower).map(|f| f.id()), Some(fractal.id()));
        }
    }

    /// Keys that are not a card's letter select nothing.
    #[test]
    fn other_keys_select_nothing() {
        let unused = letter_for_index(FractalType::ALL.len()).map(|c| c.to_string());
        for key in ["1", " ", "", "ab", "é", "["].into_iter().chain(unused.as_deref()) {
            assert!(fractal_for_letter(key).is_none(), "{key:?}");
        }
    }
}
