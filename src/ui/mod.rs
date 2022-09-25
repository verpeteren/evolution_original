use ggez::error::GameResult;
use ggez::input::mouse::MouseButton;
use ggez::graphics::{draw, size, DrawParam, Image, Rect};
use ggez::mint::{Point2, Vector2};
use ggez::Context;

mod mouseactionstate;
mod mousebuttonstate;
pub mod imgui_wrapper;

pub use mousebuttonstate::MouseButtonState;

pub enum MouseState {
    Up(MouseButtonState),
    Down(MouseButtonState),
    Nothing,
}

pub struct Button {
    img: Image,
    pct_rect: Rect,
}

impl Button {
    pub fn new(img: Image, pct_x: f32, pct_y: f32, pct_w: f32, pct_h: f32) -> Button {
        Button {
            img: img,
            pct_rect: Rect {
                x: pct_x,
                y: pct_y,
                w: pct_w,
                h: pct_h,
            },
        }
    }

    fn pixel_rect(&self, ctx: &mut Context) -> Rect {
        let (w, h) = size(ctx);
        Rect {
            x: self.pct_rect.x * w,
            y: self.pct_rect.y * h,
            w: self.pct_rect.w * w,
            h: self.pct_rect.h * h,
        }
    }

    pub fn left_clicked(&self, ctx: &mut Context, mouse_state: &MouseState) -> bool {
        match mouse_state {
            MouseState::Up(button_state) => {
                button_state.which_button == MouseButton::Left
                    && self.pixel_rect(ctx).contains(Point2 {
                        x: button_state.x,
                        y: button_state.y,
                    })
            }
            _ => false,
        }
    }

    pub fn right_clicked(&self, ctx: &mut Context, mouse_state: &MouseState) -> bool {
        match mouse_state {
            MouseState::Up(button_state) => {
                button_state.which_button == MouseButton::Right
                    && self.pixel_rect(ctx).contains(Point2 {
                        x: button_state.x,
                        y: button_state.y,
                    })
            }
            _ => false,
        }
    }

    pub fn draw(&self, ctx: &mut Context) {
        let pixel_rect = self.pixel_rect(ctx);
        let x_scale = pixel_rect.w / self.img.width() as f32;
        let y_scale = pixel_rect.h / self.img.height() as f32;
        let params = DrawParam::new()
            .dest(Point2 {
                x: pixel_rect.x,
                y: pixel_rect.y,
            })
            .scale(Vector2 {
                x: x_scale,
                y: y_scale,
            });
        let _ = draw(ctx, &self.img, params);
    }
    pub fn pic_bytes(&self, ctx: &mut Context) -> GameResult<Vec<u8>> {
        self.img.to_rgba8(ctx)
    }
}
