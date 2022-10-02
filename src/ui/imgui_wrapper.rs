use std::time::Instant;

use gfx_core::{format, handle::RenderTargetView, memory::Typed};
use gfx_device_gl::Resources;
use ggez::graphics::{drawable_size, gfx_objects};
use ggez::Context;
use imgui::{Condition, ImString, Window};
use imgui_gfx_renderer::{Renderer, Shaders};

pub const EXEC_NAME: &'static str = "Evolution";

use crate::ui::mouseactionstate::MouseActionState;

#[readonly::make]
pub struct ImGuiWrapper {
    pub imgui: imgui::Context,
    pub renderer: Renderer<format::Rgba8, Resources>,
    last_frame: Instant,
    mouse_state: MouseActionState,
    text_multiline: ImString,
}

impl ImGuiWrapper {
    pub fn new(ctx: &mut Context) -> Self {
        // Create the imgui object
        let mut imgui = imgui::Context::create();
        let (factory, gfx_device, _, _, _) = gfx_objects(ctx);

        // Shaders
        let shaders = {
            let version = gfx_device.get_info().shading_language;
            if version.is_embedded {
                if version.major >= 3 {
                    Shaders::GlSlEs300
                } else {
                    Shaders::GlSlEs100
                }
            } else if version.major >= 4 {
                Shaders::GlSl400
            } else if version.major >= 3 {
                Shaders::GlSl130
            } else {
                Shaders::GlSl110
            }
        };

        // Renderer
        let renderer = Renderer::init(&mut imgui, &mut *factory, shaders).unwrap();

        // Create instace
        Self {
            imgui,
            renderer,
            last_frame: Instant::now(),
            mouse_state: MouseActionState::default(),
            text_multiline: ImString::with_capacity(1024),
        }
    }

    pub fn render(&mut self, ctx: &mut Context, hidpi_factor: f32) {
        // Update mouse
        self.update_mouse();

        // Create new frame
        let now = Instant::now();
        let delta = now - self.last_frame;
        let delta_s = delta.as_secs() as f32 + delta.subsec_nanos() as f32 / 1_000_000_000.0;
        self.last_frame = now;

        let (draw_width, draw_height) = drawable_size(ctx);
        self.imgui.io_mut().display_size = [draw_width, draw_height];
        self.imgui.io_mut().display_framebuffer_scale = [hidpi_factor, hidpi_factor];
        self.imgui.io_mut().delta_time = delta_s;

        let ui = self.imgui.frame();
        let t = &mut self.text_multiline;
        Window::new(EXEC_NAME)
            .size([300.0, 100.0], Condition::FirstUseEver)
            .build(&ui, || {
                ui.text(format!("This...is... {}!", EXEC_NAME));
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
                let mut s = t.to_str().to_owned();
                ui.input_text_multiline("multiline", &mut s, [300., 100.])
                    .build();
            });

        // Render
        let (factory, _, encoder, _, render_target) = gfx_objects(ctx);
        let draw_data = ui.render();
        self.renderer
            .render(
                &mut *factory,
                encoder,
                &mut RenderTargetView::new(render_target.clone()),
                draw_data,
            )
            .unwrap();
    }

    pub fn update_keyboard(&mut self, ch: char) {
        self.imgui.io_mut().add_input_character(ch);
    }

    fn update_mouse(&mut self) {
        self.imgui.io_mut().mouse_pos =
            [self.mouse_state.pos.0 as f32, self.mouse_state.pos.1 as f32];

        self.imgui.io_mut().mouse_down = [
            self.mouse_state.pressed.0,
            self.mouse_state.pressed.1,
            self.mouse_state.pressed.2,
            false,
            false,
        ];

        self.imgui.io_mut().mouse_wheel = self.mouse_state.wheel;
        self.mouse_state.wheel = 0.0;
    }

    pub fn update_mouse_pos(&mut self, x: f32, y: f32) {
        self.mouse_state.pos = (x as i32, y as i32);
    }

    pub fn update_mouse_down(&mut self, pressed: (bool, bool, bool)) {
        self.mouse_state.pressed = pressed;
    }
}
