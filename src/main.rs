extern crate ggez;

mod apt;
mod imgui_wrapper;
mod pic;
mod stack_machine;

use crate::imgui_wrapper::ImGuiWrapper;
use crate::pic::*;
use ggez::conf;
use ggez::event::{self, EventHandler, KeyCode, KeyMods, MouseButton};
use ggez::graphics;
use ggez::nalgebra as na;
use ggez::timer;
use ggez::{Context, GameResult};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use simdeez::avx2::*;
use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;
use std::time::Instant;

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;
const VIDEO_DURATION: f32 = 5000.0; //milliseconds

const THUMB_ROWS: u16 = 4;
const THUMB_COLS: u16 = 5;

const TREE_MIN: usize = 2;
const TREE_MAX: usize = 20;

struct MainState {
    pos_x: f32,
    imgui_wrapper: ImGuiWrapper,
    hidpi_factor: f32,
    thumbs: Vec<graphics::Image>,
    dt: std::time::Duration,
    frame_elapsed: f32,
    rng: StdRng,
}

impl MainState {
    fn gen_population(&mut self, ctx: &mut Context) {
        let now = Instant::now();
        self.thumbs.clear();
        for _ in 0..THUMB_ROWS * THUMB_COLS {
            let pic_type = self.rng.gen_range(0, 3);
            let pic : Box< dyn Pic<Avx2>> = 
                match pic_type {
                    0 => {
                        Box::new(MonoPic::new(TREE_MIN, TREE_MAX, false, &mut self.rng))
                    }
                    1 => {
                        Box::new(RgbPic::new(TREE_MIN, TREE_MAX, false, &mut self.rng))
                    }
                    2 => {
                        Box::new(HsvPic::new(TREE_MIN, TREE_MAX, false, &mut self.rng))
                    }
                    _ => panic!("invalid"),
                };
            let img = graphics::Image::from_rgba8(
                ctx,
                256 as u16,
                256 as u16,
                &pic.get_rgba8(256, 256, 0.0)[0..],
            ).unwrap();
            self.thumbs.push(img);
        }
       
               
        
        println!("genpop elapsed:{}", now.elapsed().as_millis());
    }

    fn new(mut ctx: &mut Context, hidpi_factor: f32) -> GameResult<MainState> {
        let imgui_wrapper = ImGuiWrapper::new(&mut ctx);

        let s = MainState {
            pos_x: 0.0,
            imgui_wrapper,
            hidpi_factor,
            thumbs: Vec::new(),
            dt: std::time::Duration::new(0, 0),
            frame_elapsed: 0.0,
            rng: StdRng::from_rng(rand::thread_rng()).unwrap(),
        };
        Ok(s)
    }
}

impl EventHandler for MainState {
    fn update(&mut self, ctx: &mut Context) -> GameResult<()> {
        self.dt = timer::delta(ctx);
        self.frame_elapsed = (self.frame_elapsed + self.dt.as_millis() as f32) % VIDEO_DURATION;
        self.pos_x = self.pos_x % 800.0 + 1.0;
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, graphics::BLACK);

        let mut y_pos = 10.0;
        for y in 0..THUMB_ROWS {
            let mut x_pos = 10.0;
            for x in 0..THUMB_COLS {
                let index = (y * THUMB_COLS + x) as usize;
                let img = &self.thumbs[index];

                graphics::draw(
                    ctx,
                    img,
                    graphics::DrawParam::new().dest(na::Point2::new(x_pos, y_pos)),
                );
                x_pos += 256.0 + 10.0;
            }
            y_pos += 256.0 + 10.0;
        }

        // Render game ui
        {
            self.imgui_wrapper.render(ctx, self.hidpi_factor);
        }

        graphics::present(ctx)?;
        Ok(())
    }

    fn mouse_motion_event(&mut self, _ctx: &mut Context, x: f32, y: f32, _dx: f32, _dy: f32) {
        self.imgui_wrapper.update_mouse_pos(x, y);
    }

    fn mouse_button_down_event(
        &mut self,
        _ctx: &mut Context,
        button: MouseButton,
        _x: f32,
        _y: f32,
    ) {
        self.imgui_wrapper.update_mouse_down((
            button == MouseButton::Left,
            button == MouseButton::Right,
            button == MouseButton::Middle,
        ));
    }

    fn mouse_button_up_event(
        &mut self,
        _ctx: &mut Context,
        _button: MouseButton,
        _x: f32,
        _y: f32,
    ) {
        self.imgui_wrapper.update_mouse_down((false, false, false));
    }

    fn key_down_event(
        &mut self,
        _ctx: &mut Context,
        keycode: KeyCode,
        _keymods: KeyMods,
        _repeat: bool,
    ) {
        match keycode {
            KeyCode::P => (),
            _ => (),
        }
    }

    fn text_input_event(&mut self, _ctx: &mut Context, ch: char) {
        self.imgui_wrapper.update_keyboard(ch);
    }
}

pub fn main() -> ggez::GameResult {
    rayon::ThreadPoolBuilder::new()
        .num_threads(0)
        .build_global()
        .unwrap();
    let hidpi_factor: f32;
    {
        // Create a dummy window so we can get monitor scaling information
        let cb = ggez::ContextBuilder::new("", "");
        let (_ctx, events_loop) = &mut cb.build()?;
        hidpi_factor = events_loop.get_primary_monitor().get_hidpi_factor() as f32;
        println!("main hidpi_factor = {}", hidpi_factor);
    }

    let cb = ggez::ContextBuilder::new("super_simple with imgui", "ggez")
        .window_setup(conf::WindowSetup::default().title("super_simple with imgui"))
        .window_mode(
            conf::WindowMode::default().dimensions(WIDTH as f32 * 1.0, HEIGHT as f32 * 1.0),
        );
    let (ref mut ctx, event_loop) = &mut cb.build()?;

    let state = &mut MainState::new(ctx, hidpi_factor)?;
    state.gen_population(ctx);
    event::run(ctx, event_loop, state)
}
