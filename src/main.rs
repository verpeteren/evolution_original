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
use ggez::{Context, GameResult};
use simdeez::avx2::*;
use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use ggez::timer;

const WIDTH: usize = 800;
const HEIGHT: usize = 800;
const DURATION: f32 = 5000.0;

struct MainState {
    pos_x: f32,
    imgui_wrapper: ImGuiWrapper,
    hidpi_factor: f32,
    //img1: graphics::Image,
    video: Vec<graphics::Image>,
    dt: std::time::Duration,
    frame_elapsed: f32
    //  img2: graphics::Image,
}

impl MainState {
    fn new(mut ctx: &mut Context, hidpi_factor: f32) -> GameResult<MainState> {
        let imgui_wrapper = ImGuiWrapper::new(&mut ctx);
        let pic = HsvPic::new(15);
        println!("{}",pic.to_lisp());
        let img1 = graphics::Image::from_rgba8(
            ctx,
            WIDTH as u16,
            HEIGHT as u16,
            &pic.get_rgba8::<Sse2>(WIDTH, HEIGHT,0.0)[0..],
        )
        .unwrap();

        let mut frames = Vec::new();
        let video_data = &pic.get_video::<Sse2>(WIDTH,HEIGHT,32,DURATION);
        for frame in video_data {
            frames.push(graphics::Image::from_rgba8(
                ctx,
                WIDTH as u16,
                HEIGHT as u16,
                &frame[0..],
            )
            .unwrap())
        }


        let s = MainState {
            pos_x: 0.0,
            imgui_wrapper,
            hidpi_factor,
            video:frames,
            dt: std::time::Duration::new(0, 0),
            frame_elapsed: 0.0
            //img1,
            // img2,
        };
        Ok(s)
    }
}

impl EventHandler for MainState {
    fn update(&mut self, ctx: &mut Context) -> GameResult<()> {
        self.dt = timer::delta(ctx);
        self.frame_elapsed = (self.frame_elapsed + self.dt.as_millis() as f32) % DURATION;
        self.pos_x = self.pos_x % 800.0 + 1.0;
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, graphics::BLACK);
        
        let pct = self.frame_elapsed / DURATION;
        //todo dont be an idiot this is wrong
        let mut frame = (pct * self.video.len() as f32) as usize;
        if frame == self.video.len() { frame = self.video.len()-1}
      //  println!("frame:{} pct:{}",frame,pct);
        let _ = graphics::draw(
            ctx,
            &self.video[frame],
            graphics::DrawParam::default().scale(na::Vector2::new(1.0, 1.0)),
        );
        //let _ = graphics::draw(ctx, &self.img2, graphics::DrawParam::default().dest(na::Point2::new(SIZE as f32,0.0)));
        // Render game stuff
        {
            let circle = graphics::Mesh::new_circle(
                ctx,
                graphics::DrawMode::fill(),
                na::Point2::new(self.pos_x, 380.0),
                100.0,
                2.0,
                graphics::WHITE,
            )?;
            graphics::draw(ctx, &circle, (na::Point2::new(0.0, 0.0),))?;
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
    rayon::ThreadPoolBuilder::new().num_threads(0).build_global().unwrap();
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

    event::run(ctx, event_loop, state)
}
