extern crate ggez;

mod apt;
mod imgui_wrapper;
mod pic;
mod stack_machine;
mod ui;
mod ggez_utility;

use crate::ui::*;
use crate::imgui_wrapper::ImGuiWrapper;
use crate::pic::*;
use ggez::conf;
use ggez::event::{self, EventHandler, KeyCode, KeyMods, MouseButton};
use ggez::graphics;
use ggez::graphics::Image;
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
use simdeez::*;
use std::time::Instant;

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;
const VIDEO_DURATION: f32 = 5000.0; //milliseconds

const THUMB_ROWS: u16 = 4;
const THUMB_COLS: u16 = 5;

const TREE_MIN: usize = 2;
const TREE_MAX: usize = 15;

enum GameState {
    Select,
    Zoom(Image)
}

struct MainState<S:Simd> {
    state: GameState,
    mouse_state: MouseState,
    imgui_wrapper: ImGuiWrapper,
    hidpi_factor: f32,
    img_buttons: Vec<Button>,
    pics: Vec<Box<dyn Pic<S>>>,
    dt: std::time::Duration,
    frame_elapsed: f32,
    rng: StdRng,        
}

impl<S:Simd> MainState<S> {
    fn gen_population(&mut self, ctx: &mut Context) {

        // todo make this layout code less dumb
        let now = Instant::now();
        self.img_buttons.clear();
        let width = 1.0/(THUMB_COLS as f32 * 1.01);        
        let height = 1.0/(THUMB_ROWS as f32 * 1.01);        
        let mut y_pct = 0.01;
        for _ in 0 .. THUMB_ROWS {
            let mut x_pct = 0.01;
            for _ in 0 .. THUMB_COLS {
                let pic_type = self.rng.gen_range(0, 4);
                //let pic_type = 0;
                let pic: Box<dyn Pic<S>> = match pic_type {
                    0 => Box::new(GradientPic::new(TREE_MIN, TREE_MAX, false, &mut self.rng)),
                    1 => Box::new(MonoPic::new(TREE_MIN, TREE_MAX, false, &mut self.rng)),
                    2 => Box::new(RgbPic::new(TREE_MIN, TREE_MAX, false, &mut self.rng)),
                    3 => Box::new(HsvPic::new(TREE_MIN, TREE_MAX, false, &mut self.rng)),
                    _ => panic!("invalid"),
                };
                let img = graphics::Image::from_rgba8(
                    ctx,
                    256 as u16,
                    256 as u16,
                    &pic.get_rgba8(256, 256, 0.0)[0..],
                )
                .unwrap();                                    
                self.pics.push(pic);                        
                self.img_buttons.push(Button::new(img,x_pct,y_pct,width-0.01,height-0.01));
                x_pct += width;
            }
            println!("--------------------");
            y_pct += height;
        }
        println!("genpop elapsed:{}", now.elapsed().as_millis());
    }

    fn new(mut ctx: &mut Context, hidpi_factor: f32) -> GameResult<MainState<S>> {
        let imgui_wrapper = ImGuiWrapper::new(&mut ctx);

        let s = MainState::<S> {            
            state: GameState::Select,
            imgui_wrapper,
            hidpi_factor,
            pics:Vec::new(),
            img_buttons: Vec::new(),
            dt: std::time::Duration::new(0, 0),
            frame_elapsed: 0.0,
            rng: StdRng::from_rng(rand::thread_rng()).unwrap(),
            mouse_state: MouseState::Nothing,
        };
        Ok(s)
    }

    fn update_select(&mut self, ctx: &mut Context) {
        for (i,img_button) in self.img_buttons.iter().enumerate() {
            if img_button.left_clicked(ctx,&self.mouse_state) {
                println!("{}",self.pics[i].to_lisp());    
                println!("button left clicked");
                break;
            }
            if img_button.right_clicked(ctx,&self.mouse_state) {
                println!("button right clicked");
                let (w,h) = graphics::size(ctx);
                let img = graphics::Image::from_rgba8(
                    ctx,
                    1920 as u16,
                    1080 as u16,
                    &self.pics[i].get_rgba8(1920 as usize, 1080 as usize, 0.0)[0..],
                ).unwrap();
                self.state = GameState::Zoom(img);
                break;
            }
        }
    }

    fn update_zoom(&mut self, ctx: &mut Context) {
        for (i,img_button) in self.img_buttons.iter().enumerate() {
            if img_button.left_clicked(ctx,&self.mouse_state) {
                println!("{}",self.pics[i].to_lisp());    
                println!("button left clicked");
            }
            if img_button.right_clicked(ctx,&self.mouse_state) {
                println!("button right clicked");
                self.state = GameState::Select;
            }
        }
    }

    fn draw_select(&mut self, ctx:&mut Context) {
        for img_button in &self.img_buttons {
            img_button.draw(ctx);
        }
        // Render game ui
        {
            self.imgui_wrapper.render(ctx, self.hidpi_factor);
        }
    }

    fn draw_zoom(&self, ctx:&mut Context, img: &Image) {        
        let _ = graphics::draw(ctx,img,graphics::DrawParam::new());
    }
}



impl<S:Simd> EventHandler for MainState<S> {

    
    fn update(&mut self, ctx: &mut Context) -> GameResult<()> {        
        self.dt = timer::delta(ctx);
        match self.state {
            GameState::Select => self.update_select(ctx),
            GameState::Zoom(_) => self.update_zoom(ctx),
        }
        self.frame_elapsed = (self.frame_elapsed + self.dt.as_millis() as f32) % VIDEO_DURATION;        
        self.mouse_state = MouseState::Nothing;
        Ok(())
    }
    
    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        graphics::clear(ctx, graphics::BLACK);
        
        match &self.state { 
            GameState::Select => self.draw_select(ctx),
            GameState::Zoom(img) => self.draw_zoom(ctx,&img),
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
        x: f32,
        y: f32,
    ) {
        self.mouse_state = MouseState::Down (
                MouseButtonState {
                    which_button: button,            
                    x,y,
                });
                
        self.imgui_wrapper.update_mouse_down((
            button == MouseButton::Left,
            button == MouseButton::Right,
            button == MouseButton::Middle,
        ));
    }

    fn mouse_button_up_event(
        &mut self,
        _ctx: &mut Context,
        button: MouseButton,
        x: f32,
        y: f32,
    ) {
        self.mouse_state = MouseState::Up (
            MouseButtonState {
                which_button: button,            
                x,y,
            });
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

    let state = &mut MainState::<Avx2>::new(ctx, hidpi_factor)?;
    state.gen_population(ctx);
    event::run(ctx, event_loop, state)
}
