// todo
// - fix up gradient to work properly when parsing
// - cross breeding of picture expressions
// - load up thumbnails in a background thread so ui isn't blocked

extern crate ggez;

mod actual_picture;
mod apt;
mod ggez_utility;
mod imgui_wrapper;
mod parser;
mod pic;
mod stack_machine;
mod ui;

use crate::actual_picture::*;
use crate::imgui_wrapper::ImGuiWrapper;
use crate::parser::*;
use crate::pic::*;
use crate::ui::*;
use ggez::conf;
use ggez::event::{self, EventHandler, KeyCode, KeyMods, MouseButton};
use ggez::graphics::{self, Image};
use ggez::timer;
use ggez::{Context, GameResult};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::*;
use rayon::*;
use simdeez::avx2::*;
use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;
use std::collections::HashMap;
use std::env;
use std::fs::{self};
use std::io::*;
use std::path::{self, Path};
use std::sync::mpsc::*;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread;
use std::time::Instant;

const WIDTH: usize = 1024;
const HEIGHT: usize = 768;
const VIDEO_DURATION: f32 = 5000.0; //milliseconds

const THUMB_ROWS: u16 = 6;
const THUMB_COLS: u16 = 7;

const TREE_MIN: usize = 1;
const TREE_MAX: usize = 5;

enum GameState {
    Select,
    Zoom(usize),
}

enum BackgroundImage {
    NotYet,
    Complete(graphics::Image),
}

struct MainState {
    state: GameState,
    mouse_state: MouseState,
    imgui_wrapper: ImGuiWrapper,
    hidpi_factor: f32,
    img_buttons: Vec<Button>,
    pics: Vec<Pic>,
    dt: std::time::Duration,
    frame_elapsed: f32,
    rng: StdRng,
    zoom_image: BackgroundImage,
    zoom_image_data: Arc<RwLock<Option<Vec<u8>>>>,
    pictures: Arc<HashMap<String, ActualPicture>>,
}

impl MainState {
    fn gen_population(&mut self, ctx: &mut Context) {
        // todo make this layout code less dumb
        let now = Instant::now();
        self.img_buttons.clear();
        let width = 1.0 / (THUMB_COLS as f32 * 1.01);
        let height = 1.0 / (THUMB_ROWS as f32 * 1.01);
        let mut y_pct = 0.01;
        for _ in 0..THUMB_ROWS {
            let mut x_pct = 0.01;
            for _ in 0..THUMB_COLS {
                let pic_type = self.rng.gen_range(0, 4);
               // let pic_type = 0;
                let pic = match pic_type {
                    0 => Pic::new_mono(TREE_MIN, TREE_MAX, false, &mut self.rng),
                    1 => Pic::new_gradient(TREE_MIN, TREE_MAX, false, &mut self.rng),
                    2 => Pic::new_rgb(TREE_MIN, TREE_MAX, false, &mut self.rng),
                    3 => Pic::new_hsv(TREE_MIN, TREE_MAX, false, &mut self.rng),
                    _ => panic!("invalid"),
                };

                let img = graphics::Image::from_rgba8(
                    ctx,
                    256 as u16,
                    256 as u16,
                    &pic.get_rgba8::<Avx2>(self.pictures.clone(),256, 256, 0.0)[0..],
                )
                .unwrap();
                self.pics.push(pic);
                self.img_buttons
                    .push(Button::new(img, x_pct, y_pct, width - 0.01, height - 0.01));
                x_pct += width;
            }
            println!("--------------------");
            y_pct += height;
        }
        println!("genpop elapsed:{}", now.elapsed().as_millis());
    }

    fn new(mut ctx: &mut Context, hidpi_factor: f32) -> GameResult<MainState> {
        let imgui_wrapper = ImGuiWrapper::new(&mut ctx);

        let s = MainState {
            state: GameState::Select,
            imgui_wrapper,
            hidpi_factor,
            pics: Vec::new(),
            img_buttons: Vec::new(),
            dt: std::time::Duration::new(0, 0),
            frame_elapsed: 0.0,
            rng: StdRng::from_rng(rand::thread_rng()).unwrap(),
            mouse_state: MouseState::Nothing,
            zoom_image: BackgroundImage::NotYet,
            zoom_image_data: Arc::new(RwLock::new(None)),
            pictures: Arc::new(load_pictures(ctx)),
        };
        Ok(s)
    }

    fn update_select(&mut self, ctx: &mut Context) {
        for (i, img_button) in self.img_buttons.iter().enumerate() {
            if img_button.left_clicked(ctx, &self.mouse_state) {
                println!("{}", self.pics[i].to_lisp());
                println!("button left clicked");
                break;
            }
            if img_button.right_clicked(ctx, &self.mouse_state) {
                println!("button right clicked");
                let pic = self.pics[i].clone();
                let arc = self.zoom_image_data.clone();
                let pics = self.pictures.clone();
                thread::spawn(move || {
                    println!("create image");
                    let img_data = pic.get_rgba8::<Avx2>(pics,1024 as usize, 768 as usize, 0.0);
                    *arc.write().unwrap() = Some(img_data)
                });
                self.state = GameState::Zoom(i);
                break;
            }
        }
    }

    fn update_zoom(&mut self, ctx: &mut Context) {
        match &self.zoom_image {
            BackgroundImage::NotYet => match &*self.zoom_image_data.read().unwrap() {
                Some(data) => {
                    println!("setting zoom image");
                    let img = graphics::Image::from_rgba8(ctx, 1024 as u16, 768 as u16, &data[0..])
                        .unwrap();
                    self.zoom_image = BackgroundImage::Complete(img)
                }
                None => (),
            },
            BackgroundImage::Complete(img) => (),
        }
        for (i, img_button) in self.img_buttons.iter().enumerate() {
            if img_button.left_clicked(ctx, &self.mouse_state) {
                println!("{}", self.pics[i].to_lisp());
                println!("button left clicked");
            }
            if img_button.right_clicked(ctx, &self.mouse_state) {
                println!("button right clicked");
                self.zoom_image = BackgroundImage::NotYet;
                *self.zoom_image_data.write().unwrap() = None;
                self.state = GameState::Select;
            }
        }
    }

    fn draw_select(&mut self, ctx: &mut Context) {
        for img_button in &self.img_buttons {
            img_button.draw(ctx);
        }
        // Render game ui
        {
            self.imgui_wrapper.render(ctx, self.hidpi_factor);
        }
    }

    fn draw_zoom(&self, ctx: &mut Context, index: usize) {
        match &self.zoom_image {
            BackgroundImage::NotYet => (),
            BackgroundImage::Complete(img) => {
                let _ = graphics::draw(ctx, img, graphics::DrawParam::new());
            }
        }
    }
}

impl EventHandler for MainState {
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
            GameState::Zoom(index) => self.draw_zoom(ctx, *index),
        }

        graphics::present(ctx)?;
        Ok(())
    }

    fn mouse_motion_event(&mut self, _ctx: &mut Context, x: f32, y: f32, _dx: f32, _dy: f32) {
        self.imgui_wrapper.update_mouse_pos(x, y);
    }

    fn mouse_button_down_event(&mut self, _ctx: &mut Context, button: MouseButton, x: f32, y: f32) {
        self.mouse_state = MouseState::Down(MouseButtonState {
            which_button: button,
            x,
            y,
        });

        self.imgui_wrapper.update_mouse_down((
            button == MouseButton::Left,
            button == MouseButton::Right,
            button == MouseButton::Middle,
        ));
    }

    fn mouse_button_up_event(&mut self, _ctx: &mut Context, button: MouseButton, x: f32, y: f32) {
        self.mouse_state = MouseState::Up(MouseButtonState {
            which_button: button,
            x,
            y,
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
pub fn load_pictures(ctx: &mut Context) -> HashMap<String, ActualPicture> {
    let pic_path = Path::new("pictures");
    let mut pictures = HashMap::new();
    match fs::read_dir(pic_path) {
        Ok(files) => {
            for file in files {
                let file_name = file.unwrap().file_name().into_string().unwrap();
                let img = graphics::Image::new(ctx, "/".to_string() + &file_name).unwrap();
                let name = file_name.split(".").nth(0).unwrap().to_string();
                pictures.insert(name.clone(), ActualPicture::new(ctx, img, name));
            }
        }
        Err(_) => (),
    }
    pictures
}

pub fn main() -> ggez::GameResult {
    match rayon::ThreadPoolBuilder::new()
        .num_threads(0)
        .build_global()
    {
        Ok(_) => (),
        Err(x) => panic!("{}", x),
    }

    let pictures_dir = if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        let mut path = path::PathBuf::from(manifest_dir);
        path.push("pictures");
        path
    } else {
        path::PathBuf::from("./pictures")
    };

    let hidpi_factor: f32;
    {
        // Create a dummy window so we can get monitor scaling information
        let cb = ggez::ContextBuilder::new("", "");
        let (_ctx, events_loop) = &mut cb.build()?;
        hidpi_factor = events_loop.get_primary_monitor().get_hidpi_factor() as f32;
        println!("main hidpi_factor = {}", hidpi_factor);
    }

    let cb = ggez::ContextBuilder::new("super_simple with imgui", "ggez")
        .add_resource_path(pictures_dir)
        .window_setup(conf::WindowSetup::default().title("super_simple with imgui"))
        .window_mode(
            conf::WindowMode::default().dimensions(WIDTH as f32 * 1.0, HEIGHT as f32 * 1.0),
        );
    let (ref mut ctx, event_loop) = &mut cb.build()?;

    let state = &mut MainState::new(ctx, hidpi_factor)?;
    state.gen_population(ctx);
    event::run(ctx, event_loop, state)
}
