// todo
// - fix up gradient to work properly when parsing
// - cross breeding of picture expressions
// - load up thumbnails in a background thread so ui isn't blocked

extern crate ggez;
extern crate image;

mod actual_picture;
mod apt;
mod ggez_utility;
mod imgui_wrapper;
mod parser;
mod pic;
mod stack_machine;
mod ui;

use std::collections::HashMap;
use std::env::var;
use std::fs::{read_dir, File};
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::thread::spawn;

use crate::actual_picture::ActualPicture;
use crate::imgui_wrapper::{ImGuiWrapper, EXEC_NAME};
use crate::pic::{lisp_to_pic, Pic};
use crate::ui::{Button, MouseButtonState, MouseState};

use clap::Parser;
use ggez::conf::{WindowMode, WindowSetup};
use ggez::event::{run, EventHandler, KeyCode, KeyMods, MouseButton};
use ggez::graphics::{clear, draw, present, window, Color, DrawParam, Image};
use ggez::timer::delta;
use ggez::{Context, ContextBuilder, GameError, GameResult};
use image::{save_buffer_with_format, ColorType, ImageFormat};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use simdeez::avx2::*;
use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;

const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;
const VIDEO_DURATION: f32 = 5000.0; //milliseconds

const THUMB_ROWS: u16 = 6;
const THUMB_COLS: u16 = 7;
const THUMB_WIDTH: u16 = 128;
const THUMB_HEIGHT: u16 = 128;

const TREE_MIN: usize = 1;
const TREE_MAX: usize = 40;

const STD_PATH: &'static str = "pictures";
const STD_FILE_OUT: &'static str = "out.png";

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Args {
    #[clap(short, long, value_parser, default_value = STD_PATH, help="The path to images that can be loaded via the Pic- operation")]
    pictures_path: String,

    #[clap(short, long, value_parser, default_value_t = WIDTH, help="The width of the generated image")]
    width: usize,

    #[clap(short, long, value_parser, default_value_t = HEIGHT, help="The height of the generated image")]
    height: usize,

    #[clap(
        short,
        long,
        value_parser,
        default_value_t = 0.0,
        help = "set the T variable"
    )]
    time: f32,

    #[clap(
        short,
        long,
        value_parser,
        help = "filename to read sexpr from and disabling the UI; Use '-' to read from stdin."
    )]
    input: Option<String>,

    #[clap(
        short,
        long,
        value_parser,
        requires("input"),
        help = "image file to write to"
    )]
    output: Option<String>,

    #[clap(
        short,
        long,
        value_parser,
        requires("input"),
        help = "The path where to store a copy of the input and output files as part of the creative workflow"
    )]
    copy_path: Option<String>,
}

struct RwArc<T>(Arc<RwLock<T>>);
impl<T> RwArc<T> {
    pub fn new(t: T) -> RwArc<T> {
        RwArc(Arc::new(RwLock::new(t)))
    }

    pub fn read(&self) -> std::sync::RwLockReadGuard<T> {
        self.0.read().unwrap()
    }

    pub fn write(&self, t: T) {
        *self.0.write().unwrap() = t;
    }

    pub fn clone(&self) -> RwArc<T> {
        RwArc(self.0.clone())
    }
}

simd_runtime_generate!(
    fn pic_get_rgba8(
        pic: &Pic,
        threaded: bool,
        pictures: Arc<HashMap<String, ActualPicture>>,
        width: usize,
        height: usize,
        t: f32,
    ) -> Vec<u8> {
        pic.get_rgba8::<S>(threaded, pictures, width, height, t)
    }
);

enum GameState {
    Select,
    Zoom,
}

enum BackgroundImage {
    NotYet,
    Almost(Vec<u8>),
    Complete(Image),
}

struct MainState {
    state: GameState,
    mouse_state: MouseState,
    imgui_wrapper: ImGuiWrapper,
    img_buttons: Vec<Button>,
    pics: Vec<Pic>,
    dt: std::time::Duration,
    frame_elapsed: f32,
    rng: StdRng,
    zoom_image: RwArc<BackgroundImage>,
    pictures: Arc<HashMap<String, ActualPicture>>,
    dimensions: (usize, usize),
    running: bool,
}

impl MainState {
    fn new(mut ctx: &mut Context, pic_path: &Path, args: &Args) -> GameResult<MainState> {
        let imgui_wrapper = ImGuiWrapper::new(&mut ctx);
        let pics =
            load_pictures(Some(&mut ctx), pic_path).map_err(|x| GameError::FilesystemError(x))?;

        let s = MainState {
            state: GameState::Select,
            imgui_wrapper,
            pics: Vec::new(),
            img_buttons: Vec::new(),
            dt: std::time::Duration::new(0, 0),
            frame_elapsed: args.time,
            rng: StdRng::from_rng(rand::thread_rng()).unwrap(),
            mouse_state: MouseState::Nothing,
            zoom_image: RwArc::new(BackgroundImage::NotYet),
            pictures: Arc::new(pics),
            dimensions: (args.width, args.height),
            running: false,
        };
        Ok(s)
    }

    fn gen_population(&mut self, ctx: &mut Context) {
        if !self.running {
            self.running = true;
            let len = (THUMB_COLS * THUMB_ROWS) as usize;
            println!(
                "Generating a new population of {} thumbnails. Please be patient",
                len
            );
            // todo make this layout code less dumb
            let mut buttons = Vec::with_capacity(len);
            let mut pics = Vec::with_capacity(len);
            let width = 1.0 / (THUMB_COLS as f32 * 1.01);
            let height = 1.0 / (THUMB_ROWS as f32 * 1.01);
            let mut y_pct = 0.01;
            let pic_names = &self.pictures.keys().collect();
            for _ in 0..THUMB_ROWS {
                let mut x_pct = 0.01;
                for _ in 0..THUMB_COLS {
                    let pic_type = self.rng.gen_range(0..5);

                    let pic = match pic_type {
                        0 => Pic::new_mono(TREE_MIN, TREE_MAX, false, &mut self.rng, pic_names),
                        1 => Pic::new_gradient(TREE_MIN, TREE_MAX, false, &mut self.rng, pic_names),
                        2 => Pic::new_rgb(TREE_MIN, TREE_MAX, false, &mut self.rng, pic_names),
                        3 => Pic::new_hsv(TREE_MIN, TREE_MAX, false, &mut self.rng, pic_names),
                        4 => {
                            Pic::new_grayscale(TREE_MIN, TREE_MAX, false, &mut self.rng, pic_names)
                        }
                        _ => panic!("invalid"),
                    };

                    let img = Image::from_rgba8(
                        ctx,
                        THUMB_WIDTH,
                        THUMB_HEIGHT,
                        &pic_get_rgba8_runtime_select(
                            &pic,
                            false,
                            self.pictures.clone(),
                            THUMB_WIDTH as usize,
                            THUMB_HEIGHT as usize,
                            self.frame_elapsed,
                        )[0..],
                    )
                    .unwrap();

                    if true {
                        //Debug stress test::see if we can parse it back
                        let sexpr = pic.to_lisp();
                        match lisp_to_pic(sexpr.clone()) {
                            Ok(_) => {}
                            Err(err) => {
                                eprintln!("-----\n{:?}\n{:?}\n{:?}", err, pic.to_tree(), &sexpr);
                            }
                        }
                    }

                    pics.push(pic);
                    buttons.push(Button::new(img, x_pct, y_pct, width - 0.01, height - 0.01));
                    x_pct += width;
                }
                y_pct += height;
            }
            self.pics = pics;
            self.img_buttons = buttons;
            self.running = false;
            println!("...done");
        }
    }

    fn update_select(&mut self, ctx: &mut Context) {
        let (width, height) = self.dimensions;
        let t = self.frame_elapsed;
        for (i, img_button) in self.img_buttons.iter().enumerate() {
            if img_button.left_clicked(ctx, &self.mouse_state) {
                println!("{}", self.pics[i].to_lisp());
                break;
            }
            if img_button.right_clicked(ctx, &self.mouse_state) {
                let pic = self.pics[i].clone();
                let arc = self.zoom_image.clone();
                let pics = self.pictures.clone();
                spawn(move || {
                    let img_data = pic_get_rgba8_runtime_select(&pic, true, pics, width, height, t);
                    arc.write(BackgroundImage::Almost(img_data));
                });
                self.state = GameState::Zoom;
                break;
            }
        }
    }

    fn update_zoom(&mut self, ctx: &mut Context) {
        let (width, height) = self.dimensions;
        let maybe_img = match &*self.zoom_image.read() {
            BackgroundImage::NotYet => None,
            BackgroundImage::Almost(data) => {
                let img = Image::from_rgba8(ctx, width as u16, height as u16, &data[0..]).unwrap();
                Some(img)
            }
            BackgroundImage::Complete(_) => None,
        };
        match maybe_img {
            None => (),
            Some(img) => self.zoom_image.write(BackgroundImage::Complete(img)),
        }
        //todo just check for clicks on the zoom image
        for (_i, img_button) in self.img_buttons.iter().enumerate() {
            if img_button.left_clicked(ctx, &self.mouse_state) {
                // println!("{}", self.pics[i].to_lisp());
            }
            if img_button.right_clicked(ctx, &self.mouse_state) {
                self.zoom_image.write(BackgroundImage::NotYet);
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
            let window = window(ctx);

            self.imgui_wrapper.render(ctx, window.scale_factor() as f32);
        }
    }

    fn draw_zoom(&self, ctx: &mut Context) {
        match &*self.zoom_image.read() {
            BackgroundImage::NotYet => (),
            BackgroundImage::Almost(_) => (),
            BackgroundImage::Complete(img) => {
                let _ = draw(ctx, img, DrawParam::new());
            }
        }
    }
}

impl EventHandler<GameError> for MainState {
    fn update(&mut self, ctx: &mut Context) -> GameResult<()> {
        self.dt = delta(ctx);
        match self.state {
            GameState::Select => self.update_select(ctx),
            GameState::Zoom => self.update_zoom(ctx),
        }
        self.frame_elapsed = (self.frame_elapsed + self.dt.as_millis() as f32) % VIDEO_DURATION;
        self.mouse_state = MouseState::Nothing;
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult<()> {
        clear(ctx, Color::BLACK);

        match &self.state {
            GameState::Select => self.draw_select(ctx),
            GameState::Zoom => self.draw_zoom(ctx),
        }

        present(ctx)?;
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

    fn text_input_event(&mut self, ctx: &mut Context, ch: char) {
        self.imgui_wrapper.update_keyboard(ch);
        if ch == ' ' {
            self.gen_population(ctx);
        }
    }
}

pub fn load_pictures(
    mut o_ctx: Option<&mut Context>,
    pic_path: &Path,
) -> Result<HashMap<String, ActualPicture>, String> {
    let mut pictures = HashMap::new();
    for file in read_dir(pic_path).expect(&format!("Cannot read path {:?}", pic_path)) {
        let short_file_name = file
            .as_ref()
            .unwrap()
            .file_name()
            .into_string()
            .expect("Cannot convert file's name ");
        let pic = match o_ctx.as_mut() {
            Some(ctx) => {
                ActualPicture::new_via_ctx(ctx, &short_file_name).expect("Cannot open file")
            }
            None => {
                let path = file.as_ref().unwrap().path();
                let full_file_name = path.to_string_lossy();
                ActualPicture::new_via_file(&full_file_name.to_owned())?
            }
        };
        pictures.insert(short_file_name, pic);
    }
    Ok(pictures)
}

fn get_picture_path(args: &Args) -> PathBuf {
    let mut path_buf = if let Ok(manifest_dir) = var("CARGO_MANIFEST_DIR") {
        PathBuf::from(manifest_dir)
    } else {
        PathBuf::from("./")
    };
    path_buf.push(args.pictures_path.clone());
    path_buf
}

fn main_gui(args: &Args) -> GameResult {
    match rayon::ThreadPoolBuilder::new()
        .num_threads(0)
        .build_global()
    {
        Ok(_) => (),
        Err(x) => panic!("{}", x),
    }

    let pic_path = get_picture_path(&args);
    let scale = 1.0;

    let cb = ContextBuilder::new(EXEC_NAME, "ggez")
        .add_resource_path(pic_path.as_path())
        .window_setup(WindowSetup::default().title(EXEC_NAME))
        .window_mode(
            WindowMode::default().dimensions(args.width as f32 * scale, args.height as f32 * scale),
        );
    let (mut ctx, event_loop) = cb.build()?;

    let mut state = MainState::new(&mut ctx, pic_path.as_path(), args).unwrap();
    state.gen_population(&mut ctx);
    run(ctx, event_loop, state)
}

fn main_cli(args: &Args) {
    let (width, height, t) = (args.width, args.height, args.time);
    let out_filename = args.output.as_ref().unwrap().to_string();
    let pic_path = get_picture_path(&args);

    let pictures = Arc::new(load_pictures(None, pic_path.as_path()).unwrap());

    let input_file_name = args.input.as_ref().unwrap();
    let mut contents = String::new();
    if input_file_name == "-" {
        let _bytes = std::io::stdin().read_to_string(&mut contents).unwrap();
    } else {
        let mut file = File::open(input_file_name).unwrap();
        file.read_to_string(&mut contents).unwrap();
    }
    let pic = lisp_to_pic(contents).unwrap();
    let rgba8 = pic_get_rgba8_runtime_select(&pic, false, pictures, width, height, t);
    let out_file = Path::new(&out_filename);
    let format = match out_file.extension() {
        Some(ext) => {
            match ext.to_str().unwrap().to_lowercase().as_str() {
                // support these?
                "tga" => ImageFormat::Tga,
                "dds" => ImageFormat::Dds,
                "hdr" => ImageFormat::Hdr,
                "farb" => ImageFormat::Farbfeld,
                // do these imply video?
                "gif" => ImageFormat::Gif,
                "avi" => ImageFormat::Avif,
                // commodity
                "bmp" => ImageFormat::Bmp,
                "ico" => ImageFormat::Ico,
                "webp" => ImageFormat::WebP,
                "pnm" => ImageFormat::Pnm,
                "tif" | "tiff" => ImageFormat::Tiff,
                "jpg" | "jpeg" => ImageFormat::Jpeg,
                "png" => ImageFormat::Png,
                _ => ImageFormat::Png,
            }
        }
        None => ImageFormat::Png,
    };

    save_buffer_with_format(
        out_file,
        &rgba8[0..],
        width as u32,
        height as u32,
        ColorType::Rgba8,
        format,
    )
    .unwrap();
}

pub fn main() {
    let mut args = Args::parse();
    let run_gui = match &args.input {
        None => true,
        Some(_x) => {
            if args.output.is_none() {
                args.output = Some(STD_FILE_OUT.to_string());
            }
            false
        }
    };
    if run_gui {
        main_gui(&args).unwrap();
    } else {
        main_cli(&args);
    }
}
