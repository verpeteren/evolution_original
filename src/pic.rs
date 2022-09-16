use std::collections::HashMap;
use std::sync::mpsc::{ channel, Receiver };
use std::sync::Arc;

use crate::actual_picture::ActualPicture;
use crate::apt::APTNode;
use crate::ggez_utility::{get_random_color, lerp_color};
use crate::parser::{Token, Lexer};
use crate::stack_machine::StackMachine;

use ggez::graphics::Color;
use rand::rngs::StdRng;
use simdeez::Simd;
use rand::prelude::*;
use rayon::prelude::*;

const GRADIENT_STOP_CHANCE: usize = 5; // 1 in 5
const MAX_GRADIENT_COUNT: usize = 10;
const MIN_GRADIENT_COUNT: usize = 2;
pub const GRADIENT_SIZE: usize = 512;

use CoordinateSystem::*;

#[derive(Clone, Debug, PartialEq)]
enum CoordinateSystem {
    Polar,
    Cartesian,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GradientData {
    colors: Vec<(Color, bool)>,
    index: APTNode,
    coord: CoordinateSystem,
}

#[derive(Clone)]
pub struct GrayscaleData {
    c: APTNode,
    coord: CoordinateSystem,
}

#[derive(Clone)]
pub struct MonoData {
    c: APTNode,
    coord: CoordinateSystem,
}

#[derive(Clone)]
pub struct RGBData {
    r: APTNode,
    g: APTNode,
    b: APTNode,
    coord: CoordinateSystem,
}

#[derive(Clone)]
pub struct HSVData {
    h: APTNode,
    s: APTNode,
    v: APTNode,
    coord: CoordinateSystem,
}

#[derive(Clone)]
pub enum Pic {
    Mono(MonoData),
    Grayscale(GrayscaleData),
    RGB(RGBData),
    HSV(HSVData),
    Gradient(GradientData),
}

impl Pic {
    pub fn new_mono(
        min: usize,
        max: usize,
        video: bool,
        rng: &mut StdRng,
        pic_names: &Vec<&String>,
    ) -> Pic {
        let tree = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        //let tree = APTNode::Cell2(vec![APTNode::X,APTNode::Y,APTNode::Constant(1.0)]);
        //let tree = APTNode::Picture("barn".to_string(),vec![APTNode::X,APTNode::Y]);
        Pic::Mono(MonoData {
            c: tree,
            coord: Polar,
        })
    }

    pub fn new_grayscale(
        min: usize,
        max: usize,
        video: bool,
        rng: &mut StdRng,
        pic_names: &Vec<&String>,
    ) -> Pic {
        let tree = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        //let tree = APTNode::Cell2(vec![APTNode::X,APTNode::Y,APTNode::Constant(1.0)]);
        Pic::Grayscale(GrayscaleData {
            c: tree,
            coord: Polar,
        })
    }

    pub fn new_gradient(
        min: usize,
        max: usize,
        video: bool,
        rng: &mut StdRng,
        pic_names: &Vec<&String>,
    ) -> Pic {
        //todo cleanup
        //color theory?
        let num_colors = rng.gen_range(MIN_GRADIENT_COUNT..MAX_GRADIENT_COUNT);
        let mut colors = Vec::with_capacity(num_colors);

        for _ in 0..num_colors {
            let stop = rng.gen_range(0..GRADIENT_STOP_CHANCE);
            if stop == 0 {
                colors.push((get_random_color(rng), true));
            } else {
                colors.push((get_random_color(rng), false));
            }
        }

        Pic::Gradient(GradientData {
            colors: colors,
            index: APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names),
            coord: Polar,
        })
    }

    pub fn new_rgb(
        min: usize,
        max: usize,
        video: bool,
        rng: &mut StdRng,
        pic_names: &Vec<&String>,
    ) -> Pic {
        let r = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        let g = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        let b = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        //let noise = APTNode::FBM::<S>(vec![APTNode::X,APTNode::Y]);
        Pic::RGB(RGBData {
            r,
            g,
            b,
            coord: Polar,
        })
    }

    pub fn new_hsv(
        min: usize,
        max: usize,
        video: bool,
        rng: &mut StdRng,
        pic_names: &Vec<&String>,
    ) -> Pic {
        let h = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        let s = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        let v = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        Pic::HSV(HSVData {
            h,
            s,
            v,
            coord: Polar,
        })
    }

    pub fn to_lisp(&self) -> String {
        match self {
            Pic::Mono(data) => format!("( Mono\n {} )", data.c.to_lisp()),
            Pic::Grayscale(data) => format!("( Grayscale\n {} )", data.c.to_lisp()),
            Pic::Gradient(data) => {
                let mut colors = "( Colors ".to_string();
                for (color, stop) in &data.colors {
                    if *stop {
                        colors += &format!(" ( StopColor {} {} {} )", color.r, color.g, color.b);
                    } else {
                        colors += &format!(" ( Color {} {} {} )", color.r, color.g, color.b);
                    }
                }
                format!("( Gradient\n {} {} )", colors, data.index.to_lisp())
            }
            Pic::RGB(data) => format!(
                "( RGB\n {}\n {}\n {} )",
                data.r.to_lisp(),
                data.g.to_lisp(),
                data.b.to_lisp()
            ),
            Pic::HSV(data) => format!(
                "( HSV\n {}\n {}\n {} )",
                data.h.to_lisp(),
                data.s.to_lisp(),
                data.v.to_lisp()
            ),
        }
    }

    pub fn get_video<S: Simd>(
        &self,
        pics: Arc<HashMap<String, ActualPicture>>,
        w: usize,
        h: usize,
        fps: u16,
        d: f32,
    ) -> Vec<Vec<u8>> {
        let frames = (fps as f32 * (d / 1000.0)) as i32;
        let frame_dt = 2.0 / frames as f32;

        let mut t = -1.0;
        let mut result = Vec::new();
        for _ in 0..frames {
            let frame_buffer = self.get_rgba8::<S>(true, pics.clone(), w, h, t);
            result.push(frame_buffer);
            t += frame_dt;
        }
        result
    }

    pub fn get_rgba8<S: Simd>(
        &self,
        threaded: bool,
        pics: Arc<HashMap<String, ActualPicture>>,
        w: usize,
        h: usize,
        t: f32,
    ) -> Vec<u8> {
        match self {
            Pic::Mono(data) => Pic::get_rgba8_mono::<S>(data, threaded, pics, w, h, t),
            Pic::Grayscale(data) => Pic::get_rgba8_grayscale::<S>(data, threaded, pics, w, h, t),
            Pic::Gradient(data) => Pic::get_rgba8_gradient::<S>(data, threaded, pics, w, h, t),
            Pic::RGB(data) => Pic::get_rgba8_rgb::<S>(data, threaded, pics, w, h, t),
            Pic::HSV(data) => Pic::get_rgba8_hsv::<S>(data, threaded, pics, w, h, t),
        }
    }

    fn get_rgba8_gradient<S: Simd>(
        data: &GradientData,
        threaded: bool,
        pics: Arc<HashMap<String, ActualPicture>>,
        w: usize,
        h: usize,
        t: f32,
    ) -> Vec<u8> {
        unsafe {
            let ts = S::set1_ps(t);
            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);
            let sm = StackMachine::<S>::build(&data.index);
            /*
            let mut min = 999999.0;
            let mut max = -99999.0;
            */

            let color_count = data.colors.iter().filter(|(_, stop)| !stop).count();
            let mut gradient = Vec::<Color>::new(); //todo actually compute this
            let step = (GRADIENT_SIZE as f32 / color_count as f32) / GRADIENT_SIZE as f32;
            let mut positions = Vec::<f32>::new();
            positions.push(0.0);
            let mut pos = step;
            for i in 1..data.colors.len() - 1 {
                let (_, stop) = data.colors[i];
                if stop {
                    positions.push(*positions.last().unwrap());
                } else {
                    positions.push(pos);
                    pos += step;
                }
            }
            positions.push(1.0);

            for i in 0..GRADIENT_SIZE {
                let pct = i as f32 / GRADIENT_SIZE as f32;
                let color2pos = positions.iter().position(|n| *n >= pct).unwrap();
                if color2pos == 0 {
                    gradient.push(data.colors[0].0);
                } else {
                    let color1 = data.colors[color2pos - 1].0;
                    let color2 = data.colors[color2pos].0;
                    let pct2 = positions[color2pos];
                    let pct1 = positions[color2pos - 1];
                    let range = pct2 - pct1;
                    let pct = (pct - pct1) / range;
                    gradient.push(lerp_color(color1, color2, pct));
                }
            }

            let process = |(y_pixel, chunk): (usize, &mut [u8])| {
                let mut stack = Vec::with_capacity(sm.instructions.len());
                stack.set_len(sm.instructions.len());

                let y = S::set1_ps((y_pixel as f32 / h as f32) * 2.0 - 1.0);
                let x_step = 2.0 / (w - 1) as f32;
                let mut x = S::setzero_ps();
                for i in (0..S::VF32_WIDTH).rev() {
                    x[i] = -1.0 + (x_step * i as f32);
                }
                let x_step = S::set1_ps(x_step * S::VF32_WIDTH as f32);

                for i in (0..w * 4).step_by(S::VF32_WIDTH * 4) {
                    let v = if data.coord == Cartesian {
                        sm.execute(&mut stack, pics.clone(), x, y, ts)
                    } else {
                        let (r, theta) = cartesian_to_polar::<S>(x, y);
                        sm.execute(&mut stack, pics.clone(), r, theta, ts)
                    };
                    let scaled_v = (v + S::set1_ps(1.0)) * S::set1_ps(0.5);
                    let index = S::cvtps_epi32(scaled_v * S::set1_ps(GRADIENT_SIZE as f32));

                    for j in 0..S::VF32_WIDTH {
                        let c = gradient[index[j] as usize % GRADIENT_SIZE];
                        chunk[i + j * 4] = (c.r * 255.0) as u8;
                        chunk[i + 1 + j * 4] = (c.g * 255.0) as u8;
                        chunk[i + 2 + j * 4] = (c.b * 255.0) as u8;
                        chunk[i + 3 + j * 4] = 255 as u8;
                    }
                    x = x + x_step;
                }
            };

            if threaded {
                result.par_chunks_mut(4 * w).enumerate().for_each(process);
            } else {
                result.chunks_exact_mut(4 * w).enumerate().for_each(process);
            }

            // println!("min:{} max:{} range:{}",min,max,max-min);
            result
        }
    }

    fn get_rgba8_grayscale<S: Simd>(
        data: &GrayscaleData,
        threaded: bool,
        pics: Arc<HashMap<String, ActualPicture>>,
        w: usize,
        h: usize,
        t: f32,
    ) -> Vec<u8> {
        unsafe {
            let ts = S::set1_ps(t);
            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);
            let sm = StackMachine::<S>::build(&data.c);
            /*
            let mut min = 999999.0;
            let mut max = -99999.0;
            */

            let process = |(y_pixel, chunk): (usize, &mut [u8])| {
                let mut stack = Vec::with_capacity(sm.instructions.len());
                stack.set_len(sm.instructions.len());

                let y = S::set1_ps((y_pixel as f32 / h as f32) * 2.0 - 1.0);
                let x_step = 2.0 / (w - 1) as f32;
                let mut x = S::setzero_ps();
                for i in (0..S::VF32_WIDTH).rev() {
                    x[i] = -1.0 + (x_step * i as f32);
                }
                let x_step = S::set1_ps(x_step * S::VF32_WIDTH as f32);

                for i in (0..w * 4).step_by(S::VF32_WIDTH * 4) {
                    let v = if data.coord == Cartesian {
                        sm.execute(&mut stack, pics.clone(), x, y, ts)
                    } else {
                        let (r, theta) = cartesian_to_polar::<S>(x, y);
                        sm.execute(&mut stack, pics.clone(), r, theta, ts)
                    };

                    // if v[0] > max { max = v[0]; }
                    // if v[0] < min { min = v[0]; }

                    let cs = (v + S::set1_ps(1.0)) * S::set1_ps(127.5);

                    for j in 0..S::VF32_WIDTH {
                        let c = (cs[j] as i32 % 256) as u8;
                        chunk[i + j * 4] = c;
                        chunk[i + 1 + j * 4] = c;
                        chunk[i + 2 + j * 4] = c;
                        chunk[i + 3 + j * 4] = 255 as u8;
                    }
                    x = x + x_step;
                }
            };

            if threaded {
                result.par_chunks_mut(4 * w).enumerate().for_each(process);
            } else {
                result.chunks_exact_mut(4 * w).enumerate().for_each(process);
            }
            // println!("min:{} max:{} range:{}",min,max,max-min);
            result
        }
    }

    fn get_rgba8_mono<S: Simd>(
        data: &MonoData,
        threaded: bool,
        pics: Arc<HashMap<String, ActualPicture>>,
        w: usize,
        h: usize,
        t: f32,
    ) -> Vec<u8> {
        unsafe {
            let ts = S::set1_ps(t);
            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);
            let sm = StackMachine::<S>::build(&data.c);
            /*
            let mut min = 999999.0;
            let mut max = -99999.0;
            */

            let process = |(y_pixel, chunk): (usize, &mut [u8])| {
                let mut stack = Vec::with_capacity(sm.instructions.len());
                stack.set_len(sm.instructions.len());

                let y = S::set1_ps((y_pixel as f32 / h as f32) * 2.0 - 1.0);
                let x_step = 2.0 / (w - 1) as f32;
                let mut x = S::setzero_ps();
                for i in (0..S::VF32_WIDTH).rev() {
                    x[i] = -1.0 + (x_step * i as f32);
                }
                let x_step = S::set1_ps(x_step * S::VF32_WIDTH as f32);

                for i in (0..w * 4).step_by(S::VF32_WIDTH * 4) {
                    let v = if data.coord == Cartesian {
                        sm.execute(&mut stack, pics.clone(), x, y, ts)
                    } else {
                        let (r, theta) = cartesian_to_polar::<S>(x, y);
                        sm.execute(&mut stack, pics.clone(), r, theta, ts)
                    };

                    for j in 0..S::VF32_WIDTH {
                        let c = if v[j] >= 0.0 { 255 } else { 0 };
                        chunk[i + j * 4] = c;
                        chunk[i + 1 + j * 4] = c;
                        chunk[i + 2 + j * 4] = c;
                        chunk[i + 3 + j * 4] = 255 as u8;
                    }
                    x = x + x_step;
                }
            };

            if threaded {
                result.par_chunks_mut(4 * w).enumerate().for_each(process);
            } else {
                result.chunks_exact_mut(4 * w).enumerate().for_each(process);
            }
            // println!("min:{} max:{} range:{}",min,max,max-min);
            result
        }
    }

    fn get_rgba8_rgb<S: Simd>(
        data: &RGBData,
        threaded: bool,
        pics: Arc<HashMap<String, ActualPicture>>,
        w: usize,
        h: usize,
        t: f32,
    ) -> Vec<u8> {
        unsafe {
            let ts = S::set1_ps(t);

            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);

            let r_sm = StackMachine::<S>::build(&data.r);
            let g_sm = StackMachine::<S>::build(&data.g);
            let b_sm = StackMachine::<S>::build(&data.b);
            let max_len = *[
                r_sm.instructions.len(),
                g_sm.instructions.len(),
                b_sm.instructions.len(),
            ]
            .iter()
            .max()
            .unwrap();

            let process = |(y_pixel, chunk): (usize, &mut [u8])| {
                let mut stack = Vec::with_capacity(max_len);
                stack.set_len(max_len);
                let y = S::set1_ps((y_pixel as f32 / h as f32) * 2.0 - 1.0);
                let x_step = 2.0 / (w - 1) as f32;
                let mut x = S::setzero_ps();
                for i in (0..S::VF32_WIDTH).rev() {
                    x[i] = -1.0 + (x_step * i as f32);
                }
                let x_step = S::set1_ps(x_step * S::VF32_WIDTH as f32);

                for i in (0..w * 4).step_by(S::VF32_WIDTH * 4) {
                    let (rs, gs, bs) = if data.coord == Cartesian {
                        let rs = (r_sm.execute(&mut stack, pics.clone(), x, y, ts)
                            + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let gs = (g_sm.execute(&mut stack, pics.clone(), x, y, ts)
                            + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let bs = (b_sm.execute(&mut stack, pics.clone(), x, y, ts)
                            + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        (rs, gs, bs)
                    } else {
                        let (x, y) = cartesian_to_polar::<S>(x, y);
                        let rs = (r_sm.execute(&mut stack, pics.clone(), x, y, ts)
                            + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let gs = (g_sm.execute(&mut stack, pics.clone(), x, y, ts)
                            + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let bs = (b_sm.execute(&mut stack, pics.clone(), x, y, ts)
                            + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        (rs, gs, bs)
                    };

                    for j in 0..S::VF32_WIDTH {
                        let r = (rs[j] as i32 % 255) as u8;
                        let g = (gs[j] as i32 % 255) as u8;
                        let b = (bs[j] as i32 % 255) as u8;
                        chunk[i + j * 4] = r;
                        chunk[i + 1 + j * 4] = g;
                        chunk[i + 2 + j * 4] = b;
                        chunk[i + 3 + j * 4] = 255 as u8;
                    }
                    x = x + x_step;
                }
            };
            if threaded {
                result.par_chunks_mut(4 * w).enumerate().for_each(process);
            } else {
                result.chunks_exact_mut(4 * w).enumerate().for_each(process);
            }

            result
        }
    }

    fn get_rgba8_hsv<S: Simd>(
        data: &HSVData,
        threaded: bool,
        pics: Arc<HashMap<String, ActualPicture>>,
        w: usize,
        h: usize,
        t: f32,
    ) -> Vec<u8> {
        unsafe {            
            let ts = S::set1_ps(t);
            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);

            let h_sm = StackMachine::<S>::build(&data.h);
            let s_sm = StackMachine::<S>::build(&data.s);
            let v_sm = StackMachine::<S>::build(&data.v);
            let max_len = *[
                h_sm.instructions.len(),
                s_sm.instructions.len(),
                v_sm.instructions.len(),
            ]
            .iter()
            .max()
            .unwrap();

            let process = |(y_pixel, chunk): (usize, &mut [u8])| {
                let mut stack = Vec::with_capacity(max_len);
                stack.set_len(max_len);
                let y = S::set1_ps((y_pixel as f32 / h as f32) * 2.0 - 1.0);
                let x_step = 2.0 / (w - 1) as f32;
                let mut x = S::setzero_ps();
                for i in (0..S::VF32_WIDTH).rev() {
                    x[i] = -1.0 + (x_step * i as f32);
                }
                let x_step = S::set1_ps(x_step * S::VF32_WIDTH as f32);

                for i in (0..w * 4).step_by(S::VF32_WIDTH * 4) {
                    let (hs, ss, vs) = if data.coord == Cartesian {
                        let hs = (h_sm.execute(&mut stack, pics.clone(), x, y, ts)
                            + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        let ss = (s_sm.execute(&mut stack, pics.clone(), x, y, ts)
                            + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        let vs = (v_sm.execute(&mut stack, pics.clone(), x, y, ts)
                            + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        (hs, ss, vs)
                    } else {
                        let (x, y) = cartesian_to_polar::<S>(x, y);
                        let hs = (h_sm.execute(&mut stack, pics.clone(), x, y, ts)
                            + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        let ss = (s_sm.execute(&mut stack, pics.clone(), x, y, ts)
                            + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        let vs = (v_sm.execute(&mut stack, pics.clone(), x, y, ts)
                            + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        (hs, ss, vs)
                    };

                    let (mut rs, mut gs, mut bs) =
                        hsv_to_rgb::<S>(wrap_0_1::<S>(hs), wrap_0_1::<S>(ss), wrap_0_1::<S>(vs));
                    rs = rs * S::set1_ps(255.0);
                    gs = gs * S::set1_ps(255.0);
                    bs = bs * S::set1_ps(255.0);
                    for j in 0..S::VF32_WIDTH {
                        let r = (rs[j] as i32 % 255) as u8;
                        let g = (gs[j] as i32 % 255) as u8;
                        let b = (bs[j] as i32 % 255) as u8;
                        chunk[i + j * 4] = r;
                        chunk[i + 1 + j * 4] = g;
                        chunk[i + 2 + j * 4] = b;
                        chunk[i + 3 + j * 4] = 255 as u8;
                    }
                    x = x + x_step;
                }
            };
            if threaded {
                result.par_chunks_mut(4 * w).enumerate().for_each(process);
            } else {
                result.chunks_exact_mut(4 * w).enumerate().for_each(process);
            }

            //   println!("img elapsed:{}", now.elapsed().as_millis());
            result
        }
    }
}

pub fn lisp_to_pic(code: String) -> Result<Pic, String> {
    let mut pic_opt = None;
    rayon::scope(|s| {
        let (sender, receiver) = channel();
        s.spawn(|_| {
            Lexer::begin_lexing(&code, sender);
        });

        // TODO: fix race condition that crashes at parser.rs:68. Workaround:
        std::thread::sleep(std::time::Duration::from_millis(1));

        pic_opt = Some(parse_pic(&receiver))
    });
    pic_opt.unwrap()
}

pub fn extract_line_number(token: &Token) -> usize {
    match token {
        Token::OpenParen(ln) | Token::CloseParen(ln) => *ln,
        Token::Constant(_, ln) | Token::Operation(_, ln) => *ln,
    }
}

#[must_use]
pub fn expect_open_paren(receiver: &Receiver<Token>) -> Result<(), String> {
    let open_paren = receiver.recv().map_err(|_| "Unexpected end of file")?;
    match open_paren {
        Token::OpenParen(_) => Ok(()),
        Token::Operation(v, line)|Token::Constant(v, line) => {
            return Err(format!(
                "Expected '(' on line {}, got a '{}'", line, v
            ))
        },
        _ => {
            return Err(format!(
                "Expected '(' on line {}",
                extract_line_number(&open_paren)
            ))
        }
    }
}

#[must_use]
pub fn expect_close_paren(receiver: &Receiver<Token>) -> Result<(), String> {
    let close_paren = receiver.recv().map_err(|_| "Unexpected end of file")?;
    match close_paren {
        Token::CloseParen(_) => Ok(()),
        _ => {
            return Err(format!(
                "Expected '(' on line {}",
                extract_line_number(&close_paren)
            ))
        }
    }
}

#[must_use]
pub fn expect_operation(s: &str, receiver: &Receiver<Token>) -> Result<(), String> {
    let op = receiver.recv().map_err(|_| "Unexpected end of file")?;
    match op {
        Token::Operation(op_str, _) => {
            if op_str.to_lowercase() == s {
                Ok(())
            } else {
                Err(format!(
                    "Expected '{}' on line {}, found {}",
                    s,
                    extract_line_number(&op),
                    op_str
                ))
            }
        }
        _ => {
            return Err(format!(
                "Expected '{}' on line {}, found {:?}",
                s,
                extract_line_number(&op),
                op
            ))
        }
    }
}

#[must_use]
pub fn expect_operations(ops: Vec<&str>, receiver: &Receiver<Token>) -> Result<String, String> {
    let op = receiver.recv().map_err(|_| "Unexpected end of file")?;
    for s in ops {
        match op {
            Token::Operation(op_str, _) => {
                if op_str.to_lowercase() == s {
                    return Ok(op_str.to_string());
                }
            }
            _ => (),
        }
    }
    return Err(format!(
        "Unexpected token on line {}",
        extract_line_number(&op),
    ));
}

#[must_use]
pub fn expect_constant(receiver: &Receiver<Token>) -> Result<f32, String> {
    let op = receiver.recv().map_err(|_| "Unexpected end of file")?;
    match op {
        Token::Constant(vstr, line_number) => {
            let v = vstr
                .parse::<f32>()
                .map_err(|_| format!("Unable to parse number {} on line {}", vstr, line_number))?;
            Ok(v)
        }
        _ => {
            return Err(format!(
                "Expected constant on line {}, found {:?}",
                extract_line_number(&op),
                op
            ))
        }
    }
}

pub fn parse_pic(receiver: &Receiver<Token>) -> Result<Pic, String> {
    expect_open_paren(receiver)?;
    let pic_type = receiver.recv().map_err(|_| "Unexpected end of file")?;
    match pic_type {
        Token::Operation(s, line_number) => match &s.to_lowercase()[..] {
            "mono" => Ok(Pic::Mono(MonoData{
                c: APTNode::parse_apt_node(receiver)?,
                coord: Cartesian,
            })),
            "grayscale" => Ok(Pic::Grayscale(GrayscaleData {
                c: APTNode::parse_apt_node(receiver)?,
                coord: Cartesian,
            })),
            "rgb" => Ok(Pic::RGB(RGBData {
                r: APTNode::parse_apt_node(receiver)?,
                g: APTNode::parse_apt_node(receiver)?,
                b: APTNode::parse_apt_node(receiver)?,
                coord: Cartesian,
            })),
            "hsv" => Ok(Pic::HSV(HSVData {
                h: APTNode::parse_apt_node(receiver)?,
                s: APTNode::parse_apt_node(receiver)?,
                v: APTNode::parse_apt_node(receiver)?,
                coord: Cartesian,
            })),
            "gradient" => {
                let mut colors = Vec::new();
                expect_open_paren(receiver)?;
                expect_operation("colors", receiver)?;
                loop {
                    let _token = receiver.recv().map_err(|_| "Unexpected end of file")?;
                    match expect_operations(vec!["color", "stopcolor"], receiver){
                        Err(e) => {
                            if e.starts_with("Unexpected token on line ") {
                                break;
                            } else {
                                panic!("{:?}", e);
                            }
                        },
                        Ok(color_type) => {
                            let r = expect_constant(receiver)?;
                            let g = expect_constant(receiver)?;
                            let b = expect_constant(receiver)?;
                            if color_type == "color" {
                                colors.push((Color::new(r, g, b, 1.0), false));
                            } else {
                                colors.push((Color::new(r, g, b, 1.0), true));
                            }
                            expect_close_paren(receiver)?;
                        }
                    }
                }
                Ok(Pic::Gradient(GradientData {
                    colors: colors,
                    index: APTNode::parse_apt_node(receiver)?,
                    coord: Cartesian,
                }))
            }
            _ => Err(format!("Unknown pic type {} at line {}", s, line_number)),
        },
        _ => Err(format!("Invalid picture type")), //todo line number etc
    }
}

#[inline(always)]
fn wrap_0_1<S: Simd>(v: S::Vf32) -> S::Vf32 {
    unsafe {
        let mut r = S::setzero_ps();
        for i in 0..S::VF32_WIDTH {
            r[i] = v[i] % 1.0001;
        }
        r
    }
}

#[inline(always)]
fn cartesian_to_polar<S: Simd>(x: S::Vf32, y: S::Vf32) -> (S::Vf32, S::Vf32) {
    unsafe {
        let zero = S::set1_ps(0.0);
        let pi = S::set1_ps(3.14159);
        let pix2 = S::set1_ps(3.14159 * 2.0);

        let mask = S::cmpge_ps(x, zero);
        let adjust = S::blendv_ps(pi, zero, mask);
        let mask = S::cmplt_ps(y, zero) & mask;
        let adjust = S::blendv_ps(adjust, pix2, mask);

        let r = S::sqrt_ps(x * x + y * y);
        let theta = S::fast_atan_ps(y / x) + adjust;
        (r, theta)
    }
}

fn hsv_to_rgb<S: Simd>(h: S::Vf32, s: S::Vf32, v: S::Vf32) -> (S::Vf32, S::Vf32, S::Vf32) {
    unsafe {
        let six = S::set1_ps(6.0);
        let one = S::set1_ps(1.0);
        let hi = S::cvtps_epi32(S::fast_floor_ps(h * six));
        let f = h * six - S::cvtepi32_ps(hi);
        let p = v * (one - s);
        let q = v * (one - f * s);
        let t = v * (one - (one - f) * s);

        let mut r = S::setzero_ps();
        let mut g = S::setzero_ps();
        let mut b = S::setzero_ps();

        for i in 0..S::VF32_WIDTH {
            match hi[i] % 6 {
                0 => {
                    r[i] = v[i];
                    g[i] = t[i];
                    b[i] = p[i];
                }
                1 => {
                    r[i] = q[i];
                    g[i] = v[i];
                    b[i] = p[i];
                }
                2 => {
                    r[i] = p[i];
                    g[i] = v[i];
                    b[i] = t[i];
                }
                3 => {
                    r[i] = p[i];
                    g[i] = q[i];
                    b[i] = v[i];
                }
                4 => {
                    r[i] = t[i];
                    g[i] = p[i];
                    b[i] = v[i];
                }
                _ => {
                    r[i] = v[i];
                    g[i] = p[i];
                    b[i] = q[i];
                }
            }
        }

        (r, g, b)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::Token;

    #[test]
    fn test_pic_new_mono() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let pic = Pic::new_mono(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        match pic {
            Pic::Mono(MonoData{c, coord}) => {
                let len = c.get_children().unwrap().len();
                assert!( len > 0 && len < 60);
                assert_eq!(coord, CoordinateSystem::Polar);
            },
            _ => {panic!("wrong type");},
        };
    }

    #[test]
    fn test_pic_new_grayscale() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let pic = Pic::new_grayscale(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        match pic {
            Pic::Grayscale(GrayscaleData{c, coord}) => {
                let len = c.get_children().unwrap().len();
                assert!( len > 0 && len < 60);
                assert_eq!(coord, CoordinateSystem::Polar);
            },
            _ => {panic!("wrong type");},
        };
    }

    #[test]
    fn test_pic_new_gradient() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let pic = Pic::new_gradient(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        match pic {
            Pic::Gradient(GradientData{colors, index, coord}) => {
                let len = colors.len();
                assert!(len > 1 && len < 10);
                let len = index.get_children().unwrap().len();
                assert!( len > 0 && len < 60);
                assert_eq!(coord, CoordinateSystem::Polar);
            },
            _ => {panic!("wrong type");},
        };
    }

    #[test]
    fn test_pic_new_rgb() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let pic = Pic::new_rgb(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        match pic {
            Pic::RGB(RGBData{r, g, b, coord}) => {
                let len = r.get_children().unwrap().len();
                assert!( len > 0 && len < 60);

                let len = g.get_children().unwrap().len();
                assert!( len > 0 && len < 60);

                let len = b.get_children().unwrap().len();
                assert!( len > 0 && len < 60);

                assert_eq!(coord, CoordinateSystem::Polar);
            },
            _ => {panic!("wrong type");},
        };
    }

    #[test]
    fn test_pic_new_hsv() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let pic = Pic::new_hsv(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        match pic {
            Pic::HSV(HSVData{h, s, v, coord}) => {
                let len = h.get_children().unwrap().len();
                assert!( len > 0 && len < 60);

                let len = s.get_children().unwrap().len();
                assert!( len > 0 && len < 60);

                let len = v.get_children().unwrap().len();
                assert!( len > 0 && len < 60);

                assert_eq!(coord, CoordinateSystem::Polar);
            },
            _ => {panic!("wrong type");},
        };
    }

    #[test]
    fn test_pic_to_lisp() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        let pic = Pic::new_mono(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        let sexpr = pic.to_lisp();
        assert!(sexpr.starts_with("( Mono\n "));
        assert!(sexpr.ends_with(" )"));
        assert!(sexpr.lines().collect::<Vec<_>>().len() > 1);

        let pic = Pic::new_grayscale(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        let sexpr = pic.to_lisp();
        assert!(sexpr.starts_with("( Grayscale\n "));
        assert!(sexpr.ends_with(" )"));
        assert!(sexpr.lines().collect::<Vec<_>>().len() > 1);

        let pic = Pic::new_gradient(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        let sexpr = pic.to_lisp();
        assert!(sexpr.starts_with("( Gradient\n "));
        assert!(sexpr.ends_with(" )"));
        assert!(sexpr.contains("( Colors ") || sexpr.contains(" ( StopColor "));
        assert!(sexpr.contains(" ( Color "));
        assert!(sexpr.lines().collect::<Vec<_>>().len() > 0);

        let pic = Pic::new_rgb(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        let sexpr = pic.to_lisp();
        assert!(sexpr.starts_with("( RGB\n"));
        assert!(sexpr.ends_with(" )"));
        assert!(sexpr.lines().collect::<Vec<_>>().len() > 3);

        let pic = Pic::new_hsv(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        let sexpr = pic.to_lisp();
        assert!(sexpr.starts_with("( HSV\n"));
        assert!(sexpr.ends_with(" )"));
        assert!(sexpr.lines().collect::<Vec<_>>().len() > 1);
    }

    // todo: refactor into a separate module e.g. parser::token
    #[test]
    fn test_extract_line_number() {
        assert_eq!(extract_line_number(&Token::OpenParen(6)), 6);
        assert_eq!(extract_line_number(&Token::CloseParen(6)), 6);
        assert_eq!(extract_line_number(&Token::Operation("blablabla", 6)), 6);
        assert_eq!(extract_line_number(&Token::Constant("blablabla", 6)), 6);
    }

}
