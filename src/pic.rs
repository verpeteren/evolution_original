use crate::apt::*;
use crate::stack_machine::*;
use crate::ggez_utility::*;
use rand::rngs::StdRng;
use rand::*;
use rayon::prelude::*;
use ggez::graphics::Color;
use simdeez::*;
use std::time::Instant;


const MAX_GRADIENT_COUNT : usize = 10;
const MIN_GRADIENT_COUNT : usize = 2;
const GRADIENT_SIZE : usize = 512;

#[derive(Clone)]
pub struct GradientData {
    gradient: Vec<Color>,
    index: APTNode,
}

#[derive(Clone)]
pub struct MonoData {    
    c: APTNode,
}

#[derive(Clone)]
pub struct RGBData {
    r: APTNode,
    g: APTNode,
    b: APTNode,
}

#[derive(Clone)]
pub struct HSVData {
    h: APTNode,
    s: APTNode,
    v: APTNode,
}

#[derive(Clone)]
pub enum Pic {
    Mono(MonoData),
    RGB(RGBData),
    HSV(HSVData),
    Gradient(GradientData),    
}


impl Pic {
    pub fn new_mono(min: usize, max: usize, video: bool, rng: &mut StdRng) -> Pic
    {
        let tree = APTNode::generate_tree(rng.gen_range(min, max), video, rng);
        //let tree = APTNode::Cell2(vec![APTNode::X,APTNode::Y,APTNode::Constant(1.0)]);
        Pic::Mono(MonoData { c: tree })
    }

    pub fn new_gradient(min: usize, max: usize, video: bool, rng: &mut StdRng) -> Pic
    {
         //todo cleanup 
        //color theory? 
        let num_colors = rng.gen_range(MIN_GRADIENT_COUNT,MAX_GRADIENT_COUNT);        
        let mut gradient = Vec::with_capacity(GRADIENT_SIZE);
        let mut pos = Vec::with_capacity(num_colors);
        let mut colors = Vec::with_capacity(num_colors);
        pos.push(0.0);        
        for _ in 1..num_colors-1 {
            pos.push(rng.gen_range(0.0,1.0))
        }
        pos.push(1.0);
        pos.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for _ in 0 .. num_colors {
            colors.push(get_random_color(rng));
        }
        println!("colorlen:{} poslen:{}",colors.len(),pos.len());
        
        for i in 0 .. GRADIENT_SIZE {
            let pct = i as f32 / GRADIENT_SIZE as f32;
            let color2pos = pos.iter().position(|n| *n >= pct).unwrap();      
            if color2pos == 0 {
                gradient.push(colors[0]);
            } else {
                let color1 = colors[color2pos-1];
                let color2 = colors[color2pos];
                let pct2 = pos[color2pos];
                let pct1 = pos[color2pos-1];
                let range = pct2-pct1;
                let pct = (pct - pct1)/range;
                gradient.push(lerp_color(color1,color2,pct));
            }
        }                        
        Pic::Gradient(
            GradientData {
            gradient: gradient,
            index: APTNode::generate_tree(rng.gen_range(min, max), video, rng)
        })
    }

    pub fn new_rgb(min: usize, max: usize, video: bool, rng: &mut StdRng) -> Pic {
        let r = APTNode::generate_tree(rng.gen_range(min, max), video, rng);
        let g = APTNode::generate_tree(rng.gen_range(min, max), video, rng);
        let b = APTNode::generate_tree(rng.gen_range(min, max), video, rng);
        //let noise = APTNode::FBM::<S>(vec![APTNode::X,APTNode::Y]);
        Pic::RGB(RGBData { r, g, b })
    }

    pub fn new_hsv(min: usize, max: usize, video: bool, rng: &mut StdRng) -> Pic {
        let h = APTNode::generate_tree(rng.gen_range(min, max), video, rng);
        let s = APTNode::generate_tree(rng.gen_range(min, max), video, rng);
        let v = APTNode::generate_tree(rng.gen_range(min, max), video, rng);
        Pic::HSV(HSVData { h, s, v })
    }

   

    pub fn to_lisp(&self) -> String {
        match self {
            Pic::Mono(data) => format!("Mono\n {}", data.c.to_lisp()),
            Pic::Gradient(data) => format!("Gradient\n {}", data.index.to_lisp()),
            Pic::RGB(data) =>  format!(
                "RGB\n{} \n{}\n{}",
                data.r.to_lisp(),
                data.g.to_lisp(),
                data.b.to_lisp()
                ),
            Pic::HSV(data) =>  format!(
                "HSV\n{} \n{}\n{}",
                data.h.to_lisp(),
                data.s.to_lisp(),
                data.v.to_lisp()
            )                                            
        }
    }

    pub fn get_video<S:Simd>(&self, w: usize, h: usize, fps: u16, d: f32) -> Vec<Vec<u8>> {
        let now = Instant::now();
        let frames = (fps as f32 * (d / 1000.0)) as i32;
        let frame_dt = 2.0 / frames as f32;

        let mut t = -1.0;
        let mut result = Vec::new();
        for _ in 0..frames {
            let frame_buffer = self.get_rgba8::<S>(w, h, t);
            result.push(frame_buffer);
            t += frame_dt;
        }
        println!("img elapsed:{}", now.elapsed().as_millis());
        result
    }

    pub fn get_rgba8<S:Simd>(&self,w: usize, h:usize, t:f32) -> Vec<u8> {
        match self {
            Pic::Mono(data) => Pic::get_rgba8_mono::<S>(data,w,h,t),
            Pic::Gradient(data) => Pic::get_rgba8_gradient::<S>(data,w,h,t),
            Pic::RGB(data) => Pic::get_rgba8_rgb::<S>(data,w,h,t),
            Pic::HSV(data) =>Pic::get_rgba8_hsv::<S>(data,w,h,t),
        }
    }

    fn get_rgba8_gradient<S:Simd>(data:&GradientData, w: usize, h: usize, t: f32) -> Vec<u8> {
        unsafe {
            let now = Instant::now();
            let ts = S::set1_ps(t);
            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);
            let sm = StackMachine::<S>::build(&data.index);
            let mut min = 999999.0;
            let mut max = -99999.0;
            result
                .par_chunks_mut(4 * w)
                .enumerate()
                .for_each(|(y_pixel, chunk)| {
                    
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
                        let v = sm.execute(&mut stack, x, y, ts);                                                                       
                        let scaled_v = (v + S::set1_ps(1.0)) * S::set1_ps(0.5);
                        let index = S::cvtps_epi32(scaled_v * S::set1_ps(GRADIENT_SIZE as f32));

                        for j in 0..S::VF32_WIDTH {
                            let c = data.gradient[index[j] as usize % GRADIENT_SIZE];                            
                            chunk[i + j * 4] = (c.r * 255.0) as u8;
                            chunk[i + 1 + j * 4] = (c.g * 255.0) as u8;
                            chunk[i + 2 + j * 4] = (c.b * 255.0) as u8;
                            chunk[i + 3 + j * 4] = 255 as u8;
                        }
                        x = x + x_step;
                    }
                    
                });
           // println!("min:{} max:{} range:{}",min,max,max-min);
            println!("img elapsed:{}", now.elapsed().as_millis());
            result
        }
    }

    fn get_rgba8_mono<S:Simd>(data:&MonoData, w: usize, h: usize, t: f32) -> Vec<u8> {
        unsafe {
            let now = Instant::now();
            let ts = S::set1_ps(t);
            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);
            let sm = StackMachine::<S>::build(&data.c);
            let mut min = 999999.0;
            let mut max = -99999.0;
            result
                .par_chunks_mut(4 * w)
                .enumerate()
                .for_each(|(y_pixel, chunk)| {
                    
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
                        let v = sm.execute(&mut stack, x, y, ts);
                        
                        
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
                    
                });
           // println!("min:{} max:{} range:{}",min,max,max-min);
            println!("img elapsed:{}", now.elapsed().as_millis());
            result
        }
    }

    fn get_rgba8_rgb<S:Simd>(data:&RGBData, w: usize, h: usize, t: f32) -> Vec<u8> {
        unsafe {
            let now = Instant::now();
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

            result
                .par_chunks_mut(4 * w)
                .enumerate()
                .for_each(|(y_pixel, chunk)| {
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
                        let rs = (r_sm.execute(&mut stack, x, y, ts) + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let gs = (g_sm.execute(&mut stack, x, y, ts) + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let bs = (b_sm.execute(&mut stack, x, y, ts) + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
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
                });
            println!("img elapsed:{}", now.elapsed().as_millis());
            result
        }
    }

    fn get_rgba8_hsv<S:Simd>(data:&HSVData, w: usize, h: usize, t: f32) -> Vec<u8> {
        unsafe {
            let now = Instant::now();
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

            result
                .par_chunks_mut(4 * w)
                .enumerate()
                .for_each(|(y_pixel, chunk)| {
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
                        let hs = (h_sm.execute(&mut stack, x, y, ts) + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        let ss = (s_sm.execute(&mut stack, x, y, ts) + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        let vs = (v_sm.execute(&mut stack, x, y, ts) + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        let (mut rs, mut gs, mut bs) = hsv_to_rgb::<S>(
                            wrap_0_1::<S>(hs),
                            wrap_0_1::<S>(ss),
                            wrap_0_1::<S>(vs),
                        );
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
                });
            //   println!("img elapsed:{}", now.elapsed().as_millis());
            result
        }
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
