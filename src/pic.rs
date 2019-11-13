use crate::apt::*;
use crate::stack_machine::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;
use simdeez::*;
use std::time::Instant;

pub trait Pic {
    fn get_rgba8<S: Simd>(&self, w: usize, h: usize) -> Vec<u8>;
    fn get_rgba8_single_thread<S: Simd>(&self, w: usize, h: usize) -> Vec<u8>;
    fn to_lisp(&self) -> String;
}

pub struct MonoPic {
    c: APTNode,
}
impl MonoPic {
    pub fn new(size: usize) -> MonoPic {
        let seed = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 20, 31, 32,
        ];
        let mut rng = StdRng::from_seed(seed);
        let tree = APTNode::generate_tree(size, &mut rng);
        // let tree = APTNode::Sin(vec![APTNode::X]);

        MonoPic { c: tree }
    }
}

impl Pic for MonoPic {
    fn to_lisp(&self) -> String {
        format!("Mono\n {}",self.c.to_lisp())
    }
    fn get_rgba8<S: Simd>(&self, w: usize, h: usize) -> Vec<u8> {
        unsafe {
            let now = Instant::now();

            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);

            let sm = StackMachine::<S>::build(&self.c);

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
                        let v = sm.execute(&mut stack, x, y);
                        let cs = (v + S::set1_ps(1.0)) * S::set1_ps(127.5);

                        for j in 0..S::VF32_WIDTH {
                            let c = (cs[j] as i32 % 255) as u8;
                            chunk[i + j * 4] = c;
                            chunk[i + 1 + j * 4] = c;
                            chunk[i + 2 + j * 4] = c;
                            chunk[i + 3 + j * 4] = 255 as u8;
                        }
                        x = x + x_step;
                    }
                });
            println!("parallel elapsed:{}", now.elapsed().as_millis());
            result
        }
    }

    fn get_rgba8_single_thread<S: Simd>(&self, w: usize, h: usize) -> Vec<u8> {
        unsafe {
            let now = Instant::now();

            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);

            let sm = StackMachine::<S>::build(&self.c);
            let mut max = S::set1_ps(-9999999.0);
            let mut min = S::set1_ps(9999999.0);
            result
                .chunks_mut(4 * w)
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
                        let v = sm.execute(&mut stack, x, y);

                        max = S::max_ps(max, v);
                        min = S::min_ps(min, v);
                        //todo think about a more rigorous way of converting [-1.0,1.0] -> [0,255]
                        let cs = (v + S::set1_ps(1.0)) * S::set1_ps(127.5);
                        //println!("x:{} = v:{} = cs:{}",x[0],v[0],cs[0]);

                        for j in 0..S::VF32_WIDTH {
                            let c = (cs[j] as i32 % 255) as u8;
                            chunk[i + j * 4] = c;
                            chunk[i + 1 + j * 4] = c;
                            chunk[i + 2 + j * 4] = c;
                            chunk[i + 3 + j * 4] = 255 as u8;
                        }
                        x = x + x_step;
                    }
                });
            println!("parallel elapsed:{}", now.elapsed().as_millis());

            let mut smax = -99999.0;
            let mut smin = 99999.0;
            for i in 0..S::VF32_WIDTH {
                if max[i] > smax {
                    smax = max[i];
                }
                if min[i] < smin {
                    smin = min[i];
                }
            }

            println!("min:{}  max:{} range:{}", smin, smax, smax - smin);
            result
        }
    }
}

pub struct RgbPic {
    r: APTNode,
    g: APTNode,
    b: APTNode,
}
impl RgbPic {
    pub fn new(size: usize) -> RgbPic {
        let seed = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 20, 31, 32,
        ];
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let r = APTNode::generate_tree(size, &mut rng);
        let g = APTNode::generate_tree(size, &mut rng);
        let b = APTNode::generate_tree(size, &mut rng);
        //let noise = APTNode::FBM::<S>(vec![APTNode::X,APTNode::Y]);
        RgbPic { r, g, b }
    }
}
impl Pic for RgbPic {
    fn to_lisp(&self) -> String {
        format!("RGB\n{} \n{}\n{}",self.r.to_lisp(),self.g.to_lisp(),self.b.to_lisp())
    }

    fn get_rgba8<S: Simd>(&self, w: usize, h: usize) -> Vec<u8> {
        unsafe {
            let now = Instant::now();

            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);

            let r_sm = StackMachine::<S>::build(&self.r);
            let g_sm = StackMachine::<S>::build(&self.g);
            let b_sm = StackMachine::<S>::build(&self.b);
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
                        let rs =
                            (r_sm.execute(&mut stack, x, y) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                        let gs =
                            (g_sm.execute(&mut stack, x, y) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                        let bs =
                            (b_sm.execute(&mut stack, x, y) + S::set1_ps(1.0)) * S::set1_ps(128.0);
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
            println!("parallel elapsed:{}", now.elapsed().as_millis());
            result
        }
    }

    fn get_rgba8_single_thread<S: Simd>(&self, w: usize, h: usize) -> Vec<u8> {
        unsafe {
            let now = Instant::now();

            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);

            let r_sm = StackMachine::<S>::build(&self.r);
            let g_sm = StackMachine::<S>::build(&self.g);
            let b_sm = StackMachine::<S>::build(&self.b);
            let max_len = *[
                r_sm.instructions.len(),
                g_sm.instructions.len(),
                b_sm.instructions.len(),
            ]
            .iter()
            .max()
            .unwrap();

            result
                .chunks_mut(4 * w)
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
                        let rs =
                            (r_sm.execute(&mut stack, x, y) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                        let gs =
                            (g_sm.execute(&mut stack, x, y) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                        let bs =
                            (b_sm.execute(&mut stack, x, y) + S::set1_ps(1.0)) * S::set1_ps(128.0);
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
            println!("parallel elapsed:{}", now.elapsed().as_millis());
            result
        }
    }
}

pub struct HsvPic {
    h: APTNode,
    s: APTNode,
    v: APTNode,
}
impl HsvPic {
    pub fn new(size: usize) -> HsvPic {
        let seed = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 20, 31, 32,
        ];
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let h = APTNode::generate_tree(size, &mut rng);
        let s = APTNode::generate_tree(size, &mut rng);
        let v = APTNode::generate_tree(size, &mut rng);               
        HsvPic { h, s, v }
    }
}

impl Pic for HsvPic {
    
    fn to_lisp(&self) -> String {
        format!("HSV\n{} \n{}\n{}",self.h.to_lisp(),self.s.to_lisp(),self.v.to_lisp())
    }

    fn get_rgba8<S: Simd>(&self, w: usize, h: usize) -> Vec<u8> {
        unsafe {
            let now = Instant::now();

            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);

            let h_sm = StackMachine::<S>::build(&self.h);
            let s_sm = StackMachine::<S>::build(&self.s);
            let v_sm = StackMachine::<S>::build(&self.v);
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
                        let hs =
                            (h_sm.execute(&mut stack, x, y) + S::set1_ps(1.0)) * S::set1_ps(0.5);
                        let ss =
                            (s_sm.execute(&mut stack, x, y) + S::set1_ps(1.0)) * S::set1_ps(0.5);
                        let vs =
                            (v_sm.execute(&mut stack, x, y) + S::set1_ps(1.0)) * S::set1_ps(0.5);
                        let (mut rs, mut gs, mut bs) = hsv_to_rgb::<S>(wrap_0_1::<S>(hs), wrap_0_1::<S>(ss), wrap_0_1::<S>(vs));
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
            println!("parallel elapsed:{}", now.elapsed().as_millis());
            result
        }
    }

    fn get_rgba8_single_thread<S: Simd>(&self, w: usize, h: usize) -> Vec<u8> {
        unsafe {
            let now = Instant::now();

            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);

            let h_sm = StackMachine::<S>::build(&self.h);
            let s_sm = StackMachine::<S>::build(&self.s);
            let v_sm = StackMachine::<S>::build(&self.v);
            let max_len = *[
                h_sm.instructions.len(),
                s_sm.instructions.len(),
                v_sm.instructions.len(),
            ]
            .iter()
            .max()
            .unwrap();

            result
                .chunks_mut(4 * w)
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
                        let hs =
                            (h_sm.execute(&mut stack, x, y) + S::set1_ps(1.0)) * S::set1_ps(0.5);
                        let ss =
                            (s_sm.execute(&mut stack, x, y) + S::set1_ps(1.0)) * S::set1_ps(0.5);
                        let vs =
                            (v_sm.execute(&mut stack, x, y) + S::set1_ps(1.0)) * S::set1_ps(0.5);
                        let (mut rs, mut gs, mut bs) = hsv_to_rgb::<S>(wrap_0_1::<S>(hs), wrap_0_1::<S>(ss), wrap_0_1::<S>(vs));
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
            println!("parallel elapsed:{}", now.elapsed().as_millis());
            result
        }
    }
}

#[inline(always)]
fn wrap_0_1<S:Simd>(v:S::Vf32) -> S::Vf32 {
    unsafe {
        let mut r = S::setzero_ps();
        for i in 0 .. S::VF32_WIDTH {
            r[i] = v[i] % 1.0001;
        }
        r
    }
}

fn hsv_to_rgb<S: Simd>(h: S::Vf32, s: S::Vf32, v: S::Vf32) -> (S::Vf32, S::Vf32, S::Vf32) {
    unsafe {        
        let six = S::set1_ps(6.0);
        let one = S::set1_ps(1.0);
        let hi = S::cvtps_epi32(S::fastfloor_ps(h * six));
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
