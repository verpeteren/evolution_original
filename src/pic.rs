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
       // let tree = APTNode::generate_tree(size, &mut rng);
        let tree = APTNode::Sin(vec![APTNode::X]);
        
        MonoPic { c: tree }
    }
}

impl Pic for MonoPic {
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
                        
                        max = S::max_ps(max,v);
                        min = S::min_ps(min,v);
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
            for i in 0 .. S::VF32_WIDTH {
                if max[i] > smax { smax = max[i];}
                if min[i] < smin { smin = min[i]; }
            }

            println!("min:{}  max:{} range:{}",smin,smax,smax-smin);
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
                        let rs = (r_sm.execute(&mut stack, x, y) + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let gs = (g_sm.execute(&mut stack, x, y) + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let bs = (b_sm.execute(&mut stack, x, y) + S::set1_ps(1.0))
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
                        let rs = (r_sm.execute(&mut stack, x, y) + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let gs = (g_sm.execute(&mut stack, x, y) + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let bs = (b_sm.execute(&mut stack, x, y) + S::set1_ps(1.0))
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
            println!("parallel elapsed:{}", now.elapsed().as_millis());
            result
        }
    }
}
