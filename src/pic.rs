use crate::apt::*;
use crate::stack_machine::*;
use rand::prelude::*;
use simdeez::*;
use simdnoise::*;
use variant_count::*;

pub trait Pic<S: Simd> {
    fn get_rgba8(&self, w: usize, h: usize) -> Vec<u8>;
    fn get_rgba8_sm(&self, w: usize, h: usize) -> Vec<u8>;
}

pub struct MonoPic<S: Simd> {
    c: APTNode<S>,
}
impl<S: Simd> MonoPic<S> {
    fn new(size: usize) -> MonoPic<S> {
        let mut rng = rand::thread_rng();
        let tree = APTNode::generate_tree(size, &mut rng);
        MonoPic { c: tree }
    }
}

impl<S: Simd> Pic<S> for MonoPic<S> {
    fn get_rgba8(&self, w: usize, h: usize) -> Vec<u8> {
        unsafe {
            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);

            let x_step = 2.0 / (w - 1) as f32;
            let mut x = S::setzero_ps();
            for i in (0..S::VF32_WIDTH).rev() {
                x[i] = -1.0 + (x_step * i as f32);
            }
            let init_x = x;
            //println!("xstep1:{}",x_step);
            let x_step = S::set1_ps(x_step * S::VF32_WIDTH as f32);
            //println!("xstep2:{:?}",x_step);
            let y_step = 2.0 / h as f32;
            let mut y = -1.0;
            let mut i = 0;
            for _ in 0..h {
                for _ in 0..w / S::VF32_WIDTH {
                    let color =
                        (self.c.eval_2d(x, S::set1_ps(y)) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                    for j in 0..S::VF32_WIDTH {
                        let c = (color[j] as i32 % 255) as u8;
                        result[i + j * 4] = c;
                        result[i + 1 + j * 4] = c;
                        result[i + 2 + j * 4] = c;
                        result[i + 3 + j * 4] = 255 as u8;
                        //          println!("{},{},{},{}",i+j*4,i+1+j*4,i+2+j*4,i+3+j*4);
                    }
                    //    println!("x:{:?}",x);
                    x = x + x_step;
                    i += S::VF32_WIDTH * 4;
                }
                y += y_step;
                x = init_x;
            }
            result
        }
    }

    fn get_rgba8_sm(&self, w: usize, h: usize) -> Vec<u8> {
        unsafe {
            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);

            let x_step = 2.0 / (w - 1) as f32;
            let mut x = S::setzero_ps();
            for i in (0..S::VF32_WIDTH).rev() {
                x[i] = -1.0 + (x_step * i as f32);
            }
            let init_x = x;
            //println!("xstep1:{}",x_step);
            let x_step = S::set1_ps(x_step * S::VF32_WIDTH as f32);
            //println!("xstep2:{:?}",x_step);
            let y_step = 2.0 / h as f32;
            let mut y = -1.0;
            let mut i = 0;
            for _ in 0..h {
                for _ in 0..w / S::VF32_WIDTH {
                    let color =
                        (self.c.eval_2d(x, S::set1_ps(y)) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                    for j in 0..S::VF32_WIDTH {
                        let c = (color[j] as i32 % 255) as u8;
                        result[i + j * 4] = c;
                        result[i + 1 + j * 4] = c;
                        result[i + 2 + j * 4] = c;
                        result[i + 3 + j * 4] = 255 as u8;
                        //          println!("{},{},{},{}",i+j*4,i+1+j*4,i+2+j*4,i+3+j*4);
                    }
                    //    println!("x:{:?}",x);
                    x = x + x_step;
                    i += S::VF32_WIDTH * 4;
                }
                y += y_step;
                x = init_x;
            }
            result
        }
    }
}

pub struct RgbPic<S: Simd> {
    r: APTNode<S>,
    g: APTNode<S>,
    b: APTNode<S>,
}
impl<S: Simd> RgbPic<S> {
    pub fn new(size: usize) -> RgbPic<S> {
        let mut rng = rand::thread_rng();
        let r = APTNode::<S>::generate_tree(size, &mut rng);
        let g = APTNode::<S>::generate_tree(size, &mut rng);
        let b = APTNode::<S>::generate_tree(size, &mut rng);
        //let noise = APTNode::FBM::<S>(vec![APTNode::X,APTNode::Y]);
        unsafe { RgbPic { r, g, b } }
    }
}
impl<S: Simd> Pic<S> for RgbPic<S> {
    fn get_rgba8(&self, w: usize, h: usize) -> Vec<u8> {
        unsafe {
            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);

            let x_step = 2.0 / (w - 1) as f32;
            let mut x = S::setzero_ps();
            for i in (0..S::VF32_WIDTH).rev() {
                x[i] = -1.0 + (x_step * i as f32);
            }
            let init_x = x;
            //println!("xstep1:{}",x_step);
            let x_step = S::set1_ps(x_step * S::VF32_WIDTH as f32);
            //println!("xstep2:{:?}",x_step);
            let y_step = 2.0 / h as f32;
            let mut y = -1.0;
            let mut i = 0;
            for _ in 0..h {
                for _ in 0..w / S::VF32_WIDTH {
                    let rs =
                        (self.r.eval_2d(x, S::set1_ps(y)) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                    let gs =
                        (self.g.eval_2d(x, S::set1_ps(y)) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                    let bs =
                        (self.b.eval_2d(x, S::set1_ps(y)) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                    for j in 0..S::VF32_WIDTH {
                        let r = (rs[j] as i32 % 255) as u8;
                        let g = (gs[j] as i32 % 255) as u8;
                        let b = (bs[j] as i32 % 255) as u8;
                        result[i + j * 4] = r;
                        result[i + 1 + j * 4] = g;
                        result[i + 2 + j * 4] = b;
                        result[i + 3 + j * 4] = 255 as u8;
                        //          println!("{},{},{},{}",i+j*4,i+1+j*4,i+2+j*4,i+3+j*4);
                    }
                    //    println!("x:{:?}",x);
                    x = x + x_step;
                    i += S::VF32_WIDTH * 4;
                }
                y += y_step;
                x = init_x;
            }
            result
        }
    }

    fn get_rgba8_sm(&self, w: usize, h: usize) -> Vec<u8> {
        unsafe {
            let mut r_sm = StackMachine::<S>::new();
            let mut g_sm = StackMachine::<S>::new();
            let mut b_sm = StackMachine::<S>::new();
            r_sm.build(&self.r);
            g_sm.build(&self.g);
            b_sm.build(&self.b);

            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);

            let x_step = 2.0 / (w - 1) as f32;
            let mut x = S::setzero_ps();
            for i in (0..S::VF32_WIDTH).rev() {
                x[i] = -1.0 + (x_step * i as f32);
            }
            let init_x = x;
            //println!("xstep1:{}",x_step);
            let x_step = S::set1_ps(x_step * S::VF32_WIDTH as f32);
            //println!("xstep2:{:?}",x_step);
            let y_step = 2.0 / h as f32;
            let mut y = -1.0;
            let mut i = 0;
            for _ in 0..h {
                for _ in 0..w / S::VF32_WIDTH {
                    let rs = (r_sm.execute(x, S::set1_ps(y)) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                    let gs = (g_sm.execute(x, S::set1_ps(y)) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                    let bs = (b_sm.execute(x, S::set1_ps(y)) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                    for j in 0..S::VF32_WIDTH {
                        let r = (rs[j] as i32 % 255) as u8;
                        let g = (gs[j] as i32 % 255) as u8;
                        let b = (bs[j] as i32 % 255) as u8;
                        result[i + j * 4] = r;
                        result[i + 1 + j * 4] = g;
                        result[i + 2 + j * 4] = b;
                        result[i + 3 + j * 4] = 255 as u8;
                        //          println!("{},{},{},{}",i+j*4,i+1+j*4,i+2+j*4,i+3+j*4);
                    }
                    //    println!("x:{:?}",x);
                    x = x + x_step;
                    i += S::VF32_WIDTH * 4;
                }
                y += y_step;
                x = init_x;
            }
            result
        }
    }
}
