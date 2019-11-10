use crate::apt::*;
use crate::stack_machine::*;
use std::time::{Duration, Instant};
use rand::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use simdeez::*;
use simdnoise::*;
use variant_count::*;

pub trait Pic<S: Simd> {
    fn get_rgba8(&self, w: usize, h: usize) -> Vec<u8>;    
}

pub struct MonoPic<S: Simd> {
    c: APTNode<S>,
}
impl<S: Simd> MonoPic<S> {
    pub fn new(size: usize) -> MonoPic<S> {
        let seed = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,20,31,32];
        let mut rng = StdRng::from_seed(seed);        
        let tree = APTNode::generate_tree(size, &mut rng);
        //let tree = APTNode::Add(vec![APTNode::X,APTNode::Y]);
        MonoPic { c: tree }
    }
}

impl<S: Simd> Pic<S> for MonoPic<S> {   
    fn get_rgba8(&self, w: usize, h: usize) -> Vec<u8> {
        unsafe {
            let mut sm = StackMachine::<S>::new();           
            sm.build(&self.c);
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
                        (sm.execute(x, S::set1_ps(y)) + S::set1_ps(1.0)) * S::set1_ps(128.0);
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
        let seed = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,20,31,32];
        let mut rng = StdRng::from_seed(seed);        
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
            let now = Instant::now();

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
                    let rs = (r_sm.execute_no_bounds(x, S::set1_ps(y)) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                    let gs = (g_sm.execute_no_bounds(x, S::set1_ps(y)) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                    let bs = (b_sm.execute_no_bounds(x, S::set1_ps(y)) + S::set1_ps(1.0)) * S::set1_ps(128.0);
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
            println!("sm elapsed:{}",now.elapsed().as_millis());
            result
        }
    }
}
