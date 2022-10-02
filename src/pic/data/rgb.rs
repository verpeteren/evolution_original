use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::sync::Arc;

use crate::parser::aptnode::APTNode;
use crate::pic::actual_picture::ActualPicture;
use crate::pic::coordinatesystem::{cartesian_to_polar, CoordinateSystem};
use crate::pic::data::PicData;
use crate::pic::pic::Pic;
use crate::vm::stackmachine::StackMachine;

use rayon::prelude::*;
use simdeez::Simd;

#[derive(Clone, Debug, PartialEq)]
pub struct RGBData {
    pub r: APTNode,
    pub g: APTNode,
    pub b: APTNode,
    pub coord: CoordinateSystem,
}

impl PicData for RGBData {
    fn new(min: usize, max: usize, video: bool, rng: &mut StdRng, pic_names: &Vec<&String>) -> Pic {
        let (r, coord) = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        let (g, _coord) = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        let (b, _coord) = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        Pic::RGB(RGBData { r, g, b, coord })
    }
    fn get_rgba8<S: Simd>(
        &self,
        threaded: bool,
        pics: Arc<HashMap<String, ActualPicture>>,
        w: usize,
        h: usize,
        t: f32,
    ) -> Vec<u8> {
        unsafe {
            let ts = S::set1_ps(t);
            let wf = S::set1_ps(w as f32);
            let hf = S::set1_ps(h as f32);

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
                    let (rs, gs, bs) = if self.coord == CoordinateSystem::Cartesian {
                        let rs = (r_sm.execute(&mut stack, pics.clone(), x, y, ts, wf, hf)
                            + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let gs = (g_sm.execute(&mut stack, pics.clone(), x, y, ts, wf, hf)
                            + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let bs = (b_sm.execute(&mut stack, pics.clone(), x, y, ts, wf, hf)
                            + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        (rs, gs, bs)
                    } else {
                        let (x, y) = cartesian_to_polar::<S>(x, y);
                        let rs = (r_sm.execute(&mut stack, pics.clone(), x, y, ts, wf, hf)
                            + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let gs = (g_sm.execute(&mut stack, pics.clone(), x, y, ts, wf, hf)
                            + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        let bs = (b_sm.execute(&mut stack, pics.clone(), x, y, ts, wf, hf)
                            + S::set1_ps(1.0))
                            * S::set1_ps(128.0);
                        (rs, gs, bs)
                    };

                    for j in 0..S::VF32_WIDTH {
                        let r = (rs[j] as i32 % 255) as u8;
                        let g = (gs[j] as i32 % 255) as u8;
                        let b = (bs[j] as i32 % 255) as u8;
                        let j4 = j * 4;
                        chunk[i + j4] = r;
                        chunk[i + 1 + j4] = g;
                        chunk[i + 2 + j4] = b;
                        chunk[i + 3 + j4] = 255 as u8;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pic_new_rgb() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let pic = RGBData::new(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        match &pic {
            Pic::RGB(RGBData { r, g, b, coord: _ }) => {
                let len = r.get_children().unwrap().len();
                assert!(len > 0 && len < 60);

                let len = g.get_children().unwrap().len();
                assert!(len > 0 && len < 60);

                let len = b.get_children().unwrap().len();
                assert!(len > 0 && len < 60);
            }
            _ => {
                panic!("wrong type");
            }
        };
    }
}
