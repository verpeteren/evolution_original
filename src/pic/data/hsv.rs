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
pub struct HSVData {
    pub h: APTNode,
    pub s: APTNode,
    pub v: APTNode,
    pub coord: CoordinateSystem,
}

impl PicData for HSVData {
    fn new(min: usize, max: usize, video: bool, rng: &mut StdRng, pic_names: &Vec<&String>) -> Pic {
        let (h, coord) = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        let (s, _coord) = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        let (v, _coord) = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        Pic::HSV(HSVData { h, s, v, coord })
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
                    let (hs, ss, vs) = if self.coord == CoordinateSystem::Cartesian {
                        let hs = (h_sm.execute(&mut stack, pics.clone(), x, y, ts, wf, hf)
                            + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        let ss = (s_sm.execute(&mut stack, pics.clone(), x, y, ts, wf, hf)
                            + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        let vs = (v_sm.execute(&mut stack, pics.clone(), x, y, ts, wf, hf)
                            + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        (hs, ss, vs)
                    } else {
                        let (x, y) = cartesian_to_polar::<S>(x, y);
                        let hs = (h_sm.execute(&mut stack, pics.clone(), x, y, ts, wf, hf)
                            + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        let ss = (s_sm.execute(&mut stack, pics.clone(), x, y, ts, wf, hf)
                            + S::set1_ps(1.0))
                            * S::set1_ps(0.5);
                        let vs = (v_sm.execute(&mut stack, pics.clone(), x, y, ts, wf, hf)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pic_new_hsv() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let pic = HSVData::new(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        match &pic {
            Pic::HSV(HSVData { h, s, v, coord: _ }) => {
                let len = h.get_children().unwrap().len();
                assert!(len > 0 && len < 60);

                let len = s.get_children().unwrap().len();
                assert!(len > 0 && len < 60);

                let len = v.get_children().unwrap().len();
                assert!(len > 0 && len < 60);
            }
            _ => {
                panic!("wrong type");
            }
        };
    }
}
