use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::sync::Arc;

use crate::parser::aptnode::APTNode;
use crate::pic::actual_picture::ActualPicture;
use crate::pic::coordinatesystem::{cartesian_to_polar, CoordinateSystem};
use crate::pic::data::PicData;
use crate::pic::ggez_utility::{get_random_color, lerp_color};
use crate::pic::pic::Pic;
use crate::vm::stackmachine::StackMachine;

use ggez::graphics::Color;
use rayon::prelude::*;
use simdeez::Simd;

const GRADIENT_STOP_CHANCE: usize = 5; // 1 in 5
const MAX_GRADIENT_COUNT: usize = 10;
const MIN_GRADIENT_COUNT: usize = 2;
pub const GRADIENT_SIZE: usize = 512;

#[derive(Clone, Debug, PartialEq)]
pub struct GradientData {
    pub colors: Vec<(Color, bool)>,
    pub index: APTNode,
    pub coord: CoordinateSystem,
}

impl PicData for GradientData {
    fn new(min: usize, max: usize, video: bool, rng: &mut StdRng, pic_names: &Vec<&String>) -> Pic {
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

        let (tree, coord) =
            APTNode::create_random_tree(rng.gen_range(min..max), video, rng, pic_names);
        Pic::Gradient(GradientData {
            colors: colors,
            index: tree,
            coord,
        })
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
            let sm = StackMachine::<S>::build(&self.index);
            /*
            let mut min = 999999.0;
            let mut max = -99999.0;
            */

            let color_count = self.colors.iter().filter(|(_, stop)| !stop).count();
            let mut gradient = Vec::<Color>::new(); //todo actually compute this
            let step = (GRADIENT_SIZE as f32 / color_count as f32) / GRADIENT_SIZE as f32;
            let mut positions = Vec::<f32>::new();
            positions.push(0.0);
            let mut pos = step;
            for i in 1..self.colors.len() - 1 {
                let (_, stop) = self.colors[i];
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
                    gradient.push(self.colors[0].0);
                } else {
                    let color1 = self.colors[color2pos - 1].0;
                    let color2 = self.colors[color2pos].0;
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
                let chunk_len = chunk.len();
                for i in (0..w * 4).step_by(S::VF32_WIDTH * 4) {
                    let v = if self.coord == CoordinateSystem::Cartesian {
                        sm.execute(&mut stack, pics.clone(), x, y, ts, wf, hf)
                    } else {
                        let (r, theta) = cartesian_to_polar::<S>(x, y);
                        sm.execute(&mut stack, pics.clone(), r, theta, ts, wf, hf)
                    };
                    let scaled_v = (v + S::set1_ps(1.0)) * S::set1_ps(0.5);
                    let index = S::cvtps_epi32(scaled_v * S::set1_ps(GRADIENT_SIZE as f32));

                    for j in 0..S::VF32_WIDTH {
                        let j4 = j * 4;
                        let ij4 = i + j4;
                        if ij4 >= chunk_len {
                            break;
                        }
                        let c = gradient[index[j] as usize % GRADIENT_SIZE];
                        chunk[ij4] = (c.r * 255.0) as u8;
                        chunk[ij4 + 1] = (c.g * 255.0) as u8;
                        chunk[ij4 + 2] = (c.b * 255.0) as u8;
                        chunk[ij4 + 3] = 255 as u8;
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
    fn simplify<S: Simd>(
        &mut self,
        pics: Arc<HashMap<String, ActualPicture>>,
        w: usize,
        h: usize,
        t: f32,
    ) {
        self.index =
            self.index
                .constant_fold::<S>(&self.coord, pics, None, None, Some(w), Some(h), Some(t));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pic_new_gradient() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let pic = GradientData::new(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        match &pic {
            Pic::Gradient(GradientData {
                colors,
                index,
                coord: _coord,
            }) => {
                let len = colors.len();
                assert!(len > 1 && len < 10);
                let len = index.get_children().unwrap().len();
                assert!(len > 0 && len < 60);
            }
            _ => {
                panic!("wrong type");
            }
        };
    }
}
