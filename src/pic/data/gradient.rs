use rand::prelude::*;
use rand::rngs::StdRng;

use crate::parser::aptnode::APTNode;
use crate::pic::coordinatesystem::CoordinateSystem;
use crate::pic::data::PicData;
use crate::pic::ggez_utility::get_random_color;
use crate::pic::pic::Pic;

use ggez::graphics::Color;

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

        let (tree, coord) = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        Pic::Gradient(GradientData {
            colors: colors,
            index: tree,
            coord,
        })
    }
}
