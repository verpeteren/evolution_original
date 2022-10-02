use rand::prelude::*;
use rand::rngs::StdRng;

use crate::parser::aptnode::APTNode;
use crate::pic::coordinatesystem::CoordinateSystem;
use crate::pic::data::PicData;
use crate::pic::pic::Pic;

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
}
