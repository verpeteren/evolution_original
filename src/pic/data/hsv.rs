use rand::prelude::*;
use rand::rngs::StdRng;

use crate::parser::aptnode::APTNode;
use crate::pic::coordinatesystem::CoordinateSystem;
use crate::pic::data::PicData;
use crate::pic::pic::Pic;

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
}
