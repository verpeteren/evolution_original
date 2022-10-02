use rand::prelude::*;
use rand::rngs::StdRng;

use crate::parser::aptnode::APTNode;
use crate::pic::coordinatesystem::CoordinateSystem;
use crate::pic::data::PicData;
use crate::pic::pic::Pic;

#[derive(Clone, Debug, PartialEq)]
pub struct MonoData {
    pub c: APTNode,
    pub coord: CoordinateSystem,
}

impl PicData for MonoData {
    fn new(min: usize, max: usize, video: bool, rng: &mut StdRng, pic_names: &Vec<&String>) -> Pic {
        let (tree, coord) = APTNode::generate_tree(rng.gen_range(min..max), video, rng, pic_names);
        Pic::Mono(MonoData { c: tree, coord })
    }
}
