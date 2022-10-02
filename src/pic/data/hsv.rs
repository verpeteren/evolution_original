use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::sync::Arc;

use crate::parser::aptnode::APTNode;
use crate::pic::actual_picture::ActualPicture;
use crate::pic::coordinatesystem::CoordinateSystem;
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
        unimplemented!();
    }
}
