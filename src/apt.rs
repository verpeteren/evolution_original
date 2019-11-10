use rand::prelude::*;
use simdeez::*;
use simdnoise::*;
use variant_count::*;

#[derive(VariantCount)]
pub enum APTNode {
    Add(Vec<APTNode>),
    Sub(Vec<APTNode>),
    Mul(Vec<APTNode>),
    Div(Vec<APTNode>),
    FBM(Vec<APTNode>),
    Constant(f32),
    X,
    Y,
    Empty,
}

impl Clone for APTNode {
    fn clone(&self) -> Self {
        match self {
            APTNode::Add(children) => APTNode::Add(children.clone()),
            APTNode::Sub(children) => APTNode::Sub(children.clone()),
            APTNode::Mul(children) => APTNode::Mul(children.clone()),
            APTNode::Div(children) => APTNode::Div(children.clone()),
            APTNode::FBM(children) => APTNode::FBM(children.clone()),
            APTNode::Constant(v) => APTNode::Constant(*v),
            APTNode::X => APTNode::X,
            APTNode::Y => APTNode::Y,
            APTNode::Empty => panic!("tried to eval an empty node"),
        }
    }
}

impl APTNode {
    pub fn get_random_node(rng: &mut StdRng) -> APTNode {
        let r = rng.gen_range(0, APTNode::VARIANT_COUNT - 4);
        match r {
            0 => APTNode::Add(vec![APTNode::Empty, APTNode::Empty]),
            1 => APTNode::Sub(vec![APTNode::Empty, APTNode::Empty]),
            2 => APTNode::Mul(vec![APTNode::Empty, APTNode::Empty]),
            3 => APTNode::Div(vec![APTNode::Empty, APTNode::Empty]),
            4 => APTNode::FBM(vec![APTNode::Empty, APTNode::Empty]),
            _ => panic!("get_random_node generated unhandled r:{}", r),
        }
    }

    pub fn get_random_leaf(rng: &mut StdRng) -> APTNode {
        unsafe {
            let count = APTNode::VARIANT_COUNT;
            let r = rng.gen_range(0, 3);
            match r {
                0 => APTNode::X,
                1 => APTNode::Y,
                2 => APTNode::Constant(rng.gen_range(-1.0, 1.0)),
                _ => panic!("get_random_leaf generated unhandled r:{}", r),
            }
        }
    }

    pub fn add_random(&mut self, node: APTNode, rng: &mut StdRng) {
        let children = match self.get_children_mut() {
            Some(children) => children,
            None => panic!("tried to add_random to a leaf"),
        };
        let add_index = rng.gen_range(0, children.len());
        match children[add_index] {
            APTNode::Empty => children[add_index] = node,
            _ => children[add_index].add_random(node, rng),
        }
    }

    pub fn add_leaf(&mut self, leaf: &APTNode) -> bool {
        match self.get_children_mut() {
            None => false,
            Some(children) => {
                for i in 0..children.len() {
                    match children[i] {
                        APTNode::Empty => {
                            children[i] = leaf.clone();
                            return true;
                        }
                        _ => {
                            if !children[i].is_leaf() && children[i].add_leaf(leaf) {
                                return true;
                            }
                        }
                    }
                }
                return false;
            }
        }
    }

    pub fn generate_tree(count: usize, rng: &mut StdRng) -> APTNode {
        let mut first = APTNode::get_random_node(rng);
        for _ in 1..count {
            first.add_random(APTNode::get_random_node(rng), rng);
        }
        while first.add_leaf(&APTNode::get_random_leaf(rng)) {}
        first
    }

    pub fn get_children_mut(&mut self) -> Option<&mut Vec<APTNode>> {
        match self {
            APTNode::Add(children) => Some(children),
            APTNode::Sub(children) => Some(children),
            APTNode::Mul(children) => Some(children),
            APTNode::Div(children) => Some(children),
            APTNode::FBM(children) => Some(children),
            _ => None,
        }
    }

    pub fn get_children(&self) -> Option<&Vec<APTNode>> {
        match self {
            APTNode::Add(children) => Some(children),
            APTNode::Sub(children) => Some(children),
            APTNode::Mul(children) => Some(children),
            APTNode::Div(children) => Some(children),
            APTNode::FBM(children) => Some(children),
            _ => None,
        }
    }

    pub fn is_leaf(&self) -> bool {
        match self {
            APTNode::X | APTNode::Y | APTNode::Constant(_) | APTNode::Empty => true,
            _ => false,
        }
    }
}
