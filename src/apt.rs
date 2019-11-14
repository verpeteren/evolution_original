use rand::prelude::*;
use variant_count::*;

#[derive(VariantCount)]
pub enum APTNode {
    Add(Vec<APTNode>),
    Sub(Vec<APTNode>),
    Mul(Vec<APTNode>),
    Div(Vec<APTNode>),
    FBM(Vec<APTNode>),
    Ridge(Vec<APTNode>),
    Turbulence(Vec<APTNode>),
    Sqrt(Vec<APTNode>),
    Sin(Vec<APTNode>),
    Atan(Vec<APTNode>),
    Atan2(Vec<APTNode>),
    Tan(Vec<APTNode>),
    Log(Vec<APTNode>),
    Constant(f32),
    X,
    Y,
    T,
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
            APTNode::Ridge(children) => APTNode::Ridge(children.clone()),
            APTNode::Turbulence(children) => APTNode::Turbulence(children.clone()),
            APTNode::Sqrt(children) => APTNode::Sqrt(children.clone()),
            APTNode::Sin(children) => APTNode::Sin(children.clone()),
            APTNode::Atan(children) => APTNode::Atan(children.clone()),
            APTNode::Atan2(children) => APTNode::Atan(children.clone()),
            APTNode::Tan(children) => APTNode::Tan(children.clone()),
            APTNode::Log(children) => APTNode::Log(children.clone()),
            APTNode::Constant(v) => APTNode::Constant(*v),
            APTNode::X => APTNode::X,
            APTNode::Y => APTNode::Y,
            APTNode::T => APTNode::T,
            APTNode::Empty => panic!("tried to eval an empty node"),
        }
    }
}


impl APTNode {


    pub fn to_lisp(&self) -> String {
        match self {
            APTNode::Add(children) => format!("( + {} {} )",children[0].to_lisp(),children[1].to_lisp()),
            APTNode::Sub(children) => format!("( - {} {} )",children[0].to_lisp(),children[1].to_lisp()),
            APTNode::Mul(children) => format!("( * {} {} )",children[0].to_lisp(),children[1].to_lisp()),
            APTNode::Div(children) => format!("( / {} {} )",children[0].to_lisp(),children[1].to_lisp()),
            APTNode::FBM(children) => format!("( FBM {} {} {} )",children[0].to_lisp(),children[1].to_lisp(),children[2].to_lisp()),
            APTNode::Ridge(children) => format!("( Ridge {} {} {} )",children[0].to_lisp(),children[1].to_lisp(),children[2].to_lisp()),
            APTNode::Turbulence(children) => format!("( Turbulence {} {} {} )",children[0].to_lisp(),children[1].to_lisp(),children[2].to_lisp()),
            APTNode::Sqrt(children) => format!("( Sqrt {} )",children[0].to_lisp()),
            APTNode::Sin(children) => format!("( Sin {} )",children[0].to_lisp()),
            APTNode::Atan(children) => format!("( Atan {} )",children[0].to_lisp()),
            APTNode::Atan2(children) => format!("( Atan2 {} {} )",children[0].to_lisp(),children[1].to_lisp()),
            APTNode::Log(children) => format!("( Log {} )",children[0].to_lisp()),
            APTNode::Tan(children) => format!("( Tan {} )",children[0].to_lisp()),
            APTNode::Constant(v) => format!("{}",v),
            APTNode::X => format!("X"),
            APTNode::Y => format!("Y"),
            APTNode::T => format!("T"),
            APTNode::Empty => format!("EMPTY"),
        }
    }

    pub fn get_random_node(rng: &mut StdRng) -> APTNode {
        let r = rng.gen_range(0, APTNode::VARIANT_COUNT - 5);
        match r {
            0 => APTNode::Add(vec![APTNode::Empty, APTNode::Empty]),
            1 => APTNode::Sub(vec![APTNode::Empty, APTNode::Empty]),
            2 => APTNode::Mul(vec![APTNode::Empty, APTNode::Empty]),
            3 => APTNode::Div(vec![APTNode::Empty, APTNode::Empty]),
            4 => APTNode::FBM(vec![APTNode::Empty, APTNode::Empty, APTNode::Empty]),
            5 => APTNode::Ridge(vec![APTNode::Empty, APTNode::Empty, APTNode::Empty]),
            6 => APTNode::Turbulence(vec![APTNode::Empty, APTNode::Empty, APTNode::Empty]),
            7 => APTNode::Sqrt(vec![APTNode::Empty]),
            8 => APTNode::Sin(vec![APTNode::Empty]),
            9 => APTNode::Atan(vec![APTNode::Empty]),            
            10 => APTNode::Atan2(vec![APTNode::Empty,APTNode::Empty]),            
            11 => APTNode::Tan(vec![APTNode::Empty]),
            12 => APTNode::Log(vec![APTNode::Empty]),
            _ => panic!("get_random_node generated unhandled r:{}", r),
        }
    }

    pub fn get_random_leaf(rng: &mut StdRng) -> APTNode {
        let r = rng.gen_range(0, 3);
        match r {
            0 => APTNode::X,
            1 => APTNode::Y,            
            2 => APTNode::Constant(rng.gen_range(-1.0, 1.0)),
            _ => panic!("get_random_leaf generated unhandled r:{}", r),
        }
    }

    pub fn get_random_leaf_video(rng: &mut StdRng) -> APTNode {
        let r = rng.gen_range(0, 4);
        match r {
            0 => APTNode::X,
            1 => APTNode::Y,            
            2 => APTNode::T,
            3 => APTNode::Constant(rng.gen_range(-1.0, 1.0)),
            _ => panic!("get_random_leaf generated unhandled r:{}", r),
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


    pub fn generate_tree(count: usize,video:bool, rng: &mut StdRng) -> APTNode {
        let leaf_func = if video {
            APTNode::get_random_leaf_video
        } else {
            APTNode::get_random_leaf
        };
        let mut first = APTNode::get_random_node(rng);
        for _ in 1..count {
            first.add_random(APTNode::get_random_node(rng), rng);
        }
        while first.add_leaf(&leaf_func(rng)) {}
        first
    }

    pub fn get_children_mut(&mut self) -> Option<&mut Vec<APTNode>> {
        match self {
            APTNode::Add(children) => Some(children),
            APTNode::Sub(children) => Some(children),
            APTNode::Mul(children) => Some(children),
            APTNode::Div(children) => Some(children),
            APTNode::FBM(children) => Some(children),
            APTNode::Ridge(children) => Some(children),
            APTNode::Turbulence(children) => Some(children),
            APTNode::Sqrt(children) => Some(children),
            APTNode::Sin(children) => Some(children),
            APTNode::Atan(children) => Some(children),
            APTNode::Atan2(children) => Some(children),
            APTNode::Tan(children) => Some(children),
            APTNode::Log(children) => Some(children),
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
            APTNode::Ridge(children) => Some(children),
            APTNode::Turbulence(children) => Some(children),
            APTNode::Sqrt(children) => Some(children),
            APTNode::Sin(children) => Some(children),
            APTNode::Atan(children) => Some(children),
            APTNode::Atan2(children) => Some(children),
            APTNode::Tan(children) => Some(children),
            APTNode::Log(children) => Some(children),
            _ => None,
        }
    }

    pub fn is_leaf(&self) -> bool {
        match self {
            APTNode::X | APTNode::Y | APTNode::T | APTNode::Constant(_) | APTNode::Empty => true,
            _ => false,
        }
    }
}
