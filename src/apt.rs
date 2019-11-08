use rand::prelude::*;
use simdeez::*;
use variant_count::*;

pub struct MonoPic<S: Simd> {
    c: APTNode<S>,
}

impl<S: Simd> MonoPic<S> {
    pub fn new(size:usize) -> MonoPic<S>{
        let mut rng = rand::thread_rng();
        let tree = APTNode::generate_tree(size, &mut rng);

        MonoPic {
            c: tree
        }


    }
    pub fn get_rgba8(&self, w: usize, h: usize) -> Vec<u8> {
        unsafe {
            let vec_len = w * h * 4;
            let mut result = Vec::<u8>::with_capacity(vec_len);
            result.set_len(vec_len);

            let x_step = 2.0 / w as f32;
            let mut x = S::setzero_ps();
            for i in (0..S::VF32_WIDTH).rev() {
                x[i] = -1.0 + (x_step * i as f32);
            }
            let x_step = S::set1_ps(x_step * S::VF32_WIDTH as f32);
            let y_step = 2.0 / h as f32;
            let mut y = -1.0;
            let mut i = 0;
            for _ in 0..h {
                for _ in 0..w / S::VF32_WIDTH {
                    let color = (self.c.eval_2d(x, S::set1_ps(y)) + S::set1_ps(1.0)) * S::set1_ps(128.0);
                    for j in 0..S::VF32_WIDTH {
                        result[i+j*4] = (color[j] as i32 % 255) as u8;
                        result[i+1+j*4] = (color[j] as i32 % 255) as u8;
                        result[i+2+j*4] = (color[j] as i32 % 255) as u8;
                        result[i+3+j*4] = 255 as u8;
                    }
                    x = x + x_step;
                    i += S::VF32_WIDTH*4;
                }
                y += y_step;
            }
            result
        }
    }
}

#[derive(VariantCount)]
enum APTNode<S: Simd> {    
    Add(Vec<APTNode<S>>),
    Sub(Vec<APTNode<S>>),
    Mul(Vec<APTNode<S>>),
    Div(Vec<APTNode<S>>),
    Constant(S::Vf32),
    X,
    Y,
    Empty,
}

impl<S:Simd> Clone for APTNode<S> {
    fn clone(&self) -> Self {
        match self {
            APTNode::Add(children) => APTNode::Add(children.clone()),
            APTNode::Sub(children) => APTNode::Sub(children.clone()),
            APTNode::Mul(children) => APTNode::Mul(children.clone()),
            APTNode::Div(children) => APTNode::Div(children.clone()),
            APTNode::Constant(v) => APTNode::Constant(*v),
            APTNode::X => APTNode::X,
            APTNode::Y => APTNode::Y,
            APTNode::Empty => panic!("tried to eval an empty node"),
        }
    }
}

impl<S: Simd> APTNode<S> {
    fn get_random_node(rng:&mut ThreadRng) -> APTNode<S> {        
        let r = rng.gen_range(0, APTNode::<S>::VARIANT_COUNT - 4);
        match r {
            0 => APTNode::Add(vec![APTNode::Empty, APTNode::Empty]),
            1 => APTNode::Sub(vec![APTNode::Empty, APTNode::Empty]),
            2 => APTNode::Mul(vec![APTNode::Empty, APTNode::Empty]),
            3 => APTNode::Div(vec![APTNode::Empty, APTNode::Empty]),
            _ => panic!("get_random_node generated unhandled r:{}", r),
        }
    }

    fn get_random_leaf(rng:&mut ThreadRng) -> APTNode<S> {
        unsafe {            
            let count = APTNode::<S>::VARIANT_COUNT;
            let r = rng.gen_range(0, 3);
            match r {
                0 => APTNode::X,
                1 => APTNode::Y,
                2 => APTNode::Constant(S::set1_ps(rng.gen_range(-1.0, 1.0))),
                _ => panic!("get_random_leaf generated unhandled r:{}", r),
            }
        }
    }

    fn add_random(&mut self,node: APTNode<S>,rng:&mut ThreadRng) {
        let children = match self.get_children() {
            Some(children) => children,
            None => panic!("tried to add_random to a leaf"),
        };
        let add_index = rng.gen_range(0,children.len());
        match children[add_index] {
            APTNode::Empty => children[add_index] = node,
            _ =>  children[add_index].add_random(node,rng),
        }        
    }

    /* public bool AddLeaf(AptNode leafToAdd)
    {
        if (children == null) return false;
        for (int i = 0; i < children.Length; i++)
        {
            if (children[i] == null || children[i].IsEmpty())
            {
                children[i] = leafToAdd;
                leafToAdd.parent = this;
                return true;
            }
            else if (!children[i].IsLeaf() && children[i].AddLeaf(leafToAdd))
            {
                return true;
            }
        }
        return false;
    }*/

    fn add_leaf(&mut self, leaf: &APTNode<S>) -> bool {
        match self.get_children() {
            None => false,
            Some(children) => {
                for i in 0 .. children.len() {
                    match children[i] {
                        APTNode::Empty => {                    
                            children[i] = leaf.clone();
                            return true;   
                        }
                        _ => if !children[i].is_leaf() && children[i].add_leaf(leaf) {
                                return true;
                            }
                    }                                         
                }
                return false;
            }
        }
    }

    fn generate_tree(count:usize,rng:&mut ThreadRng) -> APTNode<S> {
        let mut first = APTNode::get_random_node(rng);
        for _ in 1..count {
            first.add_random(APTNode::get_random_node(rng),rng);
        }
        while first.add_leaf(&APTNode::get_random_leaf(rng)) {};
        first
    }

    fn get_children(&mut self) -> Option<&mut Vec<APTNode<S>>> {
        match self {
            APTNode::Add(children) => Some(children),
            APTNode::Sub(children) => Some(children),
            APTNode::Mul(children) => Some(children),
            APTNode::Div(children) => Some(children),
            _ => None,
        }
    }

    fn is_leaf(&self) -> bool {
        match self {
            APTNode::X |
            APTNode::Y |
            APTNode::Constant(_) |
            APTNode::Empty => true,
            _ => false          
        }
    }

    fn eval_2d(&self, x: S::Vf32, y: S::Vf32) -> S::Vf32 {
        unsafe {
            match self {
                APTNode::Add(children) => children[0].eval_2d(x, y) + children[1].eval_2d(x, y),
                APTNode::Sub(children) => children[0].eval_2d(x, y) - children[1].eval_2d(x, y),
                APTNode::Mul(children) => children[0].eval_2d(x, y) * children[1].eval_2d(x, y),
                APTNode::Div(children) => children[0].eval_2d(x, y) / children[1].eval_2d(x, y),
                APTNode::Constant(v) => *v,
                APTNode::X => x,
                APTNode::Y => y,
                APTNode::Empty => panic!("tried to eval an empty node"),
            }
        }
    }
}
