use crate::parser::*;
use rand::prelude::*;
use simdnoise::*;
use std::sync::mpsc::*;
use variant_count::*;
use APTNode::*;

#[derive(VariantCount, Clone)]
pub enum APTNode {
    Add(Vec<APTNode>),
    Sub(Vec<APTNode>),
    Mul(Vec<APTNode>),
    Div(Vec<APTNode>),
    FBM(Vec<APTNode>),
    Ridge(Vec<APTNode>),
    Turbulence(Vec<APTNode>),
    Cell1(Vec<APTNode>),
    Cell2(Vec<APTNode>),
    Sqrt(Vec<APTNode>),
    Sin(Vec<APTNode>),
    Atan(Vec<APTNode>),
    Atan2(Vec<APTNode>),
    Tan(Vec<APTNode>),
    Log(Vec<APTNode>),
    Abs(Vec<APTNode>),
    Floor(Vec<APTNode>),
    Ceil(Vec<APTNode>),
    Clamp(Vec<APTNode>),
    Wrap(Vec<APTNode>),
    Square(Vec<APTNode>),
    Max(Vec<APTNode>),
    Min(Vec<APTNode>),
    Mod(Vec<APTNode>),
    Picture(String, Vec<APTNode>),
    Constant(f32),
    X,
    Y,
    T,
    Empty,
}

impl APTNode {
    pub fn to_lisp(&self) -> String {
        match self {
            Add(children) => format!("( + {} {} )", children[0].to_lisp(), children[1].to_lisp()),
            Sub(children) => format!("( - {} {} )", children[0].to_lisp(), children[1].to_lisp()),
            Mul(children) => format!("( * {} {} )", children[0].to_lisp(), children[1].to_lisp()),
            Div(children) => format!("( / {} {} )", children[0].to_lisp(), children[1].to_lisp()),
            FBM(children) => format!(
                "( FBM {} {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp(),
                children[2].to_lisp()
            ),
            Ridge(children) => format!(
                "( Ridge {} {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp(),
                children[2].to_lisp()
            ),
            Cell1(children) => format!(
                "( Cell1 {} {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp(),
                children[2].to_lisp()
            ),
            Cell2(children) => format!(
                "( Cell2 {} {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp(),
                children[2].to_lisp()
            ),
            Turbulence(children) => format!(
                "( Turbulence {} {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp(),
                children[2].to_lisp()
            ),
            Sqrt(children) => format!("( Sqrt {} )", children[0].to_lisp()),
            Sin(children) => format!("( Sin {} )", children[0].to_lisp()),
            Atan(children) => format!("( Atan {} )", children[0].to_lisp()),
            Atan2(children) => format!(
                "( Atan2 {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp()
            ),
            Tan(children) => format!("( Tan {} )", children[0].to_lisp()),
            Log(children) => format!("( Log {} )", children[0].to_lisp()),
            Abs(children) => format!("( Abs {} )", children[0].to_lisp()),
            Floor(children) => format!("( Floor {} )", children[0].to_lisp()),
            Ceil(children) => format!("( Ceil {} )", children[0].to_lisp()),
            Clamp(children) => format!("( Clamp  {} )", children[0].to_lisp()),
            Wrap(children) => format!("( Wrap {} )", children[0].to_lisp()),
            Square(children) => format!("( Square {} )", children[0].to_lisp()),
            Max(children) => format!("( Max {} {})", children[0].to_lisp(), children[1].to_lisp()),
            Min(children) => format!("( Min {} {})", children[0].to_lisp(), children[1].to_lisp()),
            Mod(children) => format!("( Mod {} {})", children[0].to_lisp(), children[1].to_lisp()),
            Picture(name, children) => format!(
                "( Pic-{} {} {} )",
                name,
                children[0].to_lisp(),
                children[1].to_lisp()
            ),
            Constant(v) => format!("{}", v),
            X => format!("X"),
            Y => format!("Y"),
            T => format!("T"),
            Empty => format!("EMPTY"),
        }
    }

    pub fn str_to_node(s: &str) -> Result<APTNode, String> {
        let lower = &s.to_lowercase()[..];
        match lower {
            "+" => Ok(Add(vec![Empty, Empty])),
            "-" => Ok(Sub(vec![Empty, Empty])),
            "*" => Ok(Mul(vec![Empty, Empty])),
            "/" => Ok(Div(vec![Empty, Empty])),
            "fbm" => Ok(FBM(vec![Empty, Empty, Empty])),
            "ridge" => Ok(Ridge(vec![Empty, Empty, Empty])),
            "turbulence" => Ok(Turbulence(vec![Empty, Empty, Empty])),
            "cell1" => Ok(Cell1(vec![Empty, Empty, Empty])),
            "cell2" => Ok(Cell2(vec![Empty, Empty, Empty])),
            "sqrt" => Ok(Sqrt(vec![Empty])),
            "sin" => Ok(Sin(vec![Empty])),
            "atan" => Ok(Atan(vec![Empty])),
            "atan2" => Ok(Atan2(vec![Empty, Empty])),
            "tan" => Ok(Tan(vec![Empty])),
            "log" => Ok(Log(vec![Empty])),
            "abs" => Ok(Abs(vec![Empty])),
            "floor" => Ok(Floor(vec![Empty])),
            "ceil" => Ok(Ceil(vec![Empty])),
            "clamp" => Ok(Clamp(vec![Empty])),
            "wrap" => Ok(Wrap(vec![Empty])),
            "square" => Ok(Square(vec![Empty])),
            "max" => Ok(Max(vec![Empty, Empty])),
            "min" => Ok(Min(vec![Empty, Empty])),
            "mod" => Ok(Mod(vec![Empty, Empty])),
            _ if lower == "pic" => Ok(X),
            "x" => Ok(X),
            "y" => Ok(Y),
            "t" => Ok(T),
            _ => Err(format!("Unknown operation '{}' ", s.to_string())),
        }
    }

    pub fn get_random_node(rng: &mut StdRng, pic_names: &Vec<&String>) -> APTNode {
        let r = rng.gen_range(0, APTNode::VARIANT_COUNT - 5);

        match r {
            0 => Add(vec![Empty, Empty]),
            1 => Sub(vec![Empty, Empty]),
            2 => Mul(vec![Empty, Empty]),
            3 => Div(vec![Empty, Empty]),
            4 => FBM(vec![Empty, Empty, Empty]),
            5 => Ridge(vec![Empty, Empty, Empty]),
            6 => Turbulence(vec![Empty, Empty, Empty]),
            7 => Cell1(vec![Empty, Empty, Empty]),
            8 => Cell2(vec![Empty, Empty, Empty]),
            9 => Sqrt(vec![Empty]),
            10 => Sin(vec![Empty]),
            11 => Atan(vec![Empty]),
            12 => Atan2(vec![Empty, Empty]),
            13 => Tan(vec![Empty]),
            14 => Log(vec![Empty]),
            15 => Abs(vec![Empty]),
            16 => Floor(vec![Empty]),
            17 => Ceil(vec![Empty]),
            18 => Clamp(vec![Empty]),
            19 => Wrap(vec![Empty]),
            20 => Square(vec![Empty]),
            21 => Max(vec![Empty, Empty]),
            22 => Min(vec![Empty, Empty]),
            23 => Mod(vec![Empty, Empty]),
            24 => {
                let r = rng.gen_range(0, pic_names.len()) as usize;
                Picture(pic_names[r].to_string(), vec![Empty, Empty])
            }
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

    pub fn constant_eval(&self) -> f32 {
        match self {
            Add(children) => children[0].constant_eval() + children[1].constant_eval(),
            Sub(children) => children[0].constant_eval() - children[1].constant_eval(),
            Mul(children) => children[0].constant_eval() * children[1].constant_eval(),
            Div(children) => children[0].constant_eval() / children[1].constant_eval(),
            FBM(_) => 0.0,
            Ridge(_) => 0.0,
            Turbulence(_) => 0.0, // if the noise functions all have constants it isn't worth bothering maybe?
            Cell1(_) => 0.0,
            Cell2(_) => 0.0,
            Sqrt(children) => children[0].constant_eval().sqrt(),
            Sin(children) => children[0].constant_eval().sin(),
            Atan(children) => children[0].constant_eval().atan(),
            Atan2(children) => children[0]
                .constant_eval()
                .atan2(children[1].constant_eval()),
            Tan(children) => children[0].constant_eval().tan(),
            Log(children) => children[0].constant_eval().log2(),
            Abs(children) => children[0].constant_eval().abs(),
            Floor(children) => children[0].constant_eval().floor(),
            Ceil(children) => children[0].constant_eval().ceil(),
            Clamp(children) => {
                let v = children[0].constant_eval();
                if v > 1.0 {
                    1.0
                } else if v < -1.0 {
                    -1.0
                } else {
                    v
                }
            }
            Wrap(children) => {
                let v = children[0].constant_eval();
                if v >= -1.0 && v <= 1.0 {
                    v
                } else {
                    let t = (v + 1.0) / 2.0;
                    -1.0 + 2.0 * (t - t.floor())
                }
            }
            Square(children) => {
                let v = children[0].constant_eval();
                v * v
            }
            Max(children) => {
                let a = children[0].constant_eval();
                let b = children[1].constant_eval();
                if a >= b {
                    a
                } else {
                    b
                }
            }
            Min(children) => {
                let a = children[0].constant_eval();
                let b = children[1].constant_eval();
                if a <= b {
                    a
                } else {
                    b
                }
            }
            Mod(children) => {
                let a = children[0].constant_eval();
                let b = children[1].constant_eval();
                a % b
            }
            Picture(name, children) => {
                //todo
                0.0
            }
            Constant(v) => *v,
            _ => panic!("invalid node passed to constant_esval"),
        }
    }

    fn set_children(&self, children: Vec<APTNode>) -> Self {
        match self {
            Add(_) => Add(children),
            Sub(_) => Sub(children),
            Mul(_) => Mul(children),
            Div(_) => Div(children),
            FBM(_) => FBM(children),
            Ridge(_) => Ridge(children),
            Turbulence(_) => Turbulence(children),
            Cell1(_) => Cell1(children),
            Cell2(_) => Cell2(children),
            Sqrt(_) => Sqrt(children),
            Sin(_) => Sin(children),
            Atan(_) => Atan(children),
            Atan2(_) => Atan(children),
            Tan(_) => Tan(children),
            Log(_) => Log(children),
            Abs(_) => Abs(children),
            Floor(_) => Floor(children),
            Ceil(_) => Ceil(children),
            Clamp(_) => Clamp(children),
            Wrap(_) => Wrap(children),
            Square(_) => Square(children),
            Max(_) => Max(children),
            Min(_) => Min(children),
            Mod(_) => Mod(children),
            Picture(name, _) => Picture(name.to_string(), children),
            Constant(v) => Constant(*v),
            X => X,
            Y => Y,
            T => T,
            Empty => panic!("tried to eval an empty node"),
        }
    }

    pub fn constant_fold(&self) -> APTNode {
        match self {
            Constant(v) => Constant(*v),
            X => X,
            Y => Y,
            T => T,
            _ => {
                let children = self.get_children().unwrap();
                //foreach child -> constant_fold(child), if you get back all constants -> compute the new constant, and create it
                let folded_children: Vec<APTNode> =
                    children.iter().map(|child| child.constant_fold()).collect();
                if folded_children.iter().all(|child| match child {
                    APTNode::Constant(_) => true,
                    _ => false,
                }) {
                    let clone = self.set_children(folded_children);
                    APTNode::Constant(clone.constant_eval())
                } else {
                    let clone = self.set_children(folded_children);
                    clone
                }
            }
        }
    }

    pub fn generate_tree(
        count: usize,
        video: bool,
        rng: &mut StdRng,
        pic_names: &Vec<&String>,
    ) -> APTNode {
        let leaf_func = if video {
            APTNode::get_random_leaf_video
        } else {
            APTNode::get_random_leaf
        };
        let mut first = APTNode::get_random_node(rng, pic_names);
        for _ in 1..count {
            first.add_random(APTNode::get_random_node(rng, pic_names), rng);
        }
        while first.add_leaf(&leaf_func(rng)) {}
        first
    }

    pub fn get_children_mut(&mut self) -> Option<&mut Vec<APTNode>> {
        match self {
            Add(children) | Sub(children) | Mul(children) | Div(children) | FBM(children)
            | Ridge(children) | Turbulence(children) | Cell1(children) | Cell2(children)
            | Sqrt(children) | Sin(children) | Atan(children) | Atan2(children) | Tan(children)
            | Log(children) | Abs(children) | Floor(children) | Ceil(children)
            | Clamp(children) | Wrap(children) | Square(children) | Max(children)
            | Min(children) | Mod(children) => Some(children),
            Picture(_, children) => Some(children),
            _ => None,
        }
    }

    pub fn get_children(&self) -> Option<&Vec<APTNode>> {
        match self {
            Add(children) | Sub(children) | Mul(children) | Div(children) | FBM(children)
            | Ridge(children) | Turbulence(children) | Cell1(children) | Cell2(children)
            | Sqrt(children) | Sin(children) | Atan(children) | Atan2(children) | Tan(children)
            | Log(children) | Abs(children) | Floor(children) | Ceil(children)
            | Clamp(children) | Wrap(children) | Square(children) | Max(children)
            | Min(children) | Mod(children) => Some(children),
            Picture(_, children) => Some(children),
            _ => None,
        }
    }

    pub fn is_leaf(&self) -> bool {
        match self {
            APTNode::X | APTNode::Y | APTNode::T | APTNode::Constant(_) | APTNode::Empty => true,
            _ => false,
        }
    }

    pub fn parse_apt_node(receiver: &Receiver<Token>) -> Result<APTNode, String> {
        loop {
            match receiver.recv() {
                Ok(token) => {
                    match token {
                        Token::Operation(s, line_num) => {
                            let mut node = APTNode::str_to_node(s)
                                .map_err(|msg| msg + &format!(" on line {}", line_num))?;
                            match node.get_children_mut() {
                                Some(children) => {
                                    for child in children {
                                        *child = APTNode::parse_apt_node(receiver)?;
                                    }
                                    return Ok(node);
                                }
                                None => return Ok(node),
                            }
                        }
                        Token::Constant(vstr, line_num) => {
                            let v = vstr.parse::<f32>().map_err(|_| {
                                format!("Unable to parse number {} on line {}", vstr, line_num)
                            })?;
                            return Ok(APTNode::Constant(v));
                        }
                        _ => (), //parens don't matter
                    }
                }
                Err(_) => {
                    return Err("Unexpected end of file".to_string());
                }
            }
        }
    }
}
