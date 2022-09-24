use std::sync::mpsc::Receiver;

use crate::parser::Token;

use rand::prelude::*;
use variant_count::VariantCount;

use APTNode::*;

#[derive(VariantCount, Clone, Debug, PartialEq)]
pub enum APTNode {
    Add(Vec<APTNode>),
    Sub(Vec<APTNode>),
    Mul(Vec<APTNode>),
    Div(Vec<APTNode>),
    Mod(Vec<APTNode>),
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
    Mandelbrot(Vec<APTNode>),
    Picture(String, Vec<APTNode>),
    Constant(f32),
    Width,
    Height,
    PI,
    E,
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
            Mod(children) => format!("( % {} {} )", children[0].to_lisp(), children[1].to_lisp()),
            FBM(children) => format!(
                "( FBM {} {} {} {} {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp(),
                children[2].to_lisp(),
                children[3].to_lisp(),
                children[4].to_lisp(),
                children[5].to_lisp()
            ),
            Ridge(children) => format!(
                "( RIDGE {} {} {} {} {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp(),
                children[2].to_lisp(),
                children[3].to_lisp(),
                children[4].to_lisp(),
                children[5].to_lisp()
            ),
            Cell1(children) => format!(
                "( CELL1 {} {} {} {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp(),
                children[2].to_lisp(),
                children[3].to_lisp(),
                children[4].to_lisp()
            ),
            Cell2(children) => format!(
                "( CELL2 {} {} {} {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp(),
                children[2].to_lisp(),
                children[3].to_lisp(),
                children[4].to_lisp()
            ),
            Turbulence(children) => format!(
                "( TURBULENCE {} {} {} {} {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp(),
                children[2].to_lisp(),
                children[3].to_lisp(),
                children[4].to_lisp(),
                children[5].to_lisp()
            ),
            Sqrt(children) => format!("( SQRT {} )", children[0].to_lisp()),
            Sin(children) => format!("( SIN {} )", children[0].to_lisp()),
            Atan(children) => format!("( ATAN {} )", children[0].to_lisp()),
            Atan2(children) => format!(
                "( ATAN2 {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp()
            ),
            Tan(children) => format!("( TAN {} )", children[0].to_lisp()),
            Log(children) => format!("( LOG {} )", children[0].to_lisp()),
            Abs(children) => format!("( ABS {} )", children[0].to_lisp()),
            Floor(children) => format!("( FLOOR {} )", children[0].to_lisp()),
            Ceil(children) => format!("( CEIL {} )", children[0].to_lisp()),
            Clamp(children) => format!("( CLAMP {} )", children[0].to_lisp()),
            Wrap(children) => format!("( WRAP {} )", children[0].to_lisp()),
            Square(children) => format!("( SQUARE {} )", children[0].to_lisp()),
            Max(children) => format!(
                "( MAX {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp()
            ),
            Min(children) => format!(
                "( MIN {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp()
            ),
            Mandelbrot(children) => format!(
                "( MANDELBROT {} {} )",
                children[0].to_lisp(),
                children[1].to_lisp()
            ),
            Picture(name, children) => format!(
                "( PIC-{} {} {} )",
                name,
                children[0].to_lisp(),
                children[1].to_lisp()
            ),
            Constant(v) => {
                if v == &std::f32::consts::PI {
                    format!("PI")
                } else if v == &std::f32::consts::E {
                    format!("E")
                } else {
                    format!("{}", v)
                }
            }
            Width => format!("WIDTH"),
            Height => format!("HEIGHT"),
            PI => format!("PI"),
            E => format!("E"),
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
            "%" => Ok(Mod(vec![Empty, Empty])),
            "fbm" => Ok(FBM(vec![Empty, Empty, Empty, Empty, Empty, Empty])),
            "ridge" => Ok(Ridge(vec![Empty, Empty, Empty, Empty, Empty, Empty])),
            "turbulence" => Ok(Turbulence(vec![Empty, Empty, Empty, Empty, Empty, Empty])),
            "cell1" => Ok(Cell1(vec![Empty, Empty, Empty, Empty, Empty])),
            "cell2" => Ok(Cell2(vec![Empty, Empty, Empty, Empty, Empty])),
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
            "mandelbrot" => Ok(Mandelbrot(vec![Empty, Empty])),
            "width" => Ok(Width),
            "height" => Ok(Height),
            "pi" => Ok(PI),
            "e" => Ok(E),
            "x" => Ok(X),
            "y" => Ok(Y),
            "t" => Ok(T),
            _ => {
                if lower.starts_with("pic-") {
                    let name = lower[4..].to_owned();
                    Ok(Picture(name, vec![Empty, Empty]))
                } else {
                    Err(format!("Unknown operation '{}' ", s.to_string()))
                }
            }
        }
    }

    pub fn get_random_node(rng: &mut StdRng, pic_names: &Vec<&String>) -> APTNode {
        let r = rng.gen_range(0..APTNode::VARIANT_COUNT - 9);

        match r {
            0 => Add(vec![Empty, Empty]),
            1 => Sub(vec![Empty, Empty]),
            2 => Mul(vec![Empty, Empty]),
            3 => Div(vec![Empty, Empty]),
            4 => Mod(vec![Empty, Empty]),
            5 => FBM(vec![Empty, Empty, Empty, Empty, Empty, Empty]),
            6 => Ridge(vec![Empty, Empty, Empty, Empty, Empty, Empty]),
            7 => Turbulence(vec![Empty, Empty, Empty, Empty, Empty, Empty]),
            8 => Cell1(vec![Empty, Empty, Empty, Empty, Empty]),
            9 => Cell2(vec![Empty, Empty, Empty, Empty, Empty]),
            10 => Sqrt(vec![Empty]),
            11 => Sin(vec![Empty]),
            12 => Atan(vec![Empty]),
            13 => Atan2(vec![Empty, Empty]),
            14 => Tan(vec![Empty]),
            15 => Log(vec![Empty]),
            16 => Abs(vec![Empty]),
            17 => Floor(vec![Empty]),
            18 => Ceil(vec![Empty]),
            19 => Clamp(vec![Empty]),
            20 => Wrap(vec![Empty]),
            21 => Square(vec![Empty]),
            22 => Max(vec![Empty, Empty]),
            23 => Min(vec![Empty, Empty]),
            24 => Mandelbrot(vec![Empty, Empty]),
            25 => {
                let r = rng.gen_range(0..pic_names.len()) as usize;
                Picture(pic_names[r].to_string(), vec![Empty, Empty])
            }
            _ => panic!("get_random_node generated unhandled r:{}", r),
        }
    }

    pub fn get_random_leaf(rng: &mut StdRng) -> APTNode {
        let r = rng.gen_range(0..3);
        match r {
            0 => APTNode::X,
            1 => APTNode::Y,
            2 => APTNode::Constant(rng.gen_range(-1.0..1.0)),
            _ => panic!("get_random_leaf generated unhandled r:{}", r),
        }
    }

    pub fn get_random_leaf_video(rng: &mut StdRng) -> APTNode {
        let r = rng.gen_range(0..4);
        match r {
            0 => APTNode::X,
            1 => APTNode::Y,
            2 => APTNode::T,
            3 => APTNode::Constant(rng.gen_range(-1.0..1.0)),
            _ => panic!("get_random_leaf generated unhandled r:{}", r),
        }
    }

    pub fn add_random(&mut self, node: APTNode, rng: &mut StdRng) {
        let children = match self.get_children_mut() {
            Some(children) => children,
            None => panic!("tried to add_random to a leaf"),
        };
        let add_index = rng.gen_range(0..children.len());
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

    fn constant_eval(&self) -> f32 {
        match self {
            Add(children) => children[0].constant_eval() + children[1].constant_eval(),
            Sub(children) => children[0].constant_eval() - children[1].constant_eval(),
            Mul(children) => children[0].constant_eval() * children[1].constant_eval(),
            Div(children) => children[0].constant_eval() / children[1].constant_eval(),
            Mod(children) => {
                let a = children[0].constant_eval();
                let b = children[1].constant_eval();
                a % b
            }
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
            Mandelbrot(_children) => {
                //todo
                0.0
            }
            Picture(_name, _children) => {
                //todo
                0.0
            }
            PI => std::f32::consts::PI,
            E => std::f32::consts::E,
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
            Mod(_) => Mod(children),
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
            Mandelbrot(_) => Mandelbrot(children),
            Picture(name, _) => Picture(name.to_string(), children[1..].to_vec()),
            Constant(v) => Constant(*v),
            Width => Width,
            Height => Height,
            PI => PI,
            E => E,
            X => X,
            Y => Y,
            T => T,
            Empty => panic!("tried to eval an empty node"),
        }
    }

    fn constant_fold(&self) -> APTNode {
        match self {
            Constant(v) => Constant(*v),
            Width => Width,
            Height => Height,
            PI => PI,
            E => E,
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
            Add(children) | Sub(children) | Mul(children) | Div(children) | Mod(children)
            | FBM(children) | Ridge(children) | Turbulence(children) | Cell1(children)
            | Cell2(children) | Sqrt(children) | Sin(children) | Atan(children)
            | Atan2(children) | Tan(children) | Log(children) | Abs(children) | Floor(children)
            | Ceil(children) | Clamp(children) | Wrap(children) | Square(children)
            | Max(children) | Min(children) | Mandelbrot(children) => Some(children),
            Picture(_, children) => Some(children),
            _ => None,
        }
    }

    pub fn get_children(&self) -> Option<&Vec<APTNode>> {
        match self {
            Add(children) | Sub(children) | Mul(children) | Div(children) | Mod(children)
            | FBM(children) | Ridge(children) | Turbulence(children) | Cell1(children)
            | Cell2(children) | Sqrt(children) | Sin(children) | Atan(children)
            | Atan2(children) | Tan(children) | Log(children) | Abs(children) | Floor(children)
            | Ceil(children) | Clamp(children) | Wrap(children) | Square(children)
            | Max(children) | Min(children) | Mandelbrot(children) => Some(children),
            Picture(_, children) => Some(children),
            _ => None,
        }
    }

    pub fn is_leaf(&self) -> bool {
        match self {
            APTNode::Width
            | APTNode::Height
            | APTNode::PI
            | APTNode::E
            | APTNode::X
            | APTNode::Y
            | APTNode::T
            | APTNode::Constant(_)
            | APTNode::Empty => true,
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

#[cfg(test)]
pub mod mock {
    use super::*;

    fn mock_params(count: usize, filled: bool) -> Vec<APTNode> {
        let mut params = Vec::with_capacity(count);
        if filled {
            let base = 1.1;
            let mut val = 1.0;
            for _i in 0..count {
                params.push(APTNode::Constant(val));
                val += base;
            }
        } else {
            for _i in 0..count {
                params.push(APTNode::Empty);
            }
        }

        params
    }
    pub fn mock_params_add(filled: bool) -> Vec<APTNode> {
        mock_params(2, filled)
    }
    pub fn mock_params_sub(filled: bool) -> Vec<APTNode> {
        mock_params(2, filled)
    }
    pub fn mock_params_mul(filled: bool) -> Vec<APTNode> {
        mock_params(2, filled)
    }
    pub fn mock_params_div(filled: bool) -> Vec<APTNode> {
        mock_params(2, filled)
    }
    pub fn mock_params_fbm(filled: bool) -> Vec<APTNode> {
        mock_params(6, filled)
    }
    pub fn mock_params_ridge(filled: bool) -> Vec<APTNode> {
        mock_params(6, filled)
    }
    pub fn mock_params_turbulence(filled: bool) -> Vec<APTNode> {
        mock_params(6, filled)
    }
    pub fn mock_params_cell1(filled: bool) -> Vec<APTNode> {
        mock_params(5, filled)
    }
    pub fn mock_params_cell2(filled: bool) -> Vec<APTNode> {
        mock_params(5, filled)
    }
    pub fn mock_params_sqrt(filled: bool) -> Vec<APTNode> {
        mock_params(1, filled)
    }
    pub fn mock_params_sin(filled: bool) -> Vec<APTNode> {
        mock_params(1, filled)
    }
    pub fn mock_params_atan(filled: bool) -> Vec<APTNode> {
        mock_params(1, filled)
    }
    pub fn mock_params_atan2(filled: bool) -> Vec<APTNode> {
        mock_params(2, filled)
    }
    pub fn mock_params_tan(filled: bool) -> Vec<APTNode> {
        mock_params(1, filled)
    }
    pub fn mock_params_log(filled: bool) -> Vec<APTNode> {
        mock_params(1, filled)
    }
    pub fn mock_params_abs(filled: bool) -> Vec<APTNode> {
        mock_params(1, filled)
    }
    pub fn mock_params_floor(filled: bool) -> Vec<APTNode> {
        mock_params(1, filled)
    }
    pub fn mock_params_ceil(filled: bool) -> Vec<APTNode> {
        mock_params(1, filled)
    }
    pub fn mock_params_clamp(filled: bool) -> Vec<APTNode> {
        mock_params(1, filled)
    }
    pub fn mock_params_wrap(filled: bool) -> Vec<APTNode> {
        mock_params(1, filled)
    }
    pub fn mock_params_square(filled: bool) -> Vec<APTNode> {
        mock_params(1, filled)
    }
    pub fn mock_params_max(filled: bool) -> Vec<APTNode> {
        mock_params(2, filled)
    }
    pub fn mock_params_min(filled: bool) -> Vec<APTNode> {
        mock_params(2, filled)
    }
    pub fn mock_params_mod(filled: bool) -> Vec<APTNode> {
        mock_params(2, filled)
    }
    pub fn mock_params_mandelbrot(filled: bool) -> Vec<APTNode> {
        mock_params(2, filled)
    }
    pub fn mock_params_picture(filled: bool) -> Vec<APTNode> {
        mock_params(2, filled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;

    #[test]
    fn test_aptnode_to_lisp() {
        assert_eq!(
            APTNode::Add(mock::mock_params_add(true)).to_lisp(),
            "( + 1 2.1 )"
        );
        assert_eq!(
            APTNode::Sub(mock::mock_params_sub(true)).to_lisp(),
            "( - 1 2.1 )"
        );
        assert_eq!(
            APTNode::Mul(mock::mock_params_mul(true)).to_lisp(),
            "( * 1 2.1 )"
        );
        assert_eq!(
            APTNode::Div(mock::mock_params_div(true)).to_lisp(),
            "( / 1 2.1 )"
        );
        assert_eq!(
            APTNode::Mod(mock::mock_params_mod(true)).to_lisp(),
            "( % 1 2.1 )"
        );
        assert_eq!(
            APTNode::FBM(mock::mock_params_fbm(true)).to_lisp(),
            "( FBM 1 2.1 3.1999998 4.2999997 5.3999996 6.4999995 )"
        );
        assert_eq!(
            APTNode::Ridge(mock::mock_params_ridge(true)).to_lisp(),
            "( RIDGE 1 2.1 3.1999998 4.2999997 5.3999996 6.4999995 )"
        );
        assert_eq!(
            APTNode::Cell1(mock::mock_params_cell1(true)).to_lisp(),
            "( CELL1 1 2.1 3.1999998 4.2999997 5.3999996 )"
        );
        assert_eq!(
            APTNode::Cell2(mock::mock_params_cell2(true)).to_lisp(),
            "( CELL2 1 2.1 3.1999998 4.2999997 5.3999996 )"
        );
        assert_eq!(
            APTNode::Turbulence(mock::mock_params_turbulence(true)).to_lisp(),
            "( TURBULENCE 1 2.1 3.1999998 4.2999997 5.3999996 6.4999995 )"
        );
        assert_eq!(
            APTNode::Sqrt(mock::mock_params_sqrt(true)).to_lisp(),
            "( SQRT 1 )"
        );
        assert_eq!(
            APTNode::Sin(mock::mock_params_sin(true)).to_lisp(),
            "( SIN 1 )"
        );
        assert_eq!(
            APTNode::Atan(mock::mock_params_atan(true)).to_lisp(),
            "( ATAN 1 )"
        );
        assert_eq!(
            APTNode::Atan2(mock::mock_params_atan2(true)).to_lisp(),
            "( ATAN2 1 2.1 )"
        );
        assert_eq!(
            APTNode::Tan(mock::mock_params_tan(true)).to_lisp(),
            "( TAN 1 )"
        );
        assert_eq!(
            APTNode::Log(mock::mock_params_log(true)).to_lisp(),
            "( LOG 1 )"
        );
        assert_eq!(
            APTNode::Abs(mock::mock_params_abs(true)).to_lisp(),
            "( ABS 1 )"
        );
        assert_eq!(
            APTNode::Abs(vec![APTNode::Constant(-10000.5)]).to_lisp(),
            "( ABS -10000.5 )"
        );
        assert_eq!(
            APTNode::Floor(mock::mock_params_floor(true)).to_lisp(),
            "( FLOOR 1 )"
        );
        assert_eq!(
            APTNode::Floor(vec![APTNode::Constant(10000.5)]).to_lisp(),
            "( FLOOR 10000.5 )"
        );
        assert_eq!(
            APTNode::Ceil(mock::mock_params_ceil(true)).to_lisp(),
            "( CEIL 1 )"
        );
        assert_eq!(
            APTNode::Ceil(vec![APTNode::Constant(10000.5)]).to_lisp(),
            "( CEIL 10000.5 )"
        );
        assert_eq!(
            APTNode::Clamp(mock::mock_params_clamp(true)).to_lisp(),
            "( CLAMP 1 )"
        );
        assert_eq!(
            APTNode::Wrap(mock::mock_params_wrap(true)).to_lisp(),
            "( WRAP 1 )"
        );
        assert_eq!(
            APTNode::Square(mock::mock_params_square(true)).to_lisp(),
            "( SQUARE 1 )"
        );
        assert_eq!(
            APTNode::Max(mock::mock_params_max(true)).to_lisp(),
            "( MAX 1 2.1 )"
        );
        assert_eq!(
            APTNode::Min(mock::mock_params_min(true)).to_lisp(),
            "( MIN 1 2.1 )"
        );
        assert_eq!(
            APTNode::Mandelbrot(mock::mock_params_mandelbrot(true)).to_lisp(),
            "( MANDELBROT 1 2.1 )"
        );
        assert_eq!(
            APTNode::Picture(
                "eye.jpg".to_string(),
                vec![APTNode::Constant(800.0), APTNode::Constant(600.0)]
            )
            .to_lisp(),
            "( PIC-eye.jpg 800 600 )".to_string()
        );
        assert_eq!(
            APTNode::Picture("eye.jpg".to_string(), mock::mock_params_picture(true)).to_lisp(),
            "( PIC-eye.jpg 1 2.1 )".to_string()
        );
        assert_eq!(APTNode::Constant(123.456).to_lisp(), "123.456");
        assert_eq!(APTNode::Constant(0.0).to_lisp(), "0");
        assert_eq!(APTNode::Constant(1.0).to_lisp(), "1");
        assert_eq!(APTNode::Constant(std::f32::consts::PI).to_lisp(), "PI");
        assert_eq!(APTNode::PI.to_lisp(), "PI");
        assert_eq!(APTNode::Constant(std::f32::consts::E).to_lisp(), "E");
        assert_eq!(APTNode::E.to_lisp(), "E");
        assert_eq!(APTNode::Width.to_lisp(), "WIDTH");
        assert_eq!(APTNode::Height.to_lisp(), "HEIGHT");
        assert_eq!(APTNode::X.to_lisp(), "X");
        assert_eq!(APTNode::Y.to_lisp(), "Y");
        assert_eq!(APTNode::T.to_lisp(), "T");
        assert_eq!(APTNode::Empty.to_lisp(), "EMPTY");
    }

    #[test]
    fn test_aptnode_str_to_node() {
        assert_eq!(
            APTNode::str_to_node("+"),
            Ok(APTNode::Add(mock::mock_params_add(false)))
        );
        assert_eq!(APTNode::str_to_node("-"), Ok(Sub(vec![Empty, Empty])));
        assert_eq!(APTNode::str_to_node("*"), Ok(Mul(vec![Empty, Empty])));
        assert_eq!(APTNode::str_to_node("/"), Ok(Div(vec![Empty, Empty])));
        assert_eq!(
            APTNode::str_to_node("%"),
            Ok(Mod(mock::mock_params_mod(false)))
        );
        assert_eq!(
            APTNode::str_to_node("fbm"),
            Ok(FBM(vec![Empty, Empty, Empty, Empty, Empty, Empty]))
        );
        assert_eq!(
            APTNode::str_to_node("ridge"),
            Ok(Ridge(vec![Empty, Empty, Empty, Empty, Empty, Empty]))
        );
        assert_eq!(
            APTNode::str_to_node("turbulence"),
            Ok(Turbulence(mock::mock_params_turbulence(false)))
        );
        assert_eq!(
            APTNode::str_to_node("cell1"),
            Ok(Cell1(mock::mock_params_cell1(false)))
        );
        assert_eq!(
            APTNode::str_to_node("cell2"),
            Ok(Cell2(mock::mock_params_cell2(false)))
        );
        assert_eq!(
            APTNode::str_to_node("sqrt"),
            Ok(Sqrt(mock::mock_params_sqrt(false)))
        );
        assert_eq!(
            APTNode::str_to_node("sin"),
            Ok(Sin(mock::mock_params_sin(false)))
        );
        assert_eq!(
            APTNode::str_to_node("atan"),
            Ok(Atan(mock::mock_params_atan(false)))
        );
        assert_eq!(
            APTNode::str_to_node("atan2"),
            Ok(Atan2(mock::mock_params_atan2(false)))
        );
        assert_eq!(
            APTNode::str_to_node("tan"),
            Ok(Tan(mock::mock_params_tan(false)))
        );
        assert_eq!(
            APTNode::str_to_node("log"),
            Ok(Log(mock::mock_params_log(false)))
        );
        assert_eq!(
            APTNode::str_to_node("abs"),
            Ok(Abs(mock::mock_params_abs(false)))
        );
        assert_eq!(APTNode::str_to_node("floor"), Ok(Floor(vec![Empty])));
        assert_eq!(
            APTNode::str_to_node("ceil"),
            Ok(Ceil(mock::mock_params_ceil(false)))
        );
        assert_eq!(
            APTNode::str_to_node("clamp"),
            Ok(Clamp(mock::mock_params_clamp(false)))
        );
        assert_eq!(
            APTNode::str_to_node("wrap"),
            Ok(Wrap(mock::mock_params_wrap(false)))
        );
        assert_eq!(
            APTNode::str_to_node("square"),
            Ok(Square(mock::mock_params_square(false)))
        );
        assert_eq!(
            APTNode::str_to_node("max"),
            Ok(Max(mock::mock_params_max(false)))
        );
        assert_eq!(
            APTNode::str_to_node("min"),
            Ok(Min(mock::mock_params_min(false)))
        );
        assert_eq!(
            APTNode::str_to_node("mandelbrot"),
            Ok(Mandelbrot(mock::mock_params_mandelbrot(false)))
        );
        assert_eq!(
            APTNode::str_to_node("PIC-eye.jpg"),
            Ok(Picture(
                "eye.jpg".to_string(),
                mock::mock_params_picture(false)
            ))
        );
        assert_eq!(APTNode::str_to_node("Width"), Ok(Width));
        assert_eq!(APTNode::str_to_node("Height"), Ok(Height));
        assert_eq!(APTNode::str_to_node("Pi"), Ok(PI));
        assert_eq!(APTNode::str_to_node("e"), Ok(E));
        assert_eq!(APTNode::str_to_node("x"), Ok(X));
        assert_eq!(APTNode::str_to_node("y"), Ok(Y));
        assert_eq!(APTNode::str_to_node("t"), Ok(T));
        assert_eq!(
            APTNode::str_to_node("pizza 60.0 \""),
            Err("Unknown operation 'pizza 60.0 \"' ".to_string())
        );
    }

    #[test]
    fn test_aptnode_add_leaf() {
        let mut root = Add(vec![Empty, Empty]);
        assert_eq!(root.add_leaf(&APTNode::Constant(1.0)), true);
        assert_eq!(root.add_leaf(&APTNode::Constant(2.1)), true);
        assert_eq!(
            root,
            APTNode::Add(vec![APTNode::Constant(1.0), APTNode::Constant(2.1)])
        );
        assert_eq!(root.add_leaf(&APTNode::Constant(3.1)), false);
        assert_eq!(
            root,
            APTNode::Add(vec![APTNode::Constant(1.0), APTNode::Constant(2.1)])
        );
    }

    #[test]
    fn test_aptnode_constant_eval() {
        assert_eq!(
            APTNode::Add(vec![APTNode::Constant(10.0), APTNode::Constant(1.02)]).constant_eval(),
            11.02
        );
        assert_eq!(
            APTNode::Sub(vec![APTNode::Constant(10.0), APTNode::Constant(1.02)]).constant_eval(),
            8.98
        );
        assert_eq!(
            APTNode::Mul(vec![APTNode::Constant(10.0), APTNode::Constant(1.02)]).constant_eval(),
            10.2
        );
        assert_eq!(
            APTNode::Div(vec![APTNode::Constant(10.0), APTNode::Constant(1.02)]).constant_eval(),
            9.803922
        );
        // @todo: should panic
        //APTNode::Div(vec![APTNode::Constant(10.0), APTNode::Constant(0.0)]).constant_eval();
        assert_eq!(
            APTNode::FBM(vec![
                APTNode::Constant(0.0),
                APTNode::Constant(1.1),
                APTNode::Constant(2.2),
                APTNode::Constant(3.3),
                APTNode::Constant(4.4),
                APTNode::Constant(5.5)
            ])
            .constant_eval(),
            0.0
        );
        assert_eq!(
            APTNode::Ridge(mock::mock_params_ridge(true)).constant_eval(),
            0.0
        );
        assert_eq!(
            APTNode::Cell1(mock::mock_params_cell1(true)).constant_eval(),
            0.0
        );
        assert_eq!(
            APTNode::Cell2(mock::mock_params_cell2(true)).constant_eval(),
            0.0
        );
        assert_eq!(
            APTNode::Turbulence(mock::mock_params_turbulence(true)).constant_eval(),
            0.0
        );
        assert_eq!(
            APTNode::Sqrt(vec![APTNode::Constant(16.0)]).constant_eval(),
            4.0
        );
        assert_eq!(
            APTNode::Sin(mock::mock_params_sin(true)).constant_eval(),
            0.84147096
        );
        assert_eq!(
            APTNode::Atan(mock::mock_params_atan(true)).constant_eval(),
            0.7853982
        );
        assert_eq!(
            APTNode::Atan2(mock::mock_params_atan2(true)).constant_eval(),
            0.44441923
        );
        assert_eq!(
            APTNode::Tan(mock::mock_params_tan(true)).constant_eval(),
            1.5574077
        );
        assert_eq!(
            APTNode::Log(mock::mock_params_log(true)).constant_eval(),
            0.0
        );
        assert_eq!(
            APTNode::Log(vec![APTNode::Constant(10000.5)]).constant_eval(),
            13.287785
        );
        assert_eq!(
            APTNode::Abs(mock::mock_params_abs(true)).constant_eval(),
            1.0
        );
        assert_eq!(
            APTNode::Abs(vec![APTNode::Constant(10000.5)]).constant_eval(),
            10000.5
        );
        assert_eq!(
            APTNode::Floor(mock::mock_params_floor(true)).constant_eval(),
            1.0
        );
        assert_eq!(
            APTNode::Floor(vec![APTNode::Constant(-10000.5)]).constant_eval(),
            -10001.0
        );
        assert_eq!(
            APTNode::Floor(vec![APTNode::Constant(10000.5)]).constant_eval(),
            10000.0
        );
        assert_eq!(
            APTNode::Ceil(mock::mock_params_ceil(true)).constant_eval(),
            1.0
        );
        assert_eq!(
            APTNode::Ceil(vec![APTNode::Constant(10000.5)]).constant_eval(),
            10001.0
        );
        assert_eq!(
            APTNode::Ceil(vec![APTNode::Constant(-10000.5)]).constant_eval(),
            -10000.0
        );
        assert_eq!(
            APTNode::Clamp(mock::mock_params_clamp(true)).constant_eval(),
            1.0
        );
        assert_eq!(
            APTNode::Clamp(vec![APTNode::Constant(10000.5)]).constant_eval(),
            1.0
        );
        assert_eq!(
            APTNode::Clamp(vec![APTNode::Constant(1.0)]).constant_eval(),
            1.0
        );
        assert_eq!(
            APTNode::Clamp(vec![APTNode::Constant(0.8)]).constant_eval(),
            0.8
        );
        assert_eq!(
            APTNode::Clamp(vec![APTNode::Constant(-0.8)]).constant_eval(),
            -0.8
        );
        assert_eq!(
            APTNode::Clamp(vec![APTNode::Constant(-1.0)]).constant_eval(),
            -1.0
        );
        assert_eq!(
            APTNode::Clamp(vec![APTNode::Constant(-10000.5)]).constant_eval(),
            -1.0
        );
        assert_eq!(
            APTNode::Wrap(mock::mock_params_wrap(true)).constant_eval(),
            1.0
        );
        assert_eq!(
            APTNode::Wrap(vec![APTNode::Constant(10000.5)]).constant_eval(),
            0.5
        );
        assert_eq!(
            APTNode::Wrap(vec![APTNode::Constant(1.0)]).constant_eval(),
            1.0
        );
        assert_eq!(
            APTNode::Wrap(vec![APTNode::Constant(0.8)]).constant_eval(),
            0.8
        );
        assert_eq!(
            APTNode::Wrap(vec![APTNode::Constant(-0.8)]).constant_eval(),
            -0.8
        );
        assert_eq!(
            APTNode::Wrap(vec![APTNode::Constant(-1.0)]).constant_eval(),
            -1.0
        );
        assert_eq!(
            APTNode::Wrap(vec![APTNode::Constant(-10000.5)]).constant_eval(),
            -0.5
        );
        assert_eq!(
            APTNode::Square(mock::mock_params_square(true)).constant_eval(),
            1.0
        );
        assert_eq!(
            APTNode::Square(vec![APTNode::Constant(4.5)]).constant_eval(),
            20.25
        );
        assert_eq!(
            APTNode::Max(mock::mock_params_max(true)).constant_eval(),
            2.1
        );
        assert_eq!(
            APTNode::Max(vec![APTNode::Constant(1.0), APTNode::Constant(-2.1)]).constant_eval(),
            1.0
        );
        assert_eq!(
            APTNode::Min(mock::mock_params_min(true)).constant_eval(),
            1.0
        );
        assert_eq!(
            APTNode::Min(vec![APTNode::Constant(1.0), APTNode::Constant(-2.1)]).constant_eval(),
            -2.1
        );
        assert_eq!(
            APTNode::Mod(mock::mock_params_mod(true)).constant_eval(),
            1.0
        );
        assert_eq!(
            APTNode::Mod(vec![APTNode::Constant(2.1), APTNode::Constant(1.0)]).constant_eval(),
            0.099999905
        );
        assert_eq!(
            APTNode::Mandelbrot(mock::mock_params_mandelbrot(true)).constant_eval(),
            0.0
        );
        assert_eq!(
            APTNode::Picture("eye.jpg".to_string(), mock::mock_params_picture(true))
                .constant_eval(),
            0.0
        );
        assert_eq!(APTNode::PI.constant_eval(), std::f32::consts::PI);
        assert_eq!(APTNode::E.constant_eval(), std::f32::consts::E);
        assert_eq!(APTNode::Constant(123.456).constant_eval(), 123.456);
        assert_eq!(APTNode::Constant(0.0).constant_eval(), 0.0);
        assert_eq!(APTNode::Constant(1.0).constant_eval(), 1.0);
    }

    #[should_panic(expected = "invalid node passed to constant_esval")]
    #[test]
    fn test_aptnode_constant_eval_width() {
        APTNode::Width.constant_eval();
    }

    #[should_panic(expected = "invalid node passed to constant_esval")]
    #[test]
    fn test_aptnode_constant_eval_height() {
        APTNode::Height.constant_eval();
    }

    #[should_panic(expected = "invalid node passed to constant_esval")]
    #[test]
    fn test_aptnode_constant_eval_x() {
        APTNode::X.constant_eval();
    }

    #[should_panic(expected = "invalid node passed to constant_esval")]
    #[test]
    fn test_aptnode_constant_eval_y() {
        APTNode::Y.constant_eval();
    }

    #[should_panic(expected = "invalid node passed to constant_esval")]
    #[test]
    fn test_aptnode_constant_eval_t() {
        APTNode::T.constant_eval();
    }

    #[should_panic(expected = "invalid node passed to constant_esval")]
    #[test]
    fn test_aptnode_constant_eval_eval() {
        APTNode::Empty.constant_eval();
    }

    #[test]
    fn test_aptnode_constant_set_children() {
        //@todo: check that the vector lengths are correct for each of the enums
        assert_eq!(
            APTNode::str_to_node("abs")
                .unwrap()
                .set_children(mock::mock_params_abs(true)),
            APTNode::Abs(mock::mock_params_abs(true))
        );

        assert_eq!(
            APTNode::str_to_node("+")
                .unwrap()
                .set_children(vec![APTNode::Constant(1.0), APTNode::Constant(2.1)]),
            APTNode::Add(vec![APTNode::Constant(1.0), APTNode::Constant(2.1)])
        );

        assert_eq!(
            APTNode::str_to_node("atan")
                .unwrap()
                .set_children(mock::mock_params_atan(true)),
            APTNode::Atan(mock::mock_params_atan(true))
        );

        if false {
            assert_eq!(
                APTNode::str_to_node("atan2")
                    .unwrap()
                    .set_children(mock::mock_params_atan2(true)),
                APTNode::Atan2(mock::mock_params_atan2(true))
            );
        }

        assert_eq!(
            APTNode::str_to_node("ceil")
                .unwrap()
                .set_children(mock::mock_params_ceil(true)),
            APTNode::Ceil(mock::mock_params_ceil(true))
        );

        assert_eq!(
            APTNode::str_to_node("cell1")
                .unwrap()
                .set_children(mock::mock_params_cell1(true)),
            APTNode::Cell1(mock::mock_params_cell1(true))
        );

        assert_eq!(
            APTNode::str_to_node("cell2")
                .unwrap()
                .set_children(mock::mock_params_cell2(true)),
            APTNode::Cell2(mock::mock_params_cell2(true))
        );

        assert_eq!(
            APTNode::str_to_node("clamp")
                .unwrap()
                .set_children(mock::mock_params_clamp(true)),
            APTNode::Clamp(mock::mock_params_clamp(true))
        );

        if false {
            assert_eq!(
                APTNode::Constant(0.0).set_children(vec![APTNode::Constant(123.456)]),
                APTNode::Constant(123.456)
            );
        }

        assert_eq!(
            APTNode::str_to_node("/")
                .unwrap()
                .set_children(vec![APTNode::Constant(0.0), APTNode::Constant(9992.1111)]),
            APTNode::Div(vec![APTNode::Constant(0.0), APTNode::Constant(9992.1111)])
        );

        assert_eq!(
            APTNode::str_to_node("fbm").unwrap().set_children(vec![
                APTNode::Constant(0.0),
                APTNode::Constant(1.1),
                APTNode::Constant(2.2),
                APTNode::Constant(3.3),
                APTNode::Constant(4.4),
                APTNode::Constant(5.5)
            ]),
            APTNode::FBM(vec![
                APTNode::Constant(0.0),
                APTNode::Constant(1.1),
                APTNode::Constant(2.2),
                APTNode::Constant(3.3),
                APTNode::Constant(4.4),
                APTNode::Constant(5.5)
            ])
        );

        assert_eq!(
            APTNode::str_to_node("floor")
                .unwrap()
                .set_children(mock::mock_params_floor(true)),
            APTNode::Floor(mock::mock_params_floor(true))
        );

        assert_eq!(
            APTNode::str_to_node("log")
                .unwrap()
                .set_children(mock::mock_params_log(true)),
            APTNode::Log(mock::mock_params_log(true))
        );

        assert_eq!(
            APTNode::Mandelbrot(vec![APTNode::Constant(1.0), APTNode::Constant(2.1)])
                .set_children(mock::mock_params_mandelbrot(true)),
            APTNode::Mandelbrot(mock::mock_params_mandelbrot(true))
        );

        assert_eq!(
            APTNode::str_to_node("max")
                .unwrap()
                .set_children(mock::mock_params_max(true)),
            APTNode::Max(mock::mock_params_max(true))
        );

        assert_eq!(
            APTNode::str_to_node("min")
                .unwrap()
                .set_children(mock::mock_params_min(true)),
            APTNode::Min(mock::mock_params_min(true))
        );

        assert_eq!(
            APTNode::str_to_node("*")
                .unwrap()
                .set_children(vec![APTNode::Constant(1.0), APTNode::Constant(2.0)]),
            APTNode::Mul(vec![APTNode::Constant(1.0), APTNode::Constant(2.0)])
        );

        assert_eq!(
            APTNode::str_to_node("%")
                .unwrap()
                .set_children(mock::mock_params_mod(true)),
            APTNode::Mod(mock::mock_params_mod(true))
        );

        assert_eq!(
            APTNode::str_to_node("pic-eye.jpg")
                .unwrap()
                .set_children(vec![
                    APTNode::Picture("cow.jpg".to_string(), mock::mock_params_picture(true)),
                    APTNode::Constant(333.0),
                    APTNode::Constant(444.0)
                ]),
            APTNode::Picture(
                "eye.jpg".to_string(),
                vec![APTNode::Constant(333.0), APTNode::Constant(444.0)]
            )
        );

        assert_eq!(
            APTNode::str_to_node("ridge")
                .unwrap()
                .set_children(mock::mock_params_ridge(true)),
            APTNode::Ridge(mock::mock_params_ridge(true))
        );

        assert_eq!(
            APTNode::str_to_node("sin")
                .unwrap()
                .set_children(mock::mock_params_sin(true)),
            APTNode::Sin(mock::mock_params_sin(true))
        );

        assert_eq!(
            APTNode::str_to_node("sqrt")
                .unwrap()
                .set_children(mock::mock_params_sqrt(true)),
            APTNode::Sqrt(mock::mock_params_sqrt(true)),
        );

        assert_eq!(
            APTNode::str_to_node("square")
                .unwrap()
                .set_children(mock::mock_params_square(true)),
            APTNode::Square(mock::mock_params_square(true))
        );

        assert_eq!(
            APTNode::str_to_node("-")
                .unwrap()
                .set_children(vec![APTNode::Constant(1.1), APTNode::Constant(2.0)]),
            APTNode::Sub(vec![APTNode::Constant(1.1), APTNode::Constant(2.0)])
        );

        assert_eq!(
            APTNode::str_to_node("tan")
                .unwrap()
                .set_children(mock::mock_params_tan(true)),
            APTNode::Tan(mock::mock_params_tan(true))
        );

        assert_eq!(
            APTNode::str_to_node("turbulence")
                .unwrap()
                .set_children(mock::mock_params_turbulence(true)),
            APTNode::Turbulence(mock::mock_params_turbulence(true))
        );

        assert_eq!(
            APTNode::str_to_node("wrap")
                .unwrap()
                .set_children(mock::mock_params_wrap(true)),
            APTNode::Wrap(mock::mock_params_wrap(true))
        );

        assert_eq!(
            APTNode::Width.set_children(vec![APTNode::Empty]),
            APTNode::Width
        );
        assert_eq!(
            APTNode::Height.set_children(vec![APTNode::Empty]),
            APTNode::Height
        );
        assert_eq!(APTNode::PI.set_children(vec![APTNode::Empty]), APTNode::PI);
        assert_eq!(APTNode::E.set_children(vec![APTNode::Empty]), APTNode::E);

        assert_eq!(APTNode::X.set_children(vec![APTNode::Empty]), APTNode::X);

        assert_eq!(APTNode::Y.set_children(vec![APTNode::Empty]), APTNode::Y);

        assert_eq!(APTNode::T.set_children(vec![APTNode::Empty]), APTNode::T);
    }

    #[should_panic(expected = "tried to eval an empty node")]
    #[test]
    fn test_aptnode_constant_set_children_empty() {
        APTNode::Empty.set_children(vec![APTNode::Constant(10.0)]);
    }

    #[test]
    fn test_aptnode_constant_fold() {
        assert_eq!(APTNode::Width.constant_fold(), APTNode::Width);
        assert_eq!(APTNode::Height.constant_fold(), APTNode::Height);
        assert_eq!(APTNode::PI.constant_fold(), APTNode::PI);
        assert_eq!(APTNode::E.constant_fold(), APTNode::E);
        assert_eq!(APTNode::X.constant_fold(), APTNode::X);
        assert_eq!(APTNode::Y.constant_fold(), APTNode::Y);
        assert_eq!(APTNode::T.constant_fold(), APTNode::T);
        assert_eq!(
            APTNode::Add(vec![Constant(1.0), APTNode::Constant(2.0)]).constant_fold(),
            APTNode::Constant(3.0)
        );
        assert_eq!(
            APTNode::Add(vec![
                Constant(1.0),
                APTNode::Mul(vec![Constant(6.0), APTNode::Constant(0.5)])
            ])
            .constant_fold(),
            APTNode::Constant(4.0)
        );
    }

    #[test]
    fn test_aptnode_get_children_mut() {
        assert_eq!(
            APTNode::Add(vec![APTNode::Constant(1.0), APTNode::Constant(2.1)])
                .get_children_mut()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Sub(vec![APTNode::Constant(1.1), APTNode::Constant(2.0)])
                .get_children_mut()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Mul(vec![APTNode::Constant(1.0), APTNode::Constant(2.0)])
                .get_children_mut()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Div(vec![APTNode::Constant(0.0), APTNode::Constant(9992.1111)])
                .get_children_mut()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::FBM(vec![
                APTNode::Constant(0.0),
                APTNode::Constant(1.1),
                APTNode::Constant(2.2)
            ])
            .get_children_mut()
            .unwrap()
            .len(),
            3
        );
        assert_eq!(
            APTNode::Ridge(mock::mock_params_ridge(true))
                .get_children_mut()
                .unwrap()
                .len(),
            6
        );
        assert_eq!(
            APTNode::Cell1(mock::mock_params_cell1(true))
                .get_children_mut()
                .unwrap()
                .len(),
            5
        );
        assert_eq!(
            APTNode::Cell2(mock::mock_params_cell2(true))
                .get_children_mut()
                .unwrap()
                .len(),
            5
        );
        assert_eq!(
            APTNode::Turbulence(mock::mock_params_turbulence(true))
                .get_children_mut()
                .unwrap()
                .len(),
            6
        );
        assert_eq!(
            APTNode::Sqrt(mock::mock_params_sqrt(true))
                .get_children_mut()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Sin(mock::mock_params_sin(true))
                .get_children_mut()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Atan(mock::mock_params_atan(true))
                .get_children_mut()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Atan2(mock::mock_params_atan2(true))
                .get_children_mut()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Tan(mock::mock_params_tan(true))
                .get_children_mut()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Log(mock::mock_params_log(true))
                .get_children_mut()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Abs(mock::mock_params_abs(true))
                .get_children_mut()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Floor(mock::mock_params_floor(true))
                .get_children_mut()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Ceil(mock::mock_params_ceil(true))
                .get_children_mut()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Clamp(mock::mock_params_clamp(true))
                .get_children_mut()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Wrap(mock::mock_params_wrap(true))
                .get_children_mut()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Square(mock::mock_params_square(true))
                .get_children_mut()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Max(mock::mock_params_max(true))
                .get_children_mut()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Min(mock::mock_params_min(true))
                .get_children_mut()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Mod(mock::mock_params_mod(true))
                .get_children_mut()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Mandelbrot(mock::mock_params_mandelbrot(true))
                .get_children_mut()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Picture("eye.jpg".to_string(), mock::mock_params_picture(true))
                .get_children_mut()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(APTNode::Constant(1.2).get_children_mut(), None);
        assert_eq!(APTNode::Width.get_children_mut(), None);
        assert_eq!(APTNode::Height.get_children_mut(), None);
        assert_eq!(APTNode::PI.get_children_mut(), None);
        assert_eq!(APTNode::E.get_children_mut(), None);
        assert_eq!(APTNode::X.get_children_mut(), None);
        assert_eq!(APTNode::Y.get_children_mut(), None);
        assert_eq!(APTNode::T.get_children_mut(), None);
        assert_eq!(APTNode::Empty.get_children(), None);
    }

    #[test]
    fn test_aptnode_get_children() {
        assert_eq!(
            APTNode::Add(vec![APTNode::Constant(1.0), APTNode::Constant(2.1)])
                .get_children()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Sub(vec![APTNode::Constant(1.1), APTNode::Constant(2.0)])
                .get_children()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Mul(vec![APTNode::Constant(1.0), APTNode::Constant(2.0)])
                .get_children()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Div(vec![APTNode::Constant(0.0), APTNode::Constant(9992.1111)])
                .get_children()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::FBM(vec![
                APTNode::Constant(0.0),
                APTNode::Constant(1.1),
                APTNode::Constant(2.2)
            ])
            .get_children()
            .unwrap()
            .len(),
            3
        );
        assert_eq!(
            APTNode::Ridge(mock::mock_params_ridge(true))
                .get_children()
                .unwrap()
                .len(),
            6
        );
        assert_eq!(
            APTNode::Cell1(mock::mock_params_cell1(true))
                .get_children()
                .unwrap()
                .len(),
            5
        );
        assert_eq!(
            APTNode::Cell2(mock::mock_params_cell2(true))
                .get_children()
                .unwrap()
                .len(),
            5
        );
        assert_eq!(
            APTNode::Turbulence(mock::mock_params_turbulence(true))
                .get_children()
                .unwrap()
                .len(),
            6
        );
        assert_eq!(
            APTNode::Sqrt(mock::mock_params_sqrt(true))
                .get_children()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Sin(mock::mock_params_sin(true))
                .get_children()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Atan(mock::mock_params_atan(true))
                .get_children()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Atan2(mock::mock_params_atan2(true))
                .get_children()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Tan(mock::mock_params_tan(true))
                .get_children()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Log(mock::mock_params_log(true))
                .get_children()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Abs(mock::mock_params_abs(true))
                .get_children()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Floor(mock::mock_params_floor(true))
                .get_children()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Ceil(mock::mock_params_ceil(true))
                .get_children()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Clamp(mock::mock_params_clamp(true))
                .get_children()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Wrap(mock::mock_params_wrap(true))
                .get_children()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Square(mock::mock_params_square(true))
                .get_children()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            APTNode::Max(mock::mock_params_max(true))
                .get_children()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Min(mock::mock_params_min(true))
                .get_children()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Mod(mock::mock_params_mod(true))
                .get_children()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Mandelbrot(mock::mock_params_mandelbrot(true))
                .get_children()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(
            APTNode::Picture("eye.jpg".to_string(), mock::mock_params_picture(true))
                .get_children()
                .unwrap()
                .len(),
            2
        );
        assert_eq!(APTNode::Constant(1.2).get_children(), None);
        assert_eq!(APTNode::Width.get_children(), None);
        assert_eq!(APTNode::Height.get_children(), None);
        assert_eq!(APTNode::PI.get_children(), None);
        assert_eq!(APTNode::E.get_children(), None);
        assert_eq!(APTNode::X.get_children(), None);
        assert_eq!(APTNode::Y.get_children(), None);
        assert_eq!(APTNode::T.get_children(), None);
        assert_eq!(APTNode::Empty.get_children(), None);
    }

    #[test]
    fn test_aptnode_aptnode_is_leaf() {
        assert_eq!(
            APTNode::Add(vec![APTNode::Constant(1.0), APTNode::Constant(2.1)]).is_leaf(),
            false
        );
        assert_eq!(
            APTNode::Sub(vec![APTNode::Constant(1.1), APTNode::Constant(2.0)]).is_leaf(),
            false
        );
        assert_eq!(
            APTNode::Mul(vec![APTNode::Constant(1.0), APTNode::Constant(2.0)]).is_leaf(),
            false
        );
        assert_eq!(
            APTNode::Div(vec![APTNode::Constant(0.0), APTNode::Constant(9992.1111)]).is_leaf(),
            false
        );
        assert_eq!(
            APTNode::FBM(vec![
                APTNode::Constant(0.0),
                APTNode::Constant(1.1),
                APTNode::Constant(2.2)
            ])
            .is_leaf(),
            false
        );
        assert_eq!(
            APTNode::Ridge(mock::mock_params_ridge(true)).is_leaf(),
            false
        );
        assert_eq!(
            APTNode::Cell1(mock::mock_params_cell1(true)).is_leaf(),
            false
        );
        assert_eq!(
            APTNode::Cell2(mock::mock_params_cell2(true)).is_leaf(),
            false
        );
        assert_eq!(
            APTNode::Turbulence(mock::mock_params_turbulence(true)).is_leaf(),
            false
        );
        assert_eq!(APTNode::Sqrt(mock::mock_params_sqrt(true)).is_leaf(), false);
        assert_eq!(APTNode::Sin(mock::mock_params_sin(true)).is_leaf(), false);
        assert_eq!(APTNode::Atan(mock::mock_params_atan(true)).is_leaf(), false);
        assert_eq!(
            APTNode::Atan2(mock::mock_params_atan2(true)).is_leaf(),
            false
        );
        assert_eq!(APTNode::Tan(mock::mock_params_tan(true)).is_leaf(), false);
        assert_eq!(APTNode::Log(mock::mock_params_log(true)).is_leaf(), false);
        assert_eq!(APTNode::Abs(mock::mock_params_abs(true)).is_leaf(), false);
        assert_eq!(
            APTNode::Floor(mock::mock_params_floor(true)).is_leaf(),
            false
        );
        assert_eq!(APTNode::Ceil(mock::mock_params_ceil(true)).is_leaf(), false);
        assert_eq!(
            APTNode::Clamp(mock::mock_params_clamp(true)).is_leaf(),
            false
        );
        assert_eq!(APTNode::Wrap(mock::mock_params_wrap(true)).is_leaf(), false);
        assert_eq!(
            APTNode::Square(mock::mock_params_square(true)).is_leaf(),
            false
        );
        assert_eq!(APTNode::Max(mock::mock_params_max(true)).is_leaf(), false);
        assert_eq!(APTNode::Min(mock::mock_params_min(true)).is_leaf(), false);
        assert_eq!(APTNode::Mod(mock::mock_params_mod(true)).is_leaf(), false);
        assert_eq!(
            APTNode::Mandelbrot(mock::mock_params_mandelbrot(true)).is_leaf(),
            false
        );
        assert_eq!(
            APTNode::Picture("eye.jpg".to_string(), mock::mock_params_picture(true)).is_leaf(),
            false
        );
        assert_eq!(APTNode::Constant(1.2).is_leaf(), true);
        assert_eq!(APTNode::X.is_leaf(), true);
        assert_eq!(APTNode::Width.is_leaf(), true);
        assert_eq!(APTNode::Height.is_leaf(), true);
        assert_eq!(APTNode::PI.is_leaf(), true);
        assert_eq!(APTNode::E.is_leaf(), true);
        assert_eq!(APTNode::Y.is_leaf(), true);
        assert_eq!(APTNode::T.is_leaf(), true);
        assert_eq!(APTNode::Empty.is_leaf(), true);
    }

    #[test]
    fn test_aptnode_get_random_node() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let name = "eye.jpg".to_string();

        let pic_names = vec![&name];
        for _i in 0..100 {
            match APTNode::get_random_node(&mut rng, &pic_names) {
                Constant(_) | X | Y | T | Empty => {
                    panic!("This APTNode was not expected");
                }
                _ => {}
            }
        }
    }

    #[test]
    fn test_aptnode_get_random_leaf() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        for _i in 0..100 {
            match APTNode::get_random_leaf(&mut rng) {
                Constant(value) => {
                    assert!(value >= -1.0 && value <= 1.0);
                }
                X | Y | Empty => {}
                _ => {
                    panic!("This APTNode was not expected");
                }
            }
        }
    }

    #[test]
    fn test_aptnode_get_random_leaf_video() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        for _i in 0..100 {
            match APTNode::get_random_leaf(&mut rng) {
                Constant(value) => {
                    assert!(value >= -1.0 && value <= 1.0);
                }
                X | Y | T | Empty => {}
                _ => {
                    panic!("This APTNode was not expected");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "tried to add_random to a leaf")]
    fn test_aptnode_add_random_fail() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let mut this_node = APTNode::X;
        let that_node = APTNode::Y;
        this_node.add_random(that_node, &mut rng);
    }

    #[test]
    #[ignore] // findout what is wrong here
    fn test_aptnode_add_random() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let mut this_node = APTNode::Add(vec![APTNode::Constant(1.0), APTNode::Constant(2.1)]);
        let that_node = APTNode::Add(vec![APTNode::Constant(1.0), APTNode::Constant(2.1)]);
        this_node.add_random(that_node.clone(), &mut rng);
        let kids = this_node.get_children().unwrap();
        assert!(kids.get(0).unwrap() == &that_node || kids.get(1).unwrap() == &that_node);
    }
}
