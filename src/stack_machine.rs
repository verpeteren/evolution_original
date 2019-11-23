use crate::apt::*;
use simdeez::*;
use simdnoise::*;
const SIMPLEX_MULTIPLIER: f32 = 7.35;
const SIMPLEX_OFFSET: f32 = 0.028;
const CELL1_MULTUPLIER: f32 = 1.661291;
const CELL1_OFFSET: f32 = 1.0;

pub enum Instruction<S: Simd> {
    Add,
    Sub,
    Mul,
    Div,
    FBM,
    Ridge,
    Turbulence,
    Cell1,
    Cell2,
    Sqrt,
    Sin,
    Atan,
    Atan2,
    Tan,
    Log,
    Abs,
    Constant(S::Vf32),
    X,
    Y,
    T,
}

pub struct StackMachine<S: Simd> {
    pub instructions: Vec<Instruction<S>>,
}

impl<S: Simd> StackMachine<S> {
    pub fn get_instruction(node: &APTNode) -> Instruction<S> {
        match node {
            APTNode::Add(_) => Instruction::Add,
            APTNode::Sub(_) => Instruction::Sub,
            APTNode::Mul(_) => Instruction::Mul,
            APTNode::Div(_) => Instruction::Div,
            APTNode::FBM(_) => Instruction::FBM,
            APTNode::Ridge(_) => Instruction::Ridge,
            APTNode::Turbulence(_) => Instruction::Turbulence,
            APTNode::Cell1(_) => Instruction::Cell1,
            APTNode::Cell2(_) => Instruction::Cell2,
            APTNode::Sqrt(_) => Instruction::Sqrt,
            APTNode::Sin(_) => Instruction::Sin,
            APTNode::Atan(_) => Instruction::Atan,
            APTNode::Atan2(_) => Instruction::Atan2,
            APTNode::Tan(_) => Instruction::Tan,
            APTNode::Log(_) => Instruction::Log,
            APTNode::Abs(_) => Instruction::Abs,
            APTNode::Constant(v) => Instruction::Constant(unsafe { S::set1_ps(*v) }),
            APTNode::X => Instruction::X,
            APTNode::Y => Instruction::Y,
            APTNode::T => Instruction::T,
            APTNode::Empty => panic!("got empty building stack machine"),
        }
    }

    fn build_helper(&mut self, node: &APTNode) {
        match node.get_children() {
            Some(children) => {
                for child in children.iter().rev() {
                    self.build_helper(child);
                }
            }
            None => (),
        }
        self.instructions.push(StackMachine::get_instruction(node));
    }

    pub fn build(node: &APTNode) -> StackMachine<S> {
        let mut sm = StackMachine {
            instructions: Vec::new(),
        };
        sm.build_helper(node);
        sm
    }

    #[inline(always)]
    pub fn deal_with_nan(mut a: S::Vf32) -> S::Vf32 {
        for i in 0..S::VF32_WIDTH {
            if a[i] == std::f32::INFINITY {
                //   println!("inf");
                a[i] = 1.0;
            } else if a[i] == std::f32::NEG_INFINITY {
                //  println!("neg inf");
                a[i] = -1.0;
            } else if a[i].is_nan() {
                //  println!("nan");
                a[i] = 0.0;
            }
        }
        a
    }

    pub fn execute(&self, stack: &mut Vec<S::Vf32>, x: S::Vf32, y: S::Vf32, t: S::Vf32) -> S::Vf32 {
        unsafe {
            let mut sp = 0;
            for ins in &self.instructions {
                match ins {
                    Instruction::Add => {
                        sp -= 1;
                        stack[sp - 1] = stack[sp] + stack[sp - 1];
                    }
                    Instruction::Sub => {
                        sp -= 1;
                        stack[sp - 1] = stack[sp] - stack[sp - 1];
                    }
                    Instruction::Mul => {
                        sp -= 1;
                        stack[sp - 1] = stack[sp] * stack[sp - 1];
                    }
                    Instruction::Div => {
                        sp -= 1;
                        stack[sp - 1] = StackMachine::<S>::deal_with_nan(stack[sp] / stack[sp - 1]);
                    }
                    Instruction::FBM => {
                        sp -= 2;
                        let freq = stack[sp - 1] * S::set1_ps(25.0);
                        let lacunarity = S::set1_ps(0.5);
                        let gain = S::set1_ps(2.0);
                        let octaves = 3;
                        stack[sp - 1] = simdnoise::simplex::fbm_2d::<S>(
                            stack[sp + 1] * freq,
                            stack[sp] * freq,
                            lacunarity,
                            gain,
                            octaves,
                            3,
                        ) * S::set1_ps(SIMPLEX_MULTIPLIER)
                            - S::set1_ps(SIMPLEX_OFFSET); //todo clamp between -1 and 1??
                    }
                    Instruction::Ridge => {
                        sp -= 2;
                        let freq = stack[sp - 1] * S::set1_ps(25.0);
                        let lacunarity = S::set1_ps(0.5);
                        let gain = S::set1_ps(2.0);
                        let octaves = 3;
                        stack[sp - 1] = simdnoise::simplex::ridge_2d::<S>(
                            stack[sp + 1] * freq,
                            stack[sp] * freq,
                            lacunarity,
                            gain,
                            octaves,
                            3,
                        ) * S::set1_ps(SIMPLEX_OFFSET)
                            - S::set1_ps(SIMPLEX_OFFSET); //todo clamp between -1 and 1??
                    }
                    Instruction::Turbulence => {
                        sp -= 2;
                        let freq = stack[sp - 1] * S::set1_ps(25.0);
                        let lacunarity = S::set1_ps(0.5);
                        let gain = S::set1_ps(2.0);
                        let octaves = 3;
                        stack[sp - 1] = simdnoise::simplex::turbulence_2d::<S>(
                            stack[sp + 1] * freq,
                            stack[sp] * freq,
                            lacunarity,
                            gain,
                            octaves,
                            3,
                        ) * S::set1_ps(SIMPLEX_MULTIPLIER)
                            - S::set1_ps(SIMPLEX_OFFSET); //todo clamp between -1 and 1?? \
                    }
                    Instruction::Cell1 => {
                        sp -= 2;
                        let freq = stack[sp - 1] * S::set1_ps(4.0);
                        stack[sp - 1] = simdnoise::cellular::cellular_2d::<S>(
                            stack[sp + 1] * freq,
                            stack[sp] * freq,
                            CellDistanceFunction::Euclidean,
                            CellReturnType::Distance,
                            S::set1_ps(0.45),
                            1,
                        ) * S::set1_ps(CELL1_MULTUPLIER)
                            - S::set1_ps(CELL1_OFFSET); //todo clamp between -1 and 1?? \
                    }
                    Instruction::Cell2 => {
                        sp -= 2;
                        let freq = stack[sp - 1] * S::set1_ps(4.0);
                        stack[sp - 1] = simdnoise::cellular::cellular_2d::<S>(
                            stack[sp + 1] * freq,
                            stack[sp] * freq,
                            CellDistanceFunction::Euclidean,
                            CellReturnType::CellValue,
                            S::set1_ps(0.45),
                            1,
                        );
                    }
                    Instruction::Sqrt => {
                        let v = stack[sp - 1];
                        let positive = S::sqrt_ps(v);
                        let negative = S::mul_ps(S::set1_ps(-1.0), S::sqrt_ps(S::abs_ps(v)));
                        let mask = S::cmpge_ps(v, S::setzero_ps());
                        stack[sp - 1] = S::blendv_ps(negative, positive, mask);
                    }
                    Instruction::Sin => {
                        stack[sp - 1] = S::fast_sin_ps(stack[sp - 1] * S::set1_ps(3.14159));
                    }
                    Instruction::Atan => {
                        stack[sp - 1] = S::fast_atan_ps(stack[sp - 1] * S::set1_ps(4.0))
                            * S::set1_ps(0.666666666);
                    }
                    Instruction::Atan2 => {
                        sp -= 1;
                        let x = stack[sp - 1];
                        let y = stack[sp] * S::set1_ps(4.0);
                        stack[sp - 1] = S::fast_atan2_ps(y, x) * S::set1_ps(0.318309);
                    }
                    Instruction::Tan => {
                        stack[sp - 1] = S::fast_tan_ps(stack[sp - 1] * S::set1_ps(1.57079632679));
                    }
                    Instruction::Log => {
                        let v = stack[sp - 1] * S::set1_ps(4.0);
                        let positive = S::fast_log_ps(v);
                        let negative = S::mul_ps(S::set1_ps(-1.0), S::fast_log_ps(S::abs_ps(v)));
                        let mask = S::cmpge_ps(v, S::setzero_ps());
                        stack[sp - 1] =
                            S::blendv_ps(negative, positive, mask) * S::set1_ps(0.367879);
                    }
                    Instruction::Abs => {
                        stack[sp - 1] = S::abs_ps(stack[sp - 1]);
                    }
                    Instruction::Constant(v) => {
                        stack[sp] = *v;
                        sp += 1;
                    }
                    Instruction::X => {
                        stack[sp] = x;
                        sp += 1;
                    }
                    Instruction::Y => {
                        stack[sp] = y;
                        sp += 1;
                    }
                    Instruction::T => {
                        stack[sp] = t;
                        sp += 1;
                    }
                }
            }
            stack[sp - 1]
        }
    }
}
