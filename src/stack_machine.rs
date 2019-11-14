use crate::apt::*;
use simdeez::*;

const SIMPLEX_MULTIPLIER: f32 = 7.35;
const SIMPLEX_OFFSET: f32 = 0.028;

pub enum Instruction<S: Simd> {
    Add,
    Sub,
    Mul,
    Div,
    FBM,
    Ridge,
    Turbulence,
    Sqrt,
    Sin,
    Atan,
    Tan,
    Log,
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
            APTNode::Sqrt(_) => Instruction::Sqrt,
            APTNode::Sin(_) => Instruction::Sin,
            APTNode::Atan(_) => Instruction::Atan,
            APTNode::Tan(_) => Instruction::Tan,
            APTNode::Log(_) => Instruction::Log,
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


    pub fn execute(&self, stack: &mut Vec<S::Vf32>, x: S::Vf32, y: S::Vf32,t: S::Vf32) -> S::Vf32 {
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
                        stack[sp - 1] = stack[sp] / stack[sp - 1];
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
                    Instruction::Sqrt => {
                        stack[sp - 1] = S::sqrt_ps(stack[sp - 1]);
                    }
                    Instruction::Sin => {
                        stack[sp - 1] = S::fast_sin_ps(stack[sp - 1] * S::set1_ps(3.14159));
                    }
                    Instruction::Atan => {
                        stack[sp - 1] = S::fast_atan_ps(stack[sp - 1] * S::set1_ps(3.14159));
                    }
                    Instruction::Tan => {
                        stack[sp - 1] = S::fast_tan_ps(stack[sp - 1] * S::set1_ps(3.14159)) ;
                    }
                    Instruction::Log => {
                        stack[sp - 1] = S::fast_log_ps(stack[sp - 1] * S::set1_ps(3.14159)) ;
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
            let mut result = stack[sp-1];
            for i in 0 .. S::VF32_WIDTH {
                if result[i] == std::f32::INFINITY {
                    result[i] = 1.0;
                } else if result[i] == std::f32::NEG_INFINITY {
                    result[i] = -1.0;
                }
                else if result[i].is_nan() {
                    result[i] = 0.0;
                }                   
            }
            result
        }
    }
}
