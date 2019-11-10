use crate::apt::*;
use simdeez::*;

pub enum Instruction<S: Simd> {
    Add,
    Sub,
    Mul,
    Div,
    FBM,
    Constant(S::Vf32),
    X,
    Y,
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
            APTNode::Constant(v) => Instruction::Constant(unsafe { S::set1_ps(*v) }),
            APTNode::X => Instruction::X,
            APTNode::Y => Instruction::Y,
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

    pub fn execute(&self,stack:&mut Vec<S::Vf32>, x: S::Vf32, y: S::Vf32) -> S::Vf32 {
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
                        sp -= 1;
                        let freq = S::set1_ps(3.05);
                        let lacunarity = S::set1_ps(0.5);
                        let gain = S::set1_ps(2.0);
                        let octaves = 4;
                        stack[sp - 1] = simdnoise::simplex::fbm_2d::<S>(
                            stack[sp] * freq,
                            stack[sp - 1] * freq,
                            lacunarity,
                            gain,
                            octaves,
                            3,
                        );
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
                }
            }
            stack[sp - 1]
        }
    }

    pub fn execute_no_bounds(&self,stack:&mut Vec<S::Vf32>, x: S::Vf32, y: S::Vf32) -> S::Vf32 {
        unsafe {
            let mut sp = 0;
            for ins in &self.instructions {
                match ins {
                    Instruction::Add => {
                        sp -= 1;
                        *stack.get_unchecked_mut(sp - 1) =
                            *stack.get_unchecked(sp) + *stack.get_unchecked(sp - 1);
                    }
                    Instruction::Sub => {
                        sp -= 1;
                        *stack.get_unchecked_mut(sp - 1) =
                            *stack.get_unchecked(sp) - *stack.get_unchecked(sp - 1);
                    }
                    Instruction::Mul => {
                        sp -= 1;
                        *stack.get_unchecked_mut(sp - 1) =
                            *stack.get_unchecked(sp) * *stack.get_unchecked(sp - 1);
                    }
                    Instruction::Div => {
                        sp -= 1;
                        *stack.get_unchecked_mut(sp - 1) =
                            *stack.get_unchecked(sp) / *stack.get_unchecked(sp - 1);
                    }
                    Instruction::FBM => {
                        sp -= 1;
                        let freq = S::set1_ps(3.05);
                        let lacunarity = S::set1_ps(0.5);
                        let gain = S::set1_ps(2.0);
                        let octaves = 4;
                        *stack.get_unchecked_mut(sp - 1) = simdnoise::simplex::fbm_2d::<S>(
                            *stack.get_unchecked(sp) * freq,
                            *stack.get_unchecked(sp - 1) * freq,
                            lacunarity,
                            gain,
                            octaves,
                            3,
                        );
                    }
                    Instruction::Constant(v) => {
                        *stack.get_unchecked_mut(sp) = *v;
                        sp += 1;
                    }
                    Instruction::X => {
                        *stack.get_unchecked_mut(sp) = x;
                        sp += 1;
                    }
                    Instruction::Y => {
                        *stack.get_unchecked_mut(sp) = y;
                        sp += 1;
                    }
                }
            }
            *stack.get_unchecked(sp - 1)
        }
    }
}
