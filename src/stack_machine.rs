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
    instructions: Vec<Instruction<S>>,    
    stack: Vec<S::Vf32>,
}

impl<S: Simd> StackMachine<S> {
    pub fn new() -> StackMachine<S> {
        StackMachine {
            instructions: Vec::new(),
            stack: Vec::new(),            
        }
    }

    pub fn get_instruction(node: &APTNode<S>) -> Instruction<S> {
        match node {
            APTNode::Add(_) => Instruction::Add,
            APTNode::Sub(_) => Instruction::Sub,
            APTNode::Mul(_) => Instruction::Mul,
            APTNode::Div(_) => Instruction::Div,
            APTNode::FBM(_) => Instruction::FBM,
            APTNode::Constant(v) => Instruction::Constant(*v),
            APTNode::X => Instruction::X,
            APTNode::Y => Instruction::Y,
            APTNode::Empty => panic!("got empty building stack machine"),
        }
    }

    fn build_helper(&mut self, node:&APTNode<S>) {
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

    pub fn build(&mut self, node: &APTNode<S>) {
        self.build_helper(node);
        self.stack = Vec::with_capacity(self.instructions.len());
        unsafe {
            self.stack.set_len(self.instructions.len());
        }  
    }

    pub fn execute(&mut self, x: S::Vf32, y: S::Vf32) -> S::Vf32 {        
        unsafe {
            let mut sp = 0;
            for ins in &self.instructions {                
                match ins {                    
                    Instruction::Add => {
                        sp -= 1;
                        self.stack[sp - 1] = self.stack[sp] + self.stack[sp - 1];                        
                    }
                    Instruction::Sub => {
                        sp -= 1;
                        self.stack[sp - 1] = self.stack[sp] - self.stack[sp - 1];                        
                    }
                    Instruction::Mul => {
                        sp -= 1;
                        self.stack[sp - 1] = self.stack[sp] * self.stack[sp - 1];                        
                    }
                    Instruction::Div => {
                        sp -= 1;
                        self.stack[sp - 1] = self.stack[sp] / self.stack[sp - 1];                        
                    }
                    Instruction::FBM => {
                        sp -= 1;
                        let freq = S::set1_ps(3.05);
                        let lacunarity = S::set1_ps(0.5);
                        let gain = S::set1_ps(2.0);
                        let octaves = 4;
                        self.stack[sp-1] = 
                            simdnoise::simplex::fbm_2d::<S>(
                                self.stack[sp] * freq,
                                self.stack[sp - 1] * freq,
                                lacunarity,
                                gain,
                                octaves,
                                3,
                            );                        
                    }
                    Instruction::Constant(v) => {
                        self.stack[sp] = *v;
                        sp += 1;
                    }
                    Instruction::X => {
                        self.stack[sp] = x;
                        sp += 1;
                    }
                    Instruction::Y => {
                        self.stack[sp] = y;
                        sp += 1;
                    }
                }
            }
            self.stack[sp-1]
        }
    }

    pub fn execute_no_bounds(&mut self, x: S::Vf32, y: S::Vf32) -> S::Vf32 {        
        unsafe {
            let mut sp = 0;
            for ins in &self.instructions {                
                match ins {                    
                    Instruction::Add => {
                        sp -= 1;
                        *self.stack.get_unchecked_mut(sp - 1) = *self.stack.get_unchecked(sp) + *self.stack.get_unchecked(sp - 1);                        
                    }
                    Instruction::Sub => {
                        sp -= 1;
                        *self.stack.get_unchecked_mut(sp - 1) = *self.stack.get_unchecked(sp) - *self.stack.get_unchecked(sp - 1);                        
                    }
                    Instruction::Mul => {
                        sp -= 1;
                        *self.stack.get_unchecked_mut(sp - 1) = *self.stack.get_unchecked(sp) * *self.stack.get_unchecked(sp - 1);                        
                    }
                    Instruction::Div => {
                        sp -= 1;
                        *self.stack.get_unchecked_mut(sp - 1) = *self.stack.get_unchecked(sp) / *self.stack.get_unchecked(sp - 1);                        
                    }
                    Instruction::FBM => {
                        sp -= 1;
                        let freq = S::set1_ps(3.05);
                        let lacunarity = S::set1_ps(0.5);
                        let gain = S::set1_ps(2.0);
                        let octaves = 4;
                        *self.stack.get_unchecked_mut(sp-1) = 
                            simdnoise::simplex::fbm_2d::<S>(
                                *self.stack.get_unchecked(sp) * freq,
                                *self.stack.get_unchecked(sp - 1) * freq,
                                lacunarity,
                                gain,
                                octaves,
                                3,
                            );                        
                    }
                    Instruction::Constant(v) => {
                        *self.stack.get_unchecked_mut(sp) = *v;
                        sp += 1;
                    }
                    Instruction::X => {
                        *self.stack.get_unchecked_mut(sp) = x;
                        sp += 1;
                    }
                    Instruction::Y => {
                        *self.stack.get_unchecked_mut(sp) = y;
                        sp += 1;
                    }
                }
            }
            *self.stack.get_unchecked(sp-1)
        }
    }
}
