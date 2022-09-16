use crate::actual_picture::*;
use crate::apt::*;
use simdeez::*;
use simdnoise::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::RwLock;
pub const SIMPLEX_MULTIPLIER: f32 = 7.35;
pub const SIMPLEX_OFFSET: f32 = 0.028;
pub const CELL1_MULTUPLIER: f32 = 1.661291;
pub const CELL1_OFFSET: f32 = 1.0;
use Instruction::*;

pub enum Instruction<S: Simd> {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
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
    Floor,
    Ceil,
    Clamp,
    Wrap,
    Square,
    Max,
    Min,
    Mandelbrot,
    Picture(String),
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
            APTNode::Add(_) => Add,
            APTNode::Sub(_) => Sub,
            APTNode::Mul(_) => Mul,
            APTNode::Div(_) => Div,
            APTNode::Mod(_) => Mod,
            APTNode::FBM(_) => FBM,
            APTNode::Ridge(_) => Ridge,
            APTNode::Turbulence(_) => Turbulence,
            APTNode::Cell1(_) => Cell1,
            APTNode::Cell2(_) => Cell2,
            APTNode::Sqrt(_) => Sqrt,
            APTNode::Sin(_) => Sin,
            APTNode::Atan(_) => Atan,
            APTNode::Atan2(_) => Atan2,
            APTNode::Tan(_) => Tan,
            APTNode::Log(_) => Log,
            APTNode::Abs(_) => Abs,
            APTNode::Floor(_) => Floor,
            APTNode::Ceil(_) => Ceil,
            APTNode::Clamp(_) => Clamp,
            APTNode::Wrap(_) => Wrap,
            APTNode::Square(_) => Square,
            APTNode::Max(_) => Max,
            APTNode::Min(_) => Min,
            APTNode::Mandelbrot(_) => Mandelbrot,
            APTNode::Picture(name, _) => Picture(name.to_string()),
            APTNode::Constant(v) => Constant(unsafe { S::set1_ps(*v) }),
            APTNode::X => X,
            APTNode::Y => Y,
            APTNode::T => T,
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

    pub fn execute(
        &self,
        stack: &mut Vec<S::Vf32>,
        pics: Arc<HashMap<String, ActualPicture>>,
        x: S::Vf32,
        y: S::Vf32,
        t: S::Vf32,
    ) -> S::Vf32 {
        unsafe {
            let mut sp = 0;
            for ins in &self.instructions {
                match ins {
                    Add => {
                        sp -= 1;
                        stack[sp - 1] = stack[sp] + stack[sp - 1];
                    }
                    Sub => {
                        sp -= 1;
                        stack[sp - 1] = stack[sp] - stack[sp - 1];
                    }
                    Mul => {
                        sp -= 1;
                        stack[sp - 1] = stack[sp] * stack[sp - 1];
                    }
                    Div => {
                        sp -= 1;
                        stack[sp - 1] = StackMachine::<S>::deal_with_nan(stack[sp] / stack[sp - 1]);
                    }
                    Mod => {
                        sp -= 1;
                        let a = stack[sp - 1];
                        let b = stack[sp];
                        let mut r = S::setzero_ps();
                        for i in 0..S::VF32_WIDTH {
                            r[i] = a[i] % b[i];
                        }
                        stack[sp - 1] = r;
                    }
                    FBM => {
                        sp -= 5;
                        let xfreq = stack[sp - 1] * S::set1_ps(15.0);
                        let yfreq = stack[sp + 4] * S::set1_ps(15.0);
                        let lacunarity = stack[sp + 2] * S::set1_ps(5.0);
                        let gain = stack[sp + 3] * S::set1_ps(0.5);
                        let octaves = 3;
                        stack[sp - 1] = simdnoise::simplex::fbm_2d::<S>(
                            stack[sp + 1] * xfreq,
                            stack[sp] * yfreq,
                            lacunarity,
                            gain,
                            octaves,
                            3,
                        ); //* S::set1_ps(SIMPLEX_MULTIPLIER)
                           //- S::set1_ps(SIMPLEX_OFFSET); //todo clamp between -1 and 1??
                    }
                    Ridge => {
                        sp -= 5;
                        let xfreq = stack[sp - 1] * S::set1_ps(15.0);
                        let yfreq = stack[sp + 4] * S::set1_ps(15.0);
                        let lacunarity = stack[sp + 2] * S::set1_ps(5.0);
                        let gain = stack[sp + 3] * S::set1_ps(0.5);
                        let octaves = 3;
                        stack[sp - 1] = simdnoise::simplex::ridge_2d::<S>(
                            stack[sp + 1] * xfreq,
                            stack[sp] * yfreq,
                            lacunarity,
                            gain,
                            octaves,
                            3,
                        ); // S::set1_ps(SIMPLEX_OFFSET)
                           //- S::set1_ps(SIMPLEX_OFFSET); //todo clamp between -1 and 1??
                    }
                    Turbulence => {
                        sp -= 5;
                        let xfreq = stack[sp - 1] * S::set1_ps(15.0);
                        let yfreq = stack[sp + 4] * S::set1_ps(15.0);
                        let lacunarity = stack[sp + 2] * S::set1_ps(5.0);
                        let gain = stack[sp + 3] * S::set1_ps(0.5);
                        let octaves = 3;
                        stack[sp - 1] = simdnoise::simplex::turbulence_2d::<S>(
                            stack[sp + 1] * xfreq,
                            stack[sp] * yfreq,
                            lacunarity,
                            gain,
                            octaves,
                            3,
                        ); // * S::set1_ps(SIMPLEX_MULTIPLIER)
                           //- S::set1_ps(SIMPLEX_OFFSET); //todo clamp between -1 and 1?? \
                    }
                    Cell1 => {
                        sp -= 4;
                        let xfreq = stack[sp - 1] * S::set1_ps(4.0);
                        let yfreq = stack[sp + 3] * S::set1_ps(4.0);
                        let jitter = stack[sp + 2] * S::set1_ps(0.5);
                        stack[sp - 1] = simdnoise::cellular::cellular_2d::<S>(
                            stack[sp + 1] * xfreq,
                            stack[sp] * yfreq,
                            CellDistanceFunction::Euclidean,
                            CellReturnType::Distance,
                            jitter,
                            1,
                        ); //* S::set1_ps(CELL1_MULTUPLIER)
                           //- S::set1_ps(CELL1_OFFSET); //todo clamp between -1 and 1?? \
                    }
                    Cell2 => {
                        sp -= 4;
                        let xfreq = stack[sp - 1] * S::set1_ps(4.0);
                        let yfreq = stack[sp + 3] * S::set1_ps(4.0);
                        let jitter = stack[sp + 2] * S::set1_ps(0.5);
                        stack[sp - 1] = simdnoise::cellular::cellular_2d::<S>(
                            stack[sp + 1] * xfreq,
                            stack[sp] * yfreq,
                            CellDistanceFunction::Euclidean,
                            CellReturnType::CellValue,
                            jitter,
                            1,
                        );
                    }
                    Sqrt => {
                        let v = stack[sp - 1];
                        let positive = S::sqrt_ps(v);
                        let negative = S::mul_ps(S::set1_ps(-1.0), S::sqrt_ps(S::abs_ps(v)));
                        let mask = S::cmpge_ps(v, S::setzero_ps());
                        stack[sp - 1] = S::blendv_ps(negative, positive, mask);
                    }
                    Sin => {
                        stack[sp - 1] = S::fast_sin_ps(stack[sp - 1] * S::set1_ps(3.14159));
                    }
                    Atan => {
                        stack[sp - 1] = S::fast_atan_ps(stack[sp - 1] * S::set1_ps(4.0))
                            * S::set1_ps(0.666666666);
                    }
                    Atan2 => {
                        sp -= 1;
                        let x = stack[sp - 1];
                        let y = stack[sp] * S::set1_ps(4.0);
                        stack[sp - 1] = S::fast_atan2_ps(y, x) * S::set1_ps(0.318309);
                    }
                    Tan => {
                        stack[sp - 1] = S::fast_tan_ps(stack[sp - 1] * S::set1_ps(1.57079632679));
                    }
                    Log => {
                        let v = stack[sp - 1] * S::set1_ps(4.0);
                        let positive = S::fast_ln_ps(v);
                        let negative = S::mul_ps(S::set1_ps(-1.0), S::fast_ln_ps(S::abs_ps(v)));
                        let mask = S::cmpge_ps(v, S::setzero_ps());
                        stack[sp - 1] =
                            S::blendv_ps(negative, positive, mask) * S::set1_ps(0.367879);
                    }
                    Abs => {
                        stack[sp - 1] = S::abs_ps(stack[sp - 1]);
                    }
                    Floor => {
                        stack[sp - 1] = S::fast_floor_ps(stack[sp - 1]);
                    }
                    Ceil => {
                        stack[sp - 1] = S::fast_ceil_ps(stack[sp - 1]);
                    }
                    Clamp => {
                        let mut v = stack[sp - 1];
                        for i in 0..S::VF32_WIDTH {
                            if v[i] > 1.0 {
                                v[i] = 1.0
                            } else if v[i] < -1.0 {
                                v[i] = -1.0
                            }
                        }
                        stack[sp - 1] = v;
                    }
                    Wrap => {
                        let mut v = stack[sp - 1];
                        for i in 0..S::VF32_WIDTH {
                            if v[i] < -1.0 || v[i] > 1.0 {
                                let t = (v[i] + 1.0) / 2.0;
                                v[i] = -1.0 + 2.0 * (t - t.floor());
                            }
                        }
                        stack[sp - 1] = v;
                    }
                    Square => {
                        let v = stack[sp - 1];
                        stack[sp - 1] = v * v;
                    }
                    Max => {
                        sp -= 1;
                        stack[sp - 1] = S::max_ps(stack[sp - 1], stack[sp]);
                    }
                    Min => {
                        sp -= 1;
                        stack[sp - 1] = S::min_ps(stack[sp - 1], stack[sp]);
                    }
                    Mandelbrot => {
                        sp -= 1;
                        //todo do
                    }
                    Picture(name) => {
                        sp -= 1;

                        let y = stack[sp - 1];
                        let x = stack[sp];

                        let picture = &pics[name];
                        let w = S::set1_epi32(picture.w as i32);
                        let h = S::set1_epi32(picture.h as i32);
                        let wf = S::cvtepi32_ps(w);
                        let hf = S::cvtepi32_ps(h);
                        let mut xpct = (x + S::set1_ps(1.0)) / S::set1_ps(2.0);
                        let mut ypct = (y + S::set1_ps(1.0)) / S::set1_ps(2.0);
                        for i in 0..S::VF32_WIDTH {
                            xpct[i] = xpct[i] % 1.0;
                            ypct[i] = ypct[i] % 1.0;
                        }
                        let xi = S::cvtps_epi32(xpct * wf);
                        let yi = S::cvtps_epi32(ypct * hf);
                        let index = xi + w * yi;

                        //println!("w:{:?} h{:?} xpct:{:?} ypct:{:?} index:{},{}",w[0],h[0],xpct[0],ypct[0],index[0],index[1]);
                        for i in 0..S::VF32_WIDTH {
                            stack[sp - 1][i] = picture.brightness
                                [index[i] as usize % (picture.w as usize * picture.h as usize)];
                        }
                    }
                    Constant(v) => {
                        stack[sp] = *v;
                        sp += 1;
                    }
                    X => {
                        stack[sp] = x;
                        sp += 1;
                    }
                    Y => {
                        stack[sp] = y;
                        sp += 1;
                    }
                    T => {
                        stack[sp] = t;
                        sp += 1;
                    }
                }
            }
            stack[sp - 1]
        }
    }
}
