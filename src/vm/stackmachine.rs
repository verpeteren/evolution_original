use std::collections::HashMap;
use std::sync::Arc;

use crate::parser::aptnode::APTNode;
use crate::pic::actual_picture::ActualPicture;
use crate::vm::instruction::Instruction;

use simdeez::Simd;
use simdnoise::{
    cellular::cellular_2d,
    simplex::{fbm_2d, ridge_2d, turbulence_2d},
    CellDistanceFunction, CellReturnType,
};

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
            APTNode::Mod(_) => Instruction::Mod,
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
            APTNode::Floor(_) => Instruction::Floor,
            APTNode::Ceil(_) => Instruction::Ceil,
            APTNode::Clamp(_) => Instruction::Clamp,
            APTNode::Wrap(_) => Instruction::Wrap,
            APTNode::Square(_) => Instruction::Square,
            APTNode::Max(_) => Instruction::Max,
            APTNode::Min(_) => Instruction::Min,
            APTNode::Mandelbrot(_) => Instruction::Mandelbrot,
            APTNode::Picture(name, _) => Instruction::Picture(name.to_string()),
            APTNode::Constant(v) => Instruction::Constant(unsafe { S::set1_ps(*v) }),
            APTNode::Width => Instruction::Width,
            APTNode::Height => Instruction::Height,
            APTNode::PI => Instruction::PI,
            APTNode::E => Instruction::E,
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
                a[i] = 1.0;
            } else if a[i] == std::f32::NEG_INFINITY {
                a[i] = -1.0;
            } else if a[i].is_nan() {
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
        w: S::Vf32,
        h: S::Vf32,
    ) -> S::Vf32 {
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
                    Instruction::Mod => {
                        sp -= 1;
                        let a = stack[sp - 1];
                        let b = stack[sp];
                        let mut r = S::setzero_ps();
                        for i in 0..S::VF32_WIDTH {
                            r[i] = a[i] % b[i];
                        }
                        stack[sp - 1] = r;
                    }
                    Instruction::FBM => {
                        sp -= 5;
                        let xfreq = stack[sp - 1] * S::set1_ps(15.0);
                        let yfreq = stack[sp + 4] * S::set1_ps(15.0);
                        let lacunarity = stack[sp + 2] * S::set1_ps(5.0);
                        let gain = stack[sp + 3] * S::set1_ps(0.5);
                        let octaves = 3;
                        stack[sp - 1] = fbm_2d::<S>(
                            stack[sp + 1] * xfreq,
                            stack[sp] * yfreq,
                            lacunarity,
                            gain,
                            octaves,
                            3,
                        );
                    }
                    Instruction::Ridge => {
                        sp -= 5;
                        let xfreq = stack[sp - 1] * S::set1_ps(15.0);
                        let yfreq = stack[sp + 4] * S::set1_ps(15.0);
                        let lacunarity = stack[sp + 2] * S::set1_ps(5.0);
                        let gain = stack[sp + 3] * S::set1_ps(0.5);
                        let octaves = 3;
                        stack[sp - 1] = ridge_2d::<S>(
                            stack[sp + 1] * xfreq,
                            stack[sp] * yfreq,
                            lacunarity,
                            gain,
                            octaves,
                            3,
                        );
                    }
                    Instruction::Turbulence => {
                        sp -= 5;
                        let xfreq = stack[sp - 1] * S::set1_ps(15.0);
                        let yfreq = stack[sp + 4] * S::set1_ps(15.0);
                        let lacunarity = stack[sp + 2] * S::set1_ps(5.0);
                        let gain = stack[sp + 3] * S::set1_ps(0.5);
                        let octaves = 3;
                        stack[sp - 1] = turbulence_2d::<S>(
                            stack[sp + 1] * xfreq,
                            stack[sp] * yfreq,
                            lacunarity,
                            gain,
                            octaves,
                            3,
                        );
                    }
                    Instruction::Cell1 => {
                        sp -= 4;
                        let xfreq = stack[sp - 1] * S::set1_ps(4.0);
                        let yfreq = stack[sp + 3] * S::set1_ps(4.0);
                        let jitter = stack[sp + 2] * S::set1_ps(0.5);
                        stack[sp - 1] = cellular_2d::<S>(
                            stack[sp + 1] * xfreq,
                            stack[sp] * yfreq,
                            CellDistanceFunction::Euclidean,
                            CellReturnType::Distance,
                            jitter,
                            1,
                        );
                    }
                    Instruction::Cell2 => {
                        sp -= 4;
                        let xfreq = stack[sp - 1] * S::set1_ps(4.0);
                        let yfreq = stack[sp + 3] * S::set1_ps(4.0);
                        let jitter = stack[sp + 2] * S::set1_ps(0.5);
                        stack[sp - 1] = cellular_2d::<S>(
                            stack[sp + 1] * xfreq,
                            stack[sp] * yfreq,
                            CellDistanceFunction::Euclidean,
                            CellReturnType::CellValue,
                            jitter,
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
                        let positive = S::fast_ln_ps(v);
                        let negative = S::mul_ps(S::set1_ps(-1.0), S::fast_ln_ps(S::abs_ps(v)));
                        let mask = S::cmpge_ps(v, S::setzero_ps());
                        stack[sp - 1] =
                            S::blendv_ps(negative, positive, mask) * S::set1_ps(0.367879);
                    }
                    Instruction::Abs => {
                        stack[sp - 1] = S::abs_ps(stack[sp - 1]);
                    }
                    Instruction::Floor => {
                        stack[sp - 1] = S::fast_floor_ps(stack[sp - 1]);
                    }
                    Instruction::Ceil => {
                        stack[sp - 1] = S::fast_ceil_ps(stack[sp - 1]);
                    }
                    Instruction::Clamp => {
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
                    Instruction::Wrap => {
                        let mut v = stack[sp - 1];
                        for i in 0..S::VF32_WIDTH {
                            if v[i] < -1.0 || v[i] > 1.0 {
                                let t = (v[i] + 1.0) / 2.0;
                                v[i] = -1.0 + 2.0 * (t - t.floor());
                            }
                        }
                        stack[sp - 1] = v;
                    }
                    Instruction::Square => {
                        let v = stack[sp - 1];
                        stack[sp - 1] = v * v;
                    }
                    Instruction::Max => {
                        sp -= 1;
                        stack[sp - 1] = S::max_ps(stack[sp - 1], stack[sp]);
                    }
                    Instruction::Min => {
                        sp -= 1;
                        stack[sp - 1] = S::min_ps(stack[sp - 1], stack[sp]);
                    }
                    Instruction::Mandelbrot => {
                        sp -= 1;
                        //todo do
                    }
                    Instruction::Picture(name) => {
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

                        // println!("w:{:?} h{:?} xpct:{:?} ypct:{:?} index:{},{}",w[0],h[0],xpct[0],ypct[0],index[0],index[1]);
                        let brightness_len = picture.brightness.len();
                        for i in 0..S::VF32_WIDTH {
                            let slot: usize =
                                index[i] as usize % (picture.w as usize * picture.h as usize);
                            if slot >= brightness_len {
                                break;
                            }
                            stack[sp - 1][i] = picture.brightness[slot];
                        }
                    }
                    Instruction::Constant(v) => {
                        stack[sp] = *v;
                        sp += 1;
                    }
                    Instruction::Width => {
                        stack[sp] = w;
                        sp += 1;
                    }
                    Instruction::Height => {
                        stack[sp] = h;
                        sp += 1;
                    }
                    Instruction::PI => {
                        let v = S::set1_ps(std::f32::consts::PI);
                        stack[sp] = v;
                        sp += 1;
                    }
                    Instruction::E => {
                        let v = S::set1_ps(std::f32::consts::E);
                        stack[sp] = v;
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

#[cfg(test)]
mod tests {
    use crate::parser::aptnode::mock;

    use super::*;
    use simdeez::avx2::*;
    use simdeez::scalar::*;
    use simdeez::sse2::*;
    use simdeez::sse41::*;

    simd_runtime_generate!(
        fn impl_stackmachine_get_instruction() {
            match StackMachine::<S>::get_instruction(&APTNode::Add(mock::mock_params_add(true))) {
                Instruction::Add => {}
                _ => {
                    panic!("Unexpected result");
                }
            };
            match StackMachine::<S>::get_instruction(&APTNode::Sub(mock::mock_params_sub(true))) {
                Instruction::Sub => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Mul(mock::mock_params_mul(true))) {
                Instruction::Mul => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Div(mock::mock_params_div(true))) {
                Instruction::Div => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::FBM(mock::mock_params_fbm(true))) {
                Instruction::FBM => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Ridge(mock::mock_params_ridge(true)))
            {
                Instruction::Ridge => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Cell1(mock::mock_params_cell1(true)))
            {
                Instruction::Cell1 => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Cell2(mock::mock_params_cell2(true)))
            {
                Instruction::Cell2 => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Turbulence(
                mock::mock_params_turbulence(true),
            )) {
                Instruction::Turbulence => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Sqrt(mock::mock_params_sqrt(true))) {
                Instruction::Sqrt => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Sin(mock::mock_params_sin(true))) {
                Instruction::Sin => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Atan(mock::mock_params_atan(true))) {
                Instruction::Atan => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Atan2(mock::mock_params_atan2(true)))
            {
                Instruction::Atan2 => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Tan(mock::mock_params_tan(true))) {
                Instruction::Tan => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Log(mock::mock_params_log(true))) {
                Instruction::Log => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Abs(mock::mock_params_abs(true))) {
                Instruction::Abs => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Floor(mock::mock_params_floor(true)))
            {
                Instruction::Floor => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Ceil(mock::mock_params_ceil(true))) {
                Instruction::Ceil => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Clamp(mock::mock_params_clamp(true)))
            {
                Instruction::Clamp => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Wrap(mock::mock_params_wrap(true))) {
                Instruction::Wrap => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Square(mock::mock_params_square(
                true,
            ))) {
                Instruction::Square => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Max(mock::mock_params_max(true))) {
                Instruction::Max => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Min(mock::mock_params_min(true))) {
                Instruction::Min => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Mod(mock::mock_params_mod(true))) {
                Instruction::Mod => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Mandelbrot(
                mock::mock_params_mandelbrot(true),
            )) {
                Instruction::Mandelbrot => {}
                _ => {
                    panic!("Unexpected result");
                }
            }

            let name = "eye.jpg".to_string();
            match StackMachine::<S>::get_instruction(&APTNode::Picture(
                name.clone(),
                mock::mock_params_picture(true),
            )) {
                Instruction::Picture(got) => {
                    assert_eq!(got, name);
                }
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Constant(6.0)) {
                Instruction::Constant(_) => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Width) {
                Instruction::Width => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Height) {
                Instruction::Height => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::PI) {
                Instruction::PI => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::E) {
                Instruction::E => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::X) {
                Instruction::X => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::Y) {
                Instruction::Y => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
            match StackMachine::<S>::get_instruction(&APTNode::T) {
                Instruction::T => {}
                _ => {
                    panic!("Unexpected result");
                }
            }
        }
    );

    #[test]
    fn test_stackmachine_get_instruction() {
        impl_stackmachine_get_instruction_runtime_select();
    }

    simd_runtime_generate!(
        fn impl_stackmachine_build() {
            let sm = StackMachine::<S>::build(&APTNode::Add(mock::mock_params_sub(true)));
            assert_eq!(sm.instructions.len(), 3);
        }
    );

    #[test]
    fn test_stackmachine_build() {
        impl_stackmachine_build_runtime_select();
    }

    simd_runtime_generate!(
        fn impl_stackmachine_deal_with_nan() {
            unsafe {
                let zeros = S::set1_ps(0.0);
                let ones = S::set1_ps(1.0);
                let neg_ones = S::set1_ps(-1.0);
                let neg_infs = S::set1_ps(std::f32::NEG_INFINITY);
                let infs = S::set1_ps(std::f32::INFINITY);
                let nans = S::set1_ps(std::f32::NAN);

                assert_eq!(StackMachine::<S>::deal_with_nan(ones)[0], ones[0]);
                assert_eq!(StackMachine::<S>::deal_with_nan(zeros)[0], zeros[0]);
                assert_eq!(StackMachine::<S>::deal_with_nan(neg_infs)[0], neg_ones[0]);
                assert_eq!(StackMachine::<S>::deal_with_nan(infs)[0], ones[0]);
                assert_eq!(StackMachine::<S>::deal_with_nan(nans)[0], zeros[0]);
            }
        }
    );

    #[test]
    fn test_stackmachine_deal_with_nan() {
        impl_stackmachine_deal_with_nan_runtime_select();
    }
}
