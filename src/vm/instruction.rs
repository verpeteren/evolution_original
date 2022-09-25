use simdeez::Simd;

/*
pub const SIMPLEX_MULTIPLIER: f32 = 7.35;
pub const SIMPLEX_OFFSET: f32 = 0.028;
pub const CELL1_MULTUPLIER: f32 = 1.661291;
pub const CELL1_OFFSET: f32 = 1.0;
*/

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
    Width,
    Height,
    PI,
    E,
    X,
    Y,
    T,
}
