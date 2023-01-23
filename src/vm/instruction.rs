use simdeez::Simd;

use std::fmt;

#[derive(PartialEq)]
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

impl<S> fmt::Debug for Instruction<S>
where
    S: Simd,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> Result<(), fmt::Error> {
        let name = match self {
            Instruction::Add => "Add".to_string(),
            Instruction::Sub => "Sub".to_string(),
            Instruction::Mul => "Mul".to_string(),
            Instruction::Div => "Div".to_string(),
            Instruction::Mod => "Mod".to_string(),
            Instruction::FBM => "FBM".to_string(),
            Instruction::Ridge => "Ridge".to_string(),
            Instruction::Turbulence => "Turbulence".to_string(),
            Instruction::Cell1 => "Cell1".to_string(),
            Instruction::Cell2 => "Cell2".to_string(),
            Instruction::Sqrt => "Sqrt".to_string(),
            Instruction::Sin => "Sin".to_string(),
            Instruction::Atan => "Atan".to_string(),
            Instruction::Atan2 => "Atan2".to_string(),
            Instruction::Tan => "Tan".to_string(),
            Instruction::Log => "Log".to_string(),
            Instruction::Abs => "Abs".to_string(),
            Instruction::Floor => "Floor".to_string(),
            Instruction::Ceil => "Ceil".to_string(),
            Instruction::Clamp => "Clamp".to_string(),
            Instruction::Wrap => "Wrap".to_string(),
            Instruction::Square => "Square".to_string(),
            Instruction::Max => "Max".to_string(),
            Instruction::Min => "Min".to_string(),
            Instruction::Mandelbrot => "Mandelbrot".to_string(),
            Instruction::Picture(pic_name) => format!("Picture({})", pic_name),
            Instruction::Constant(vf32) => format!("Constant({:?}", vf32),
            Instruction::Width => "Width".to_string(),
            Instruction::Height => "Height".to_string(),
            Instruction::PI => "PI".to_string(),
            Instruction::E => "E".to_string(),
            Instruction::X => "X".to_string(),
            Instruction::Y => "Y".to_string(),
            Instruction::T => "T".to_string(),
        };

        write!(f, "{}", name)
    }
}

/* impl PartialEq for Instruction {
    fn eq(&self, other: &Self) -> bool {
        self.isbn == other.isbn
    }
}
*/

#[cfg(test)]
mod test {
    use super::*;
    use simdeez::avx2::Avx2;

    #[test]
    fn test_debug() {
        assert_eq!(&format!("{:?}", Instruction::Add::<Avx2>), "Add");
        assert_eq!(&format!("{:?}", Instruction::Sub::<Avx2>), "Sub");
        assert_eq!(&format!("{:?}", Instruction::Mul::<Avx2>), "Mul");
        assert_eq!(&format!("{:?}", Instruction::Div::<Avx2>), "Div");
        assert_eq!(&format!("{:?}", Instruction::Mod::<Avx2>), "Mod");
        assert_eq!(&format!("{:?}", Instruction::FBM::<Avx2>), "FBM");
        assert_eq!(&format!("{:?}", Instruction::Ridge::<Avx2>), "Ridge");
        assert_eq!(
            &format!("{:?}", Instruction::Turbulence::<Avx2>),
            "Turbulence"
        );
        assert_eq!(&format!("{:?}", Instruction::Cell1::<Avx2>), "Cell1");
        assert_eq!(&format!("{:?}", Instruction::Cell2::<Avx2>), "Cell2");
        assert_eq!(&format!("{:?}", Instruction::Sqrt::<Avx2>), "Sqrt");
        assert_eq!(&format!("{:?}", Instruction::Sin::<Avx2>), "Sin");
        assert_eq!(&format!("{:?}", Instruction::Atan::<Avx2>), "Atan");
        assert_eq!(&format!("{:?}", Instruction::Atan2::<Avx2>), "Atan2");
        assert_eq!(&format!("{:?}", Instruction::Tan::<Avx2>), "Tan");
        assert_eq!(&format!("{:?}", Instruction::Log::<Avx2>), "Log");
        assert_eq!(&format!("{:?}", Instruction::Abs::<Avx2>), "Abs");
        assert_eq!(&format!("{:?}", Instruction::Floor::<Avx2>), "Floor");
        assert_eq!(&format!("{:?}", Instruction::Ceil::<Avx2>), "Ceil");
        assert_eq!(&format!("{:?}", Instruction::Clamp::<Avx2>), "Clamp");
        assert_eq!(&format!("{:?}", Instruction::Wrap::<Avx2>), "Wrap");
        assert_eq!(&format!("{:?}", Instruction::Square::<Avx2>), "Square");
        assert_eq!(&format!("{:?}", Instruction::Max::<Avx2>), "Max");
        assert_eq!(&format!("{:?}", Instruction::Min::<Avx2>), "Min");
        assert_eq!(
            &format!("{:?}", Instruction::Mandelbrot::<Avx2>),
            "Mandelbrot"
        );
        assert_eq!(
            &format!("{:?}", Instruction::Picture::<Avx2>("cat.png".to_string())),
            "Picture(cat.png)"
        );
        /*
        assert_eq!(
            &format!("{:?}", Instruction::Constant::<Avx2>(0.03)),
            "Constant(0.03)"
        );
        */
        assert_eq!(&format!("{:?}", Instruction::Width::<Avx2>), "Width");
        assert_eq!(&format!("{:?}", Instruction::Height::<Avx2>), "Height");
        assert_eq!(&format!("{:?}", Instruction::PI::<Avx2>), "PI");
        assert_eq!(&format!("{:?}", Instruction::E::<Avx2>), "E");
        assert_eq!(&format!("{:?}", Instruction::X::<Avx2>), "X");
        assert_eq!(&format!("{:?}", Instruction::Y::<Avx2>), "Y");
        assert_eq!(&format!("{:?}", Instruction::T::<Avx2>), "T");
    }
}
