pub mod gradient;
pub mod grayscale;
pub mod hsv;
pub mod mono;
pub mod rgb;

use rand::rngs::StdRng;

pub use crate::pic::pic::Pic;
pub use gradient::GradientData;
pub use grayscale::GrayscaleData;
pub use hsv::HSVData;
pub use mono::MonoData;
pub use rgb::RGBData;

pub trait PicData {
    fn new(min: usize, max: usize, video: bool, rng: &mut StdRng, pic_names: &Vec<&String>) -> Pic;
}
