use crate::pic::coordinatesystem::CoordinateSystem;
use crate::parser::aptnode::APTNode;

use ggez::graphics::Color;

#[derive(Clone, Debug, PartialEq)]
pub struct GradientData {
    pub colors: Vec<(Color, bool)>,
    pub index: APTNode,
    pub coord: CoordinateSystem,
}
