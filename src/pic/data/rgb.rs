use crate::parser::aptnode::APTNode;
use crate::pic::coordinatesystem::CoordinateSystem;

#[derive(Clone, Debug, PartialEq)]
pub struct RGBData {
    pub r: APTNode,
    pub g: APTNode,
    pub b: APTNode,
    pub coord: CoordinateSystem,
}
