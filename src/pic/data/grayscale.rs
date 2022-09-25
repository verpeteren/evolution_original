use crate::pic::coordinatesystem::CoordinateSystem;
use crate::parser::aptnode::APTNode;

#[derive(Clone, Debug, PartialEq)]
pub struct GrayscaleData {
    pub c: APTNode,
    pub coord: CoordinateSystem,
}
