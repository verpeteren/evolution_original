use crate::parser::aptnode::APTNode;
use crate::pic::coordinatesystem::CoordinateSystem;

#[derive(Clone, Debug, PartialEq)]
pub struct HSVData {
    pub h: APTNode,
    pub s: APTNode,
    pub v: APTNode,
    pub coord: CoordinateSystem,
}
