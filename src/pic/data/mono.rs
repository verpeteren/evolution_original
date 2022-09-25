use crate::pic::coordinatesystem::CoordinateSystem;
use crate::parser::aptnode::APTNode;

#[derive(Clone, Debug, PartialEq)]
pub struct MonoData {
    pub c: APTNode,
    pub coord: CoordinateSystem,
}
