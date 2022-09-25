use crate::parser::aptnode::APTNode;
use crate::pic::coordinatesystem::CoordinateSystem;

#[derive(Clone, Debug, PartialEq)]
pub struct MonoData {
    pub c: APTNode,
    pub coord: CoordinateSystem,
}
