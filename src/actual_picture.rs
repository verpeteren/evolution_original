use ggez::graphics::{self, Image};
use ggez::Context;
pub struct ActualPicture {
    raw_bytes: Vec<u8>,
    w: u16,
    h: u16,
    name: String,
}

impl ActualPicture {
    pub fn new(ctx: &mut Context, img: graphics::Image, name: String) -> ActualPicture {
        ActualPicture {
            raw_bytes: img.to_rgba8(ctx).unwrap(),
            w: img.width(),
            h: img.height(),
            name: name,
        }
    }
}
