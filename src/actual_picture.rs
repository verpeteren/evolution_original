use ggez::Context;
use ggez::graphics::{Image};

pub struct ActualPicture {
    pub brightness: Vec<f32>,
    pub w: u16,
    pub h: u16,
    pub name: String,
}

impl ActualPicture {
    pub fn new(ctx: &mut Context, img: Image, name: String) -> ActualPicture {
        let raw_bytes = img.to_rgba8(ctx).unwrap();
        println!("raw len:{}", raw_bytes.len());
        let brightness: Vec<f32> = raw_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let sum: u16 = chunk[0] as u16 + chunk[1] as u16 + chunk[2] as u16;
                (sum as f32 / (255.0 * 3.0)) * 2.0 - 1.0
            })
            .collect();
        println!("brightlen:{}", brightness.len());
        ActualPicture {
            brightness: brightness,
            w: img.width(),
            h: img.height(),
            name: name,
        }
    }
}
