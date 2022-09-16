use ggez::graphics::Image;
use ggez::Context;
use image::io::Reader as ImageReader;
use image::GenericImageView;

pub struct ActualPicture {
    pub brightness: Vec<f32>,
    pub w: u16,
    pub h: u16,
    pub name: String,
}

impl ActualPicture {
    pub fn new_via_ctx(
        ctx: &mut Context,
        relative_file_name: &str,
    ) -> Result<ActualPicture, String> {
        let img = Image::new(ctx, "/".to_string() + &relative_file_name).unwrap();
        let raw_bytes = img.to_rgba8(ctx).unwrap();
        Self::new_from_bytes(
            &raw_bytes[0..],
            relative_file_name,
            img.width(),
            img.height(),
        )
    }

    pub fn new_via_file(file_name: &str) -> Result<Self, String> {
        let img = ImageReader::open(file_name)
            .expect("Could not open file")
            .decode()
            .expect("Could not decode file");

        let (width, height) = img.dimensions();
        let raw_bytes = img.to_bytes();
        Self::new_from_bytes(&raw_bytes[0..], file_name, width as u16, height as u16)
    }

    pub fn new_from_bytes(raw_bytes: &[u8], name: &str, w: u16, h: u16) -> Result<Self, String> {
        let brightness: Vec<f32> = raw_bytes
            .chunks_exact(4)
            .map(|chunk| {
                let sum: u16 = chunk[0] as u16 + chunk[1] as u16 + chunk[2] as u16;
                (sum as f32 / (255.0 * 3.0)) * 2.0 - 1.0
            })
            .collect();
        Ok(Self {
            brightness,
            w,
            h,
            name: name.to_string(),
        })
    }
}
