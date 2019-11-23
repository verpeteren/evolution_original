use ggez::graphics::Color;
use rand::rngs::StdRng;
use rand::*;

pub fn lerp_color(a: Color, b: Color, pct: f32) -> Color {
    let red = a.r * (1.0 - pct) + b.r * pct;
    let green = a.g * (1.0 - pct) + b.g * pct;
    let blue = a.b * (1.0 - pct) + b.b * pct;
    let alpha = a.a * (1.0 - pct) + b.a * pct;
    Color::new(red, green, blue, alpha)
}

pub fn get_random_color(rng: &mut StdRng) -> Color {
    let r = rng.gen_range(0.0, 1.0);
    let g = rng.gen_range(0.0, 1.0);
    let b = rng.gen_range(0.0, 1.0);
    Color::new(r, g, b, 1.0)
}
