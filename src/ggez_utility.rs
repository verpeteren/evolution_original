use ggez::graphics::Color;
use rand::rngs::StdRng;
use rand::prelude::*;

pub fn lerp_color(a: Color, b: Color, pct: f32) -> Color {
    let red = a.r * (1.0 - pct) + b.r * pct;
    let green = a.g * (1.0 - pct) + b.g * pct;
    let blue = a.b * (1.0 - pct) + b.b * pct;
    let alpha = a.a * (1.0 - pct) + b.a * pct;
    Color::new(red, green, blue, alpha)
}

pub fn get_random_color(rng: &mut StdRng) -> Color {
    let r = rng.gen_range(0.0..1.0);
    let g = rng.gen_range(0.0..1.0);
    let b = rng.gen_range(0.0..1.0);
    Color::new(r, g, b, 1.0)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp_color() {
        let red = Color::RED;
        let white = Color::WHITE;
        let cyan = Color::CYAN;
        let magenta = Color::MAGENTA;

        let expected_1 = Color::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(lerp_color(red, white, 1.0), expected_1);

        let expected_2 = Color::new(1.0, 0.0, 0.0, 1.0);
        assert_eq!(lerp_color(white, red, 1.0), expected_2);

        let expected_3 = Color::new(0.54, 0.45999998, 1.0, 1.0);
        assert_eq!(lerp_color(cyan, magenta, 0.54), expected_3);

        let expected_4 = Color::new(0.14445001, 0.85555, 1.0, 1.0);
        assert_eq!(lerp_color(magenta, cyan, 0.85555), expected_4);
    }

    #[test]
    fn test_get_random_color() {
         let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
         let color = get_random_color(&mut rng);
         assert!(color.r >= 0.0 && color.r <= 1.0);
         assert!(color.g >= 0.0 && color.g <= 1.0);
         assert!(color.b >= 0.0 && color.b <= 1.0);
         assert_eq!(color.a, 1.0);
    }

}
