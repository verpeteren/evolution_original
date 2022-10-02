use std::collections::HashMap;
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;

use crate::parser::{aptnode::APTNode, lexer::Lexer, token::Token};
use crate::pic::actual_picture::ActualPicture;
use crate::pic::coordinatesystem::CoordinateSystem;
use crate::pic::data::gradient::GradientData;
use crate::pic::data::grayscale::GrayscaleData;
use crate::pic::data::hsv::HSVData;
use crate::pic::data::mono::MonoData;
use crate::pic::data::rgb::RGBData;
use crate::pic::data::PicData;

use ggez::graphics::Color;
use rand::prelude::*;
use rand::rngs::StdRng;
use simdeez::avx2::*;
use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;
use simdeez::Simd;

pub const WIDTH: usize = 1920;
pub const HEIGHT: usize = 1080;

const TREE_MIN: usize = 1;
const TREE_MAX: usize = 40;

simd_runtime_generate!(
    pub fn pic_get_rgba8(
        pic: &Pic,
        threaded: bool,
        pictures: Arc<HashMap<String, ActualPicture>>,
        width: usize,
        height: usize,
        t: f32,
    ) -> Vec<u8> {
        pic.get_rgba8::<S>(threaded, pictures, width, height, t)
    }
);

#[derive(Clone, Debug, PartialEq)]
pub enum Pic {
    Mono(MonoData),
    Grayscale(GrayscaleData),
    RGB(RGBData),
    HSV(HSVData),
    Gradient(GradientData),
}

impl Pic {
    pub fn new(rng: &mut StdRng, pic_names: &Vec<&String>) -> Self {
        let pic_type = rng.gen_range(0..5);

        let pic = match pic_type {
            0 => MonoData::new(TREE_MIN, TREE_MAX, false, rng, pic_names),
            1 => GradientData::new(TREE_MIN, TREE_MAX, false, rng, pic_names),
            2 => RGBData::new(TREE_MIN, TREE_MAX, false, rng, pic_names),
            3 => HSVData::new(TREE_MIN, TREE_MAX, false, rng, pic_names),
            4 => GrayscaleData::new(TREE_MIN, TREE_MAX, false, rng, pic_names),
            _ => panic!("invalid"),
        };
        pic
    }

    pub fn to_tree(&self) -> Vec<&APTNode> {
        match self {
            Pic::Grayscale(data) => vec![&data.c],
            Pic::Mono(data) => vec![&data.c],
            Pic::Gradient(data) => vec![&data.index],
            Pic::RGB(data) => vec![&data.r, &data.g, &data.b],
            Pic::HSV(data) => vec![&data.h, &data.s, &data.v],
        }
    }

    pub fn to_lisp(&self) -> String {
        match self {
            Pic::Mono(data) => format!(
                "( MONO {}\n ( {} ) )",
                data.coord.to_string().to_uppercase(),
                data.c.to_lisp()
            ),
            Pic::Grayscale(data) => {
                format!(
                    "( GRAYSCALE {}\n ( {} ) )",
                    data.coord.to_string().to_uppercase(),
                    data.c.to_lisp()
                )
            }
            Pic::Gradient(data) => {
                let mut colors = "( COLORS ".to_string();
                for (color, stop) in &data.colors {
                    if *stop {
                        colors += &format!(" ( STOPCOLOR {} {} {} )", color.r, color.g, color.b);
                    } else {
                        colors += &format!(" ( COLOR {} {} {} )", color.r, color.g, color.b);
                    }
                }
                format!(
                    "( GRADIENT {}\n {} {} )",
                    data.coord.to_string().to_uppercase(),
                    colors,
                    data.index.to_lisp()
                )
            }
            Pic::RGB(data) => format!(
                "( RGB {}\n ( {} )\n ( {} )\n ( {} ) )",
                data.coord.to_string().to_uppercase(),
                data.r.to_lisp(),
                data.g.to_lisp(),
                data.b.to_lisp()
            ),
            Pic::HSV(data) => format!(
                "( HSV {}\n ( {} )\n ( {} )\n ( {} ) )",
                data.coord.to_string().to_uppercase(),
                data.h.to_lisp(),
                data.s.to_lisp(),
                data.v.to_lisp()
            ),
        }
    }

    pub fn get_video<S: Simd>(
        &self,
        pics: Arc<HashMap<String, ActualPicture>>,
        w: usize,
        h: usize,
        fps: u16,
        d: f32,
    ) -> Vec<Vec<u8>> {
        let frames = (fps as f32 * (d / 1000.0)) as i32;
        let frame_dt = 2.0 / frames as f32;

        let mut t = -1.0;
        let mut result = Vec::new();
        for _ in 0..frames {
            let frame_buffer = self.get_rgba8::<S>(true, pics.clone(), w, h, t);
            result.push(frame_buffer);
            t += frame_dt;
        }
        result
    }

    pub fn coord(&self) -> &CoordinateSystem {
        match self {
            Pic::Mono(data) => &data.coord,
            Pic::Grayscale(data) => &data.coord,
            Pic::Gradient(data) => &data.coord,
            Pic::RGB(data) => &data.coord,
            Pic::HSV(data) => &data.coord,
        }
    }

    pub fn get_rgba8<S: Simd>(
        &self,
        threaded: bool,
        pics: Arc<HashMap<String, ActualPicture>>,
        w: usize,
        h: usize,
        t: f32,
    ) -> Vec<u8> {
        match self {
            Pic::Mono(data) => data.get_rgba8::<S>(threaded, pics, w, h, t),
            Pic::Grayscale(data) => data.get_rgba8::<S>(threaded, pics, w, h, t),
            Pic::Gradient(data) => data.get_rgba8::<S>(threaded, pics, w, h, t),
            Pic::RGB(data) => data.get_rgba8::<S>(threaded, pics, w, h, t),
            Pic::HSV(data) => data.get_rgba8::<S>(threaded, pics, w, h, t),
        }
    }
}

pub fn lisp_to_pic(code: String, coord: CoordinateSystem) -> Result<Pic, String> {
    let mut pic_opt = None;
    rayon::scope(|s| {
        let (sender, receiver) = channel();
        s.spawn(|_| {
            Lexer::begin_lexing(&code, sender);
        });

        // TODO: fix race condition that crashes at parser.rs:68. Workaround:
        std::thread::sleep(std::time::Duration::from_millis(1));

        pic_opt = Some(parse_pic(&receiver, coord))
    });
    pic_opt.unwrap()
}

#[must_use]
pub fn expect_open_paren(receiver: &Receiver<Token>) -> Result<(), String> {
    let open_paren = receiver.recv().map_err(|_| "Unexpected end of file")?;
    match open_paren {
        Token::OpenParen(_) => Ok(()),
        Token::Operation(v, line) | Token::Constant(v, line) => {
            return Err(format!("Expected '(' on line {}, got a '{}'", line, v))
        }
        _ => {
            return Err(format!(
                "Expected '(' on line {}",
                extract_line_number(&open_paren)
            ))
        }
    }
}

#[must_use]
pub fn expect_close_paren(receiver: &Receiver<Token>) -> Result<(), String> {
    let close_paren = receiver.recv().map_err(|_| "Unexpected end of file")?;
    match close_paren {
        Token::CloseParen(_) => Ok(()),
        _ => {
            return Err(format!(
                "Expected '(' on line {}",
                extract_line_number(&close_paren)
            ))
        }
    }
}

#[must_use]
pub fn expect_operation(s: &str, receiver: &Receiver<Token>) -> Result<(), String> {
    let op = receiver.recv().map_err(|_| "Unexpected end of file")?;
    match op {
        Token::Operation(op_str, _) => {
            if op_str.to_lowercase() == s {
                Ok(())
            } else {
                Err(format!(
                    "Expected '{}' on line {}, found {}",
                    s,
                    extract_line_number(&op),
                    op_str
                ))
            }
        }
        _ => {
            return Err(format!(
                "Expected '{}' on line {}, found {:?}",
                s,
                extract_line_number(&op),
                op
            ))
        }
    }
}

#[must_use]
pub fn expect_operations(ops: Vec<&str>, receiver: &Receiver<Token>) -> Result<String, String> {
    let op = receiver.recv().map_err(|_| "Unexpected end of file")?;
    for s in ops {
        match op {
            Token::Operation(op_str, _) => {
                if op_str.to_lowercase() == s.to_lowercase() {
                    return Ok(op_str.to_string());
                }
            }
            _ => (),
        }
    }
    return Err(format!(
        "Unexpected token on line {}",
        extract_line_number(&op),
    ));
}

#[must_use]
pub fn expect_constant(receiver: &Receiver<Token>) -> Result<f32, String> {
    let op = receiver.recv().map_err(|_| "Unexpected end of file")?;
    match op {
        Token::Constant(vstr, line_number) => {
            let v = vstr
                .parse::<f32>()
                .map_err(|_| format!("Unable to parse number {} on line {}", vstr, line_number))?;
            Ok(v)
        }
        _ => {
            return Err(format!(
                "Expected constant on line {}, found {:?}",
                extract_line_number(&op),
                op
            ))
        }
    }
}

pub fn parse_pic(
    receiver: &Receiver<Token>,
    coord_default: CoordinateSystem,
) -> Result<Pic, String> {
    let mut coord = coord_default;
    expect_open_paren(receiver)?;
    let pic_type = receiver.recv().map_err(|_| "Unexpected end of file")?;
    match pic_type {
        Token::Operation(s, line_number) => match &s.to_lowercase()[..] {
            "mono" => {
                if let Ok(coord_system) = expect_operations(
                    CoordinateSystem::list_all()
                        .iter()
                        .map(|x| x.as_str())
                        .collect(),
                    receiver,
                ) {
                    coord = coord_system.parse().unwrap();
                };
                Ok(Pic::Mono(MonoData {
                    c: APTNode::parse_apt_node(receiver)?,
                    coord,
                }))
            }
            "grayscale" => {
                if let Ok(coord_system) = expect_operations(
                    CoordinateSystem::list_all()
                        .iter()
                        .map(|x| x.as_str())
                        .collect(),
                    receiver,
                ) {
                    coord = coord_system.parse().unwrap();
                };
                Ok(Pic::Grayscale(GrayscaleData {
                    c: APTNode::parse_apt_node(receiver)?,
                    coord,
                }))
            }
            "rgb" => {
                if let Ok(coord_system) = expect_operations(
                    CoordinateSystem::list_all()
                        .iter()
                        .map(|x| x.as_str())
                        .collect(),
                    receiver,
                ) {
                    coord = coord_system.parse().unwrap();
                };
                Ok(Pic::RGB(RGBData {
                    r: APTNode::parse_apt_node(receiver)?,
                    g: APTNode::parse_apt_node(receiver)?,
                    b: APTNode::parse_apt_node(receiver)?,
                    coord,
                }))
            }
            "hsv" => {
                if let Ok(coord_system) = expect_operations(
                    CoordinateSystem::list_all()
                        .iter()
                        .map(|x| x.as_str())
                        .collect(),
                    receiver,
                ) {
                    coord = coord_system.parse().unwrap();
                };
                Ok(Pic::HSV(HSVData {
                    h: APTNode::parse_apt_node(receiver)?,
                    s: APTNode::parse_apt_node(receiver)?,
                    v: APTNode::parse_apt_node(receiver)?,
                    coord,
                }))
            }
            "gradient" => {
                if let Ok(coord_system) = expect_operations(
                    CoordinateSystem::list_all()
                        .iter()
                        .map(|x| x.as_str())
                        .collect(),
                    receiver,
                ) {
                    coord = coord_system.parse().unwrap();
                };
                let mut colors = Vec::new();
                expect_open_paren(receiver)?;
                expect_operation("colors", receiver)?;
                loop {
                    let _token = receiver.recv().map_err(|_| "Unexpected end of file")?;
                    match expect_operations(vec!["color", "stopcolor"], receiver) {
                        Err(e) => {
                            if e.starts_with("Unexpected token on line ") {
                                break;
                            } else {
                                panic!("{:?}", e);
                            }
                        }
                        Ok(color_type) => {
                            let r = expect_constant(receiver)?;
                            let g = expect_constant(receiver)?;
                            let b = expect_constant(receiver)?;
                            if color_type == "color" {
                                colors.push((Color::new(r, g, b, 1.0), false));
                            } else {
                                colors.push((Color::new(r, g, b, 1.0), true));
                            }
                            expect_close_paren(receiver)?;
                        }
                    }
                }
                Ok(Pic::Gradient(GradientData {
                    colors: colors,
                    index: APTNode::parse_apt_node(receiver)?,
                    coord,
                }))
            }
            _ => Err(format!("Unknown pic type {} at line {}", s, line_number)),
        },
        _ => Err(format!("Invalid picture type")), //todo line number etc
    }
}

pub fn extract_line_number(token: &Token) -> usize {
    match token {
        Token::OpenParen(ln) | Token::CloseParen(ln) => *ln,
        Token::Constant(_, ln) | Token::Operation(_, ln) => *ln,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::token::Token;
    use crate::pic::coordinatesystem::DEFAULT_COORDINATE_SYSTEM;
    use image::buffer::Pixels;
    use image::io::Reader as ImageReader;
    use image::{DynamicImage, GenericImageView, ImageBuffer, ImageFormat};
    use std::io::{Cursor, Read};

    #[test]
    fn test_pic_new_mono() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let pic = MonoData::new(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        match &pic {
            Pic::Mono(MonoData { c, coord: _coord }) => {
                let len = c.get_children().unwrap().len();
                assert!(len > 0 && len < 60);
            }
            _ => {
                panic!("wrong type");
            }
        };
    }

    #[test]
    fn test_pic_new_grayscale() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let pic = GrayscaleData::new(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        match &pic {
            Pic::Grayscale(GrayscaleData { c, coord: _coord }) => {
                let len = c.get_children().unwrap().len();
                assert!(len > 0 && len < 60);
            }
            _ => {
                panic!("wrong type");
            }
        };
    }

    #[test]
    fn test_pic_new_gradient() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let pic = GradientData::new(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        match &pic {
            Pic::Gradient(GradientData {
                colors,
                index,
                coord: _coord,
            }) => {
                let len = colors.len();
                assert!(len > 1 && len < 10);
                let len = index.get_children().unwrap().len();
                assert!(len > 0 && len < 60);
            }
            _ => {
                panic!("wrong type");
            }
        };
    }

    #[test]
    fn test_pic_new_rgb() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let pic = RGBData::new(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        match &pic {
            Pic::RGB(RGBData { r, g, b, coord: _ }) => {
                let len = r.get_children().unwrap().len();
                assert!(len > 0 && len < 60);

                let len = g.get_children().unwrap().len();
                assert!(len > 0 && len < 60);

                let len = b.get_children().unwrap().len();
                assert!(len > 0 && len < 60);
            }
            _ => {
                panic!("wrong type");
            }
        };
    }

    #[test]
    fn test_pic_new_hsv() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();
        let pic = HSVData::new(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        match &pic {
            Pic::HSV(HSVData { h, s, v, coord: _ }) => {
                let len = h.get_children().unwrap().len();
                assert!(len > 0 && len < 60);

                let len = s.get_children().unwrap().len();
                assert!(len > 0 && len < 60);

                let len = v.get_children().unwrap().len();
                assert!(len > 0 && len < 60);
            }
            _ => {
                panic!("wrong type");
            }
        };
    }

    #[test]
    fn test_pic_to_lisp() {
        let mut rng = StdRng::from_rng(rand::thread_rng()).unwrap();

        let pic = MonoData::new(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        let sexpr = pic.to_lisp();

        assert!(sexpr.starts_with("( MONO POLAR\n (") || sexpr.starts_with("( MONO CARTESIAN\n ("));
        assert!(sexpr.ends_with(" )"));
        assert!(sexpr.lines().collect::<Vec<_>>().len() > 1);

        let pic = GrayscaleData::new(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        let sexpr = pic.to_lisp();
        assert!(
            sexpr.starts_with("( GRAYSCALE POLAR\n (")
                || sexpr.starts_with("( GRAYSCALE CARTESIAN\n (")
        );
        assert!(sexpr.ends_with(" )"));
        assert!(sexpr.lines().collect::<Vec<_>>().len() > 1);

        let pic = GradientData::new(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        let sexpr = pic.to_lisp();
        assert!(
            sexpr.starts_with("( GRADIENT POLAR\n (")
                || sexpr.starts_with("( GRADIENT CARTESIAN\n (")
        );
        assert!(sexpr.ends_with(" )"));
        assert!(sexpr.contains("( COLORS ") || sexpr.contains(" ( STOPCOLOR "));
        assert!(sexpr.contains(" ( COLOR "));
        assert!(sexpr.lines().collect::<Vec<_>>().len() > 0);

        let pic = RGBData::new(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        let sexpr = pic.to_lisp();
        assert!(sexpr.starts_with("( RGB POLAR\n (") || sexpr.starts_with("( RGB CARTESIAN\n ("));
        assert!(sexpr.ends_with(" )"));
        assert!(sexpr.lines().collect::<Vec<_>>().len() > 3);

        let pic = HSVData::new(0, 60, false, &mut rng, &vec![&"eye.jpg".to_string()]);
        let sexpr = pic.to_lisp();
        assert!(sexpr.starts_with("( HSV POLAR\n (") || sexpr.starts_with("( HSV CARTESIAN\n ("));
        assert!(sexpr.ends_with(" )"));
        assert!(sexpr.lines().collect::<Vec<_>>().len() > 1);
    }

    // todo: refactor into a separate module e.g. parser::token
    #[test]
    fn test_extract_line_number() {
        assert_eq!(extract_line_number(&Token::OpenParen(6)), 6);
        assert_eq!(extract_line_number(&Token::CloseParen(6)), 6);
        assert_eq!(extract_line_number(&Token::Operation("blablabla", 6)), 6);
        assert_eq!(extract_line_number(&Token::Constant("blablabla", 6)), 6);
    }

    #[test]
    fn test_pic_coord() {
        assert_eq!(
            lisp_to_pic("(Mono Polar (X) )".to_string(), CoordinateSystem::Polar)
                .unwrap()
                .coord(),
            &CoordinateSystem::Polar
        );
        assert_eq!(
            lisp_to_pic(
                "(Mono Cartesian (X) )".to_string(),
                CoordinateSystem::Cartesian
            )
            .unwrap()
            .coord(),
            &CoordinateSystem::Cartesian
        );
        assert_eq!(
            lisp_to_pic("(Mono (X) )".to_string(), CoordinateSystem::Polar)
                .unwrap()
                .coord(),
            &CoordinateSystem::Polar
        );
    }

    #[test]
    //todo Currently wrong CoordinateSystems are still accepted, but ignored
    fn test_pic_coord_fail() {
        lisp_to_pic("(Mono Lunar (X) )".to_string(), CoordinateSystem::Polar)
            .unwrap()
            .coord();
    }

    #[test]
    fn test_handle_width() {
        let sexpr = "(GrayScale ( / x Width ) )";
        match lisp_to_pic(sexpr.to_string(), CoordinateSystem::Polar) {
            Ok(pic) => {
                assert_eq!(
                    pic,
                    Pic::Grayscale(GrayscaleData {
                        c: APTNode::Div(vec![APTNode::X, APTNode::Width]),
                        coord: CoordinateSystem::Polar
                    })
                );
                let resexpr = pic.to_lisp();
                assert_eq!(resexpr, "( GRAYSCALE POLAR\n ( ( / X WIDTH ) ) )");
            }
            Err(err) => {
                panic!("could not parse formula with width {:?}", err);
            }
        }
    }

    #[test]
    fn test_handle_height() {
        let sexpr = "(GrayScale ( / y Height ) )";
        match lisp_to_pic(sexpr.to_string(), CoordinateSystem::Polar) {
            Ok(pic) => {
                assert_eq!(
                    pic,
                    Pic::Grayscale(GrayscaleData {
                        c: APTNode::Div(vec![APTNode::Y, APTNode::Height]),
                        coord: CoordinateSystem::Polar
                    })
                );
                let resexpr = pic.to_lisp();
                assert_eq!(resexpr, "( GRAYSCALE POLAR\n ( ( / Y HEIGHT ) ) )");
            }
            Err(err) => {
                panic!("could not parse formula with width {:?}", err);
            }
        }
    }

    #[test]
    fn test_handle_pi() {
        let sexpr = "(GrayScale( sin (/ x PI ) ) )";
        match lisp_to_pic(sexpr.to_string(), CoordinateSystem::Polar) {
            Ok(pic) => {
                assert_eq!(
                    pic,
                    Pic::Grayscale(GrayscaleData {
                        c: APTNode::Sin(vec![APTNode::Div(vec![APTNode::X, APTNode::PI,])]),
                        coord: CoordinateSystem::Polar
                    })
                );
                let resexpr = pic.to_lisp();
                assert_eq!(resexpr, "( GRAYSCALE POLAR\n ( ( SIN ( / X PI ) ) ) )");
            }
            Err(err) => {
                panic!("could not parse formula with PI {:?}", err);
            }
        }
    }

    #[test]
    fn test_handle_e() {
        let sexpr = "(GrayScale( Log (/ x E ) ) )";
        match lisp_to_pic(sexpr.to_string(), CoordinateSystem::Polar) {
            Ok(pic) => {
                assert_eq!(
                    pic,
                    Pic::Grayscale(GrayscaleData {
                        c: APTNode::Log(vec![APTNode::Div(vec![APTNode::X, APTNode::E,])]),
                        coord: CoordinateSystem::Polar
                    })
                );
                let resexpr = pic.to_lisp();
                assert_eq!(resexpr, "( GRAYSCALE POLAR\n ( ( LOG ( / X E ) ) ) )");
            }
            Err(err) => {
                panic!("could not parse formula with E {:?}", err);
            }
        }
    }

    #[test]
    fn test_handle_mono_coord_system_polar() {
        let sexpr = "(Mono POLAR ( X ) )";
        match lisp_to_pic(sexpr.to_string(), DEFAULT_COORDINATE_SYSTEM) {
            Ok(pic) => {
                assert_eq!(
                    pic,
                    Pic::Mono(MonoData {
                        c: APTNode::X,
                        coord: CoordinateSystem::Polar
                    })
                );
                let resexpr = pic.to_lisp();
                assert_eq!(resexpr, "( MONO POLAR\n ( X ) )");
            }
            Err(err) => {
                panic!("could not parse formula with E {:?}", err);
            }
        }
    }

    #[test]
    fn test_handle_mono_coord_system_cartesian() {
        let sexpr = "(Mono CARTESIAN ( X )";
        match lisp_to_pic(sexpr.to_string(), DEFAULT_COORDINATE_SYSTEM) {
            Ok(pic) => {
                assert_eq!(
                    pic,
                    Pic::Mono(MonoData {
                        c: APTNode::X,
                        coord: CoordinateSystem::Cartesian
                    })
                );
                let resexpr = pic.to_lisp();
                assert_eq!(resexpr, "( MONO CARTESIAN\n ( X ) )"); //todo if coord != DEFAULT print
            }
            Err(err) => {
                panic!("could not parse formula with E {:?}", err);
            }
        }
    }

    #[test]
    fn test_handle_rgb_coord_system_cartesian() {
        let sexpr = "(RGB CARTESIAN ( X ) (Y) (T) )";
        match lisp_to_pic(sexpr.to_string(), DEFAULT_COORDINATE_SYSTEM) {
            Ok(pic) => {
                assert_eq!(
                    pic,
                    Pic::RGB(RGBData {
                        r: APTNode::X,
                        g: APTNode::Y,
                        b: APTNode::T,
                        coord: CoordinateSystem::Cartesian
                    })
                );
                let resexpr = pic.to_lisp();
                assert_eq!(resexpr, "( RGB CARTESIAN\n ( X )\n ( Y )\n ( T ) )");
                //todo if coord != DEFAULT print
            }
            Err(err) => {
                panic!("could not parse formula with E {:?}", err);
            }
        }
    }

    #[test]
    fn test_handle_rgb_coord_system_polar() {
        let sexpr = "(RGB POLAR ( X ) ( Y ) (T) )";
        match lisp_to_pic(sexpr.to_string(), DEFAULT_COORDINATE_SYSTEM) {
            Ok(pic) => {
                assert_eq!(
                    pic,
                    Pic::RGB(RGBData {
                        r: APTNode::X,
                        g: APTNode::Y,
                        b: APTNode::T,
                        coord: CoordinateSystem::Polar
                    })
                );
                let resexpr = pic.to_lisp();
                assert_eq!(resexpr, "( RGB POLAR\n ( X )\n ( Y )\n ( T ) )");
            }
            Err(err) => {
                panic!("could not parse formula with E {:?}", err);
            }
        }
    }

    #[test]
    fn test_handle_hsv_coord_system_cartesian() {
        let sexpr = "(HSV CARTESIAN ( X ) (Y) (T)";
        match lisp_to_pic(sexpr.to_string(), DEFAULT_COORDINATE_SYSTEM) {
            Ok(pic) => {
                assert_eq!(
                    pic,
                    Pic::HSV(HSVData {
                        h: APTNode::X,
                        s: APTNode::Y,
                        v: APTNode::T,
                        coord: CoordinateSystem::Cartesian
                    })
                );
                let resexpr = pic.to_lisp();
                assert_eq!(resexpr, "( HSV CARTESIAN\n ( X )\n ( Y )\n ( T ) )");
                //todo if coord != DEFAULT print
            }
            Err(err) => {
                panic!("could not parse formula with E {:?}", err);
            }
        }
    }

    #[test]
    fn test_handle_hsv_coord_system_polar() {
        let sexpr = "(HSV POLAR ( X ) ( Y) (T) )";
        match lisp_to_pic(sexpr.to_string(), DEFAULT_COORDINATE_SYSTEM) {
            Ok(pic) => {
                assert_eq!(
                    pic,
                    Pic::HSV(HSVData {
                        h: APTNode::X,
                        s: APTNode::Y,
                        v: APTNode::T,
                        coord: CoordinateSystem::Polar
                    })
                );
                let resexpr = pic.to_lisp();
                assert_eq!(resexpr, "( HSV POLAR\n ( X )\n ( Y )\n ( T ) )");
            }
            Err(err) => {
                panic!("could not parse formula with E {:?}", err);
            }
        }
    }
    #[test]
    fn test_handle_grayscale_coord_system_cartesian() {
        let sexpr = "(GrayScale CARTESIAN ( X )";
        match lisp_to_pic(sexpr.to_string(), DEFAULT_COORDINATE_SYSTEM) {
            Ok(pic) => {
                assert_eq!(
                    pic,
                    Pic::Grayscale(GrayscaleData {
                        c: APTNode::X,
                        coord: CoordinateSystem::Cartesian
                    })
                );
                let resexpr = pic.to_lisp();
                assert_eq!(resexpr, "( GRAYSCALE CARTESIAN\n ( X ) )"); //todo if coord != DEFAULT print
            }
            Err(err) => {
                panic!("could not parse formula with E {:?}", err);
            }
        }
    }

    #[test]
    #[ignore]
    fn test_crash_with_dims() {
        let crashes_at_dim = (100, 100);
        let pictures = Arc::new(HashMap::new());
        let pic = Pic::RGB(RGBData {
            r: APTNode::X,
            g: APTNode::Y,
            b: APTNode::T,
            coord: CoordinateSystem::Polar,
        });
        let _x = pic_get_rgba8_runtime_select(
            &pic,
            true,
            pictures,
            crashes_at_dim.0,
            crashes_at_dim.1,
            0.0,
        );
    }

    fn render_source_and_read_sample_file<'a>(
        source: String,
        sample_file: &'a str,
    ) -> (DynamicImage, DynamicImage) {
        let pictures = Arc::new(HashMap::new());
        let pic = lisp_to_pic(source, DEFAULT_COORDINATE_SYSTEM).unwrap();
        let gen_rgba8 = pic_get_rgba8_runtime_select(&pic, true, pictures, WIDTH, HEIGHT, 0.0);
        let gen_buf = ImageBuffer::from_raw(WIDTH as u32, HEIGHT as u32, gen_rgba8).unwrap();
        let generated = DynamicImage::ImageRgba8(gen_buf);

        let read_img = ImageReader::open(sample_file).unwrap().decode().unwrap();
        let read = DynamicImage::ImageRgba8(read_img.into_rgba8());

        return (generated, read);
    }

    #[test]
    fn test_parse_and_render_mono() {
        let img_file = "./samples/mono.png";
        let source = r#"( MONO CARTESIAN
         ( ( ATAN ( + ( CELL1 Y Y Y X ( - Y 0.7253959 ) ) ( ATAN X ) ) ) ) )
         "#;
        let (generated, read) = render_source_and_read_sample_file(source.to_string(), img_file);
        assert_eq!(generated.dimensions(), read.dimensions());
        assert_eq!(generated.as_bytes(), read.as_bytes());
    }

    #[test]
    fn test_parse_and_render_grayscale() {
        let img_file = "./samples/grayscale.png";
        let source = r#"( GRAYSCALE POLAR
 ( ( LOG ( + ( CELL1 ( LOG ( RIDGE ( SQRT Y ) Y Y X X 0.5701809 ) ) ( ATAN Y ) ( % Y 0.12452102 ) ( FLOOR ( ATAN2 Y Y ) ) ( SIN Y ) ) ( * ( + X ( SIN ( - ( ATAN2 Y X ) X ) ) ) ( ATAN ( LOG ( FLOOR ( SIN ( TURBULENCE Y 0.91551733 ( SQRT ( SQRT X ) ) ( MIN X Y ) -0.83923936 ( MANDELBROT Y X ) ) ) ) ) ) ) ) ) ) )
         "#;
        let (generated, read) = render_source_and_read_sample_file(source.to_string(), img_file);
        assert_eq!(generated.dimensions(), read.dimensions());
        assert_eq!(generated.as_bytes(), read.as_bytes());
    }

    #[test]
    fn test_parse_and_render_hsv() {
        let img_file = "./samples/hsv.png";
        let source = r#"( HSV CARTESIAN
 ( ( SQUARE ( / ( MANDELBROT X Y ) 0.7601185 ) ) )
 ( ( + ( TAN ( TAN ( RIDGE Y ( ATAN -0.74197626 ) ( + Y Y ) Y ( CLAMP Y ) ( + X Y ) ) ) ) ( ATAN2 X Y ) ) )
 ( ( MAX -0.9284358 Y ) ) )"#;
        let (generated, read) = render_source_and_read_sample_file(source.to_string(), img_file);
        assert_eq!(generated.dimensions(), read.dimensions());
        assert_eq!(generated.as_bytes(), read.as_bytes());
    }

    #[test]
    fn test_parse_and_render_rgb() {
        let img_file = "./samples/rgb.png";
        let source = r#"( RGB CARTESIAN
  ( ( * ( TAN ( CLAMP ( ATAN ( SQRT ( MAX ( ABS ( FLOOR ( RIDGE 0.12349105 ( + X 0.500072 ) X X ( MAX Y Y ) 0.6249633 ) ) ) ( % ( CLAMP ( * Y ( SQUARE 0.39180493 ) ) ) ( WRAP ( CELL2 Y ( MIN -0.5756769 Y ) ( ABS 0.8329663 ) Y Y ) ) ) ) ) ) ) ) ( WRAP ( MANDELBROT ( SQRT ( TURBULENCE ( WRAP X ) 0.26766992 ( MANDELBROT -0.7147219 0.46446967 ) ( LOG 0.6340864 ) Y Y ) ) ( SQUARE ( * ( SIN ( / Y ( RIDGE X Y Y 0.49542284 X ( CEIL -0.7545812 ) ) ) ) ( CEIL ( TURBULENCE ( ATAN X ) X -0.52819157 -0.86907744 0.49089026 ( ATAN -0.5986686 ) ) ) ) ) ) ) ) )
  ( ( / ( TURBULENCE ( FBM Y ( * ( RIDGE Y X X X X Y ) -0.98887086 ) 0.21490455 X X ( LOG X ) ) X ( % ( FLOOR X ) ( + X ( ATAN2 0.19268274 Y ) ) ) ( FBM Y -0.28251457 0.632663 X X X ) ( CEIL ( SQRT 0.8429725 ) ) ( WRAP ( MAX Y ( SQUARE ( TAN X ) ) ) ) ) ( FLOOR ( CELL1 ( + -0.5022187 ( LOG X ) ) ( RIDGE -0.8493159 Y ( TAN X ) Y Y Y ) ( ATAN ( SIN ( / ( ABS X ) ( CEIL 0.05049467 ) ) ) ) ( ATAN X ) ( TAN ( / ( FBM X X 0.802964 0.3002789 0.8905289 -0.06338668 ) ( SQUARE ( % X 0.48889422 ) ) ) ) ) ) ) )
  ( ( ATAN ( SIN X ) ) ) )"#;
        let (generated, read) = render_source_and_read_sample_file(source.to_string(), img_file);
        assert_eq!(generated.dimensions(), read.dimensions());
        assert_eq!(generated.as_bytes(), read.as_bytes());
    }

    #[test]
    fn test_parse_and_render_gradient() {
        let img_file = "./samples/gradient.png";
        let source = r#"( GRADIENT POLAR
 ( COLORS  ( COLOR 0.38782334 0.18356442 0.5526812 ) ( COLOR 0.40132487 0.9418049 0.79687893 ) ( SQRT ( FBM ( WRAP ( TAN ( - -0.90357685 ( ATAN Y ) ) ) ) ( ABS X ) ( ATAN2 Y X ) ( MAX Y ( MAX X X ) ) ( SQUARE ( CELL2 ( TAN Y ) Y Y X X ) ) ( * Y 0.009492159 ) ) ) )"#;
        let (generated, read) = render_source_and_read_sample_file(source.to_string(), img_file);
        assert_eq!(generated.dimensions(), read.dimensions());
        assert_eq!(generated.as_bytes(), read.as_bytes());
    }
}
