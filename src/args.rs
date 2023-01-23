use clap::Parser;

use crate::{
    CoordinateSystem, DEFAULT_COORDINATE_SYSTEM, DEFAULT_IMAGE_HEIGHT, DEFAULT_IMAGE_WIDTH,
    DEFAULT_PICTURES_PATH,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    #[clap(short, long, value_parser, default_value = DEFAULT_PICTURES_PATH, help="The path to images that can be loaded via the Pic- operation")]
    pub pictures_path: String,

    #[clap(short, long, value_parser, default_value_t = DEFAULT_IMAGE_WIDTH, help="The width of the generated image")]
    pub width: u32,

    #[clap(long, value_parser, default_value_t = DEFAULT_IMAGE_HEIGHT, help="The height of the generated image")]
    pub height: u32,

    #[clap(
        short,
        long,
        value_parser,
        default_value_t = 0.0,
        help = "set the T variable (ms)"
    )]
    pub time: f32,

    #[clap(
        short,
        long,
        value_parser,
        help = "filename to read sexpr from and disabling the UI; Use '-' to read from stdin."
    )]
    pub input: Option<String>,

    #[clap(
        short,
        long,
        value_parser,
        requires("input"),
        help = "image file to write to"
    )]
    pub output: Option<String>,

    #[clap(
        short,
        long,
        value_parser,
        requires("input"),
        help = "The path where to store a copy of the input and output files as part of the creative workflow"
    )]
    pub copy_path: Option<String>,

    #[clap(short='s', long, value_parser, default_value_t = DEFAULT_COORDINATE_SYSTEM, help="The Coordinate system to use")]
    pub coordinate_system: CoordinateSystem,
}
