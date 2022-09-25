
#[derive(Debug, PartialEq)]
pub enum Token<'a> {
    OpenParen(usize),
    CloseParen(usize),
    Operation(&'a str, usize),
    Constant(&'a str, usize),
}


