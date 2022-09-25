use std::sync::mpsc::Sender;

pub mod token;

pub use token::Token;

// Function pointer definition must be wrapped in a struct to be recursive
struct StateFunction(fn(&mut Lexer) -> Option<StateFunction>);

pub struct Lexer<'a> {
    input: &'a str,
    start: usize,
    pos: usize,
    width: usize,
    token_sender: Sender<Token<'a>>,
    current_line: usize,
}

impl<'a> Lexer<'a> {
    pub fn begin_lexing(s: &'a str, sender: Sender<Token<'a>>) {
        let mut lexer = Lexer::<'a> {
            input: s,
            start: 0,
            pos: 0,
            width: 0,
            token_sender: sender,
            current_line: 0,
        };
        lexer.run();
    }

    fn run(&mut self) {
        let mut state = Some(StateFunction(Lexer::determine_token));
        while let Some(next_state) = state {
            state = next_state.0(self)
        }
    }

    fn next(&mut self) -> Option<char> {
        if self.pos >= self.input.len() {
            self.width = 0;
            None
        } else {
            self.width = 1; // Assuming one always for now
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("unexpected end of input");
            if Lexer::is_linebreak(c) {
                self.current_line += 1;
            }
            self.pos += self.width;
            Some(c)
        }
    }

    fn backup(&mut self) {
        self.pos -= 1;
    }

    fn ignore(&mut self) {
        self.start = self.pos;
    }

    fn emit(&mut self, token: Token<'a>) {
        // println!("token:{:?}", token);
        self.token_sender.send(token).expect("token send failure");
        self.start = self.pos;
    }

    fn accept(&mut self, valid: &str) -> bool {
        if let Some(n) = self.next() {
            if valid.contains(n) {
                true
            } else {
                self.backup();
                false
            }
        } else {
            self.backup();
            return false;
        }
    }

    fn accept_run(&mut self, valid: &str) {
        loop {
            let n = self.next();
            if !(n.is_some() && valid.contains(n.expect("unexpected character in token stream"))) {
                break;
            }
        }
        self.backup();
    }

    fn lex_operation(l: &mut Lexer) -> Option<StateFunction> {
        l.accept_run("+-/*%abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.");
        l.emit(Token::Operation(&l.input[l.start..l.pos], l.current_line));
        return Some(StateFunction(Lexer::determine_token));
    }

    fn lex_number(l: &mut Lexer) -> Option<StateFunction> {
        l.accept("-");
        let digits = "0123456789";
        l.accept_run(digits);
        if l.accept(".") {
            l.accept_run(digits);
        }
        if &l.input[l.start..l.pos] == "-" {
            // special case - could indicate start of number, or subtract operation
            l.emit(Token::Operation(&l.input[l.start..l.pos], l.current_line));
        } else {
            l.emit(Token::Constant(&l.input[l.start..l.pos], l.current_line));
        }
        return Some(StateFunction(Lexer::determine_token));
    }

    fn determine_token(l: &mut Lexer) -> Option<StateFunction> {
        loop {
            match l.next() {
                Some(c) => {
                    if Lexer::is_white_space(c) {
                        l.ignore();
                    } else if c == '(' {
                        l.emit(Token::OpenParen(l.current_line));
                    } else if c == ')' {
                        l.emit(Token::CloseParen(l.current_line));
                    } else if Lexer::is_start_of_number(c) {
                        return Some(StateFunction(Lexer::lex_number));
                    } else {
                        return Some(StateFunction(Lexer::lex_operation));
                    }
                }
                None => return None,
            }
        }
    }

    fn is_start_of_number(c: char) -> bool {
        (c >= '0' && c <= '9') || c == '-' || c == '.'
    }

    fn is_white_space(c: char) -> bool {
        c == ' ' || c == '\n' || c == '\t' || c == '\r'
    }

    fn is_linebreak(c: char) -> bool {
        c == '\n'
    }
}

#[cfg(test)]
pub mod mock {
    use super::*;

    pub fn mock_lexer<'a>(code: &'a str, sender: Sender<Token<'a>>) -> Lexer<'a> {
        Lexer {
            input: code,
            start: 0,
            pos: 0,
            width: 1,
            token_sender: sender,
            current_line: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc::channel;

    const CODE: &'static str = r#"( RGB
    ( Sqrt ( Sin ( Abs Y ) ) )
    ( Atan ( Atan2 ( + X ( / ( Ridge Y -0.30377412 Y ) -0.4523425 ) ) ( + ( Turbulence 0.95225644 ( Tan Y ) Y ) -0.46079302 ) ) )
    ( Cell1 ( Ridge ( Ridge Y -0.83537865 -0.50440097 ) ( Atan2 Y X ) ( Sin 0.20003605 ) ) ( Sqrt ( Cell1 ( FBM Y X 0.8879242 ) 0.23509383 -0.4539826 ) ) ( Atan2 ( * X ( Ridge 0.6816149 X Y ) ) ( Cell1 ( Sin ( Turbulence X -0.25605845 Y ) ) -0.30595016 Y ) ) ) )
"#;

    #[test]
    fn test_lexer_next_linebreak() {
        let (sender, _receiver) = channel::<Token>();
        let mut lexer = mock::mock_lexer(CODE, sender);
        let expected = vec![
            (0, 0, 1, 1, 0),
            (1, 0, 2, 1, 0),
            (2, 0, 3, 1, 0),
            (3, 0, 4, 1, 0),
            (4, 0, 5, 1, 0),
            (5, 0, 6, 1, 1),
        ];
        for (i, exp) in expected.iter().enumerate() {
            lexer.next();
            assert_eq!((i, lexer.start), (exp.0, exp.1));
            assert_eq!((i, lexer.pos), (exp.0, exp.2));
            assert_eq!((i, lexer.width), (exp.0, exp.3));
            assert_eq!((i, lexer.current_line), (exp.0, exp.4));
        }
    }
    #[test]
    fn test_lexer_next_end() {
        let (sender, _receiver) = channel::<Token>();
        let mut lexer = mock::mock_lexer(CODE, sender);
        lexer.pos = CODE.len();
        assert_eq!(lexer.next(), None);
    }

    #[test]
    fn test_lexer_backup() {
        let (sender, _receiver) = channel::<Token>();
        let mut lexer = mock::mock_lexer(CODE, sender);
        lexer.next();
        lexer.backup();
        assert_eq!(lexer.pos, 0);
        assert_eq!(lexer.current_line, 0);
        assert_eq!(lexer.start, 0);
        assert_eq!(lexer.width, 1);
    }

    #[test]
    fn test_lexer_ignore() {
        let (sender, _receiver) = channel::<Token>();
        let mut lexer = mock::mock_lexer(CODE, sender);
        lexer.next();
        lexer.ignore();
        assert_eq!(lexer.pos, 1);
        assert_eq!(lexer.current_line, 0);
        assert_eq!(lexer.start, 1);
        assert_eq!(lexer.width, 1);
    }

    #[test]
    fn test_lexer_emit() {
        let (sender, receiver) = channel::<Token>();
        let mut lexer = mock::mock_lexer(CODE, sender);
        lexer.next();
        lexer.emit(Token::OpenParen(66));
        assert_eq!(lexer.pos, 1);
        assert_eq!(lexer.current_line, 0);
        assert_eq!(lexer.start, 1);
        assert_eq!(lexer.width, 1);
        let msg = receiver.recv().unwrap();
        assert_eq!(msg, Token::OpenParen(66));
    }

    #[test]
    fn test_lexer_accept_start_ok() {
        let (sender, _receiver) = channel::<Token>();
        let mut lexer = mock::mock_lexer(CODE, sender);
        assert_eq!(lexer.accept("("), true);
    }

    #[test]
    fn test_lexer_accept_start_not_ok() {
        let (sender, _receiver) = channel::<Token>();
        let mut lexer = mock::mock_lexer(CODE, sender);
        assert_eq!(lexer.accept(")"), false);
    }

    #[test]
    fn test_lexer_accept_spaced_ok() {
        let (sender, _receiver) = channel::<Token>();
        let mut lexer = mock::mock_lexer(CODE, sender);
        lexer.next();
        assert_eq!(lexer.accept(" \t"), true);
    }

    #[test]
    fn test_lexer_accept_spaced_not_ok() {
        let (sender, _receiver) = channel::<Token>();
        let mut lexer = mock::mock_lexer(CODE, sender);
        lexer.next();
        assert_eq!(lexer.accept("abcdef"), false);
    }

    #[test]
    fn test_lexer_accept_end() {
        let (sender, _receiver) = channel::<Token>();
        let mut lexer = mock::mock_lexer(CODE, sender);
        for _i in 0..CODE.len() {
            lexer.next();
        }
        assert_eq!(lexer.accept(")"), false);
    }

    #[test]
    fn test_is_start_of_number() {
        assert_eq!(Lexer::is_start_of_number('a'), false);
        assert_eq!(Lexer::is_start_of_number('0'), true);
        assert_eq!(Lexer::is_start_of_number('9'), true);
    }

    #[test]
    fn test_is_white_space() {
        assert_eq!(Lexer::is_white_space('a'), false);
        assert_eq!(Lexer::is_white_space(' '), true);
        assert_eq!(Lexer::is_white_space('\t'), true);
        assert_eq!(Lexer::is_white_space('\n'), true);
        assert_eq!(Lexer::is_white_space('\r'), true);
    }

    #[test]
    fn test_is_linebreak() {
        assert_eq!(Lexer::is_linebreak('a'), false);
        assert_eq!(Lexer::is_linebreak(' '), false);
        assert_eq!(Lexer::is_linebreak('\t'), false);
        assert_eq!(Lexer::is_linebreak('\n'), true);
        assert_eq!(Lexer::is_linebreak('\r'), false);
    }
}
