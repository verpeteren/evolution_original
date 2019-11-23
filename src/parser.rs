use crate::apt::*;
use std::collections::VecDeque;
use std::sync::mpsc::*;
use std::thread;

#[derive(Debug)]
pub enum Token<'a> {
    OpenParen,
    CloseParen,
    Operation(&'a str),
    Constant(&'a str),
}

struct StateFunction(fn(&mut Lexer) -> Option<StateFunction>);

pub struct Lexer<'a> {
    input: &'a str,
    start: usize,
    pos: usize,
    width: usize,
    token_sender: Sender<Token<'a>>,
}

impl<'a> Lexer<'a> {
    pub fn begin_lexing(s: &'a str, sender: Sender<Token<'a>>) {
        let mut lexer = Lexer::<'a> {
            input: s,
            start: 0,
            pos: 0,
            width: 0,
            token_sender: sender,
        };

        lexer.run();
    }

    fn run(&mut self) {
        let mut state = Some(StateFunction(determine_token));
        //while let syntax here?
        while state.is_some() {
            state = state.unwrap().0(self)
        }
    }

    //todo learn how unicode
    fn next(&mut self) -> Option<char> {
        if self.pos >= self.input.len() {
            self.width = 0;
            None
        } else {
            self.width = 1;
            let c = self.input[self.pos..].chars().next().unwrap();
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
        println!("emitting a token");
        self.token_sender.send(token);
        self.start = self.pos;
    }

    fn accept(&mut self, valid: &str) -> bool {
        let n = self.next();
        if n.is_some() && valid.contains(n.unwrap()) {
            true
        } else {
            self.backup();
            return false;
        }
    }

    fn accept_run(&mut self, valid: &str) {
        loop {
            let n = self.next();
            if !(n.is_some() && valid.contains(n.unwrap())) {
                break;
            }
        }
        self.backup();
    }
}

fn lex_operation(l: &mut Lexer) -> Option<StateFunction> {
    l.accept_run("+-/*abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789");
    l.emit(Token::Operation(&l.input[l.start..l.pos]));
    return Some(StateFunction(determine_token));
}

fn lex_number(l: &mut Lexer) -> Option<StateFunction> {
    l.accept("-");
    let digits = "0123456789";
    l.accept_run(digits);
    if l.accept(".") {
        l.accept_run(digits);
    }
    if &l.input[l.start..l.pos] == "-" {
        l.emit(Token::Operation(&l.input[l.start..l.pos]));
    } else {
        l.emit(Token::Constant(&l.input[l.start..l.pos]));
    }
    return Some(StateFunction(determine_token));
}

fn determine_token(l: &mut Lexer) -> Option<StateFunction> {
    loop {
        match l.next() {
            Some(c) => {
                if is_white_space(c) {
                    l.ignore();
                } else if c == '(' {
                    l.emit(Token::OpenParen);
                } else if c == ')' {
                    l.emit(Token::CloseParen);
                } else if is_start_of_number(c) {
                    return Some(StateFunction(lex_number));
                } else {
                    return Some(StateFunction(lex_operation));
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

pub fn parse(receiver: &Receiver<Token>) -> APTNode {
    loop {
        match receiver.recv() {
            Ok(token) => {
                match token {
                    Token::Operation(s) => {
                        println!("returning a node");
                        let mut node = APTNode::str_to_node(s);
                        match node.get_children_mut() {
                            Some(children) => {
                                for child in children {
                                    *child = parse(receiver);
                                }
                                return node;
                            }
                            None => return node,
                        }
                    }
                    Token::Constant(vstr) => {
                        let v = vstr.parse::<f32>().unwrap();
                        println!("returning a node");
                        return APTNode::Constant(v);
                    }
                    _ => (), //parens don't matter
                }
            }
            Err(_) => panic!("malformed input"),
        }
    }
}
