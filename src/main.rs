// https://www.codewars.com/kata/584daf7215ac503d5a0001ae/train/rust

#[macro_use]
extern crate nom;

use nom::{Err, IResult};
use nom::branch::alt;
use nom::character::complete::{char, digit1};
use nom::error::ErrorKind;
use std::str::FromStr;

fn main() {
    
}


#[derive(Debug, PartialEq)]
enum FuncAr2 {
    Plus,
    Minus,
    Times,
    Div,
    Pow,
}

#[derive(Debug, PartialEq)]
enum FuncAr1 {
    Cos,
    Sin,
    Tan,
    Exp,
    Ln,
}

#[derive(Debug, PartialEq)]
enum Expr {
    Var,
    Num(i32),
    FuncAr2(FuncAr2, Box<Expr>, Box<Expr>),
    FuncAr1(FuncAr1, Box<Expr>),
}

named!(parse_var<&str, Expr>,
    map!(char('x'), |_| Expr::Var));

named!(parse_num<&str, Expr>,
    map_res!(digit1, |s: &str| s.parse::<i32>().map(Expr::Num)));

named!(parse_expr<&str,Expr>,
    alt!(parse_var | parse_num));

fn expr_parser(input: &str) -> Expr {
    parse_expr(input)
        .expect(&format!("we should never get bad expressions: {}", input))
        .1
}

fn diff(expr: &str) -> String {
    expr.to_string()
}

// See https://doc.rust-lang.org/stable/rust-by-example/testing/unit_testing.html

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_var() {
        assert_eq!(parse_var("x"), Ok(("", Expr::Var)));
        assert_eq!(parse_var("y"), Err(Err::Error(("y", ErrorKind::Char))));
    }

    #[test]
    fn test_parse_num() {
        assert_eq!(parse_num("123"), Ok(("", Expr::Num(123))));
        assert_eq!(parse_num("y"), Err(Err::Error(("y", ErrorKind::Digit))));
    }

    #[test]
    fn test_parse_expr() {
        assert_eq!(parse_expr("123"), Ok(("", Expr::Num(123))));
        assert_eq!(parse_expr("x"), Ok(("", Expr::Var)));
        assert_eq!(parse_expr("y"), Err(Err::Error(("y", ErrorKind::Alt))));
    }

    #[test]
    fn test_fixed() {
        assert_eq!(diff("5"), "0");
        assert_eq!(diff("x"), "1");
        assert_eq!(diff("5"), "0");
        assert_eq!(diff("(+ x x)"), "2");
        assert_eq!(diff("(- x x)"), "0");
        assert_eq!(diff("(* x 2)"), "2");
        assert_eq!(diff("(/ x 2)"), "0.5");
        assert_eq!(diff("(^ x 2)"), "(* 2 x)");
        assert_eq!(diff("(cos x)"), "(* -1 (sin x))");
        assert_eq!(diff("(sin x)"), "(cos x)");
        assert_eq!(diff("(tan x)"), "(+ 1 (^ (tan x) 2))");
        assert_eq!(diff("(exp x)"), "(exp x)");
        assert_eq!(diff("(ln x)"), "(/ 1 x)");
        assert_eq!(diff("(+ x (+ x x))"), "3");
        assert_eq!(diff("(- (+ x x) x)"), "1");
        assert_eq!(diff("(* 2 (+ x 2))"), "2");
        assert_eq!(diff("(/ 2 (+ 1 x))"), "(/ -2 (^ (+ 1 x) 2))");
        assert_eq!(diff("(cos (+ x 1))"), "(* -1 (sin (+ x 1)))");

        let result = diff("(cos (* 2 x))");
        assert!(
            result == "(* 2 (* -1 (sin (* 2 x))))"
                || result == "(* -2 (sin (* 2 x)))"
                || result == "(* (* -1 (sin (* 2 x))) 2)"
        );

        assert_eq!(diff("(sin (+ x 1))"), "(cos (+ x 1))");
        assert_eq!(diff("(sin (* 2 x))"), "(* 2 (cos (* 2 x)))");
        assert_eq!(diff("(tan (* 2 x))"), "(* 2 (+ 1 (^ (tan (* 2 x)) 2)))");
        assert_eq!(diff("(exp (* 2 x))"), "(* 2 (exp (* 2 x)))");
        assert_eq!(diff(&diff("(sin x)")), "(* -1 (sin x))");
        assert_eq!(diff(&diff("(exp x)")), "(exp x)");

        let result = diff(&diff("(^ x 3)"));
        assert!(result == "(* 3 (* 2 x))" || result == "(* 6 x)");
    }
}

