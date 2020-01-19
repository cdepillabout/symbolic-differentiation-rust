// https://www.codewars.com/kata/584daf7215ac503d5a0001ae/train/rust

#[macro_use]
extern crate nom;

use nom::Err;
use nom::character::complete::char;
use nom::error::ErrorKind;
use nom::number::complete::float;
use std::ops::{Add, Div, Mul, Sub};
use std::str::FromStr;

fn main() {
    
}


#[derive(Clone, Copy, Debug, PartialEq)]
enum FuncAr2 {
    Plus,
    Minus,
    Times,
    Div,
    Pow,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum FuncAr1 {
    Cos,
    Sin,
    Tan,
    Exp,
    Ln,
}

#[derive(Clone, Debug, PartialEq)]
enum Expr {
    Var,
    Num(f32),
    FuncAr2(FuncAr2, Box<Expr>, Box<Expr>),
    FuncAr1(FuncAr1, Box<Expr>),
}

impl Add for Expr {
    type Output = Expr;

    fn add(self, other: Expr) -> Expr {
        Expr::FuncAr2(FuncAr2::Plus, bx(self), bx(other))
    }
}

impl Sub for Expr {
    type Output = Expr;

    fn sub(self, other: Expr) -> Expr {
        Expr::FuncAr2(FuncAr2::Minus, bx(self), bx(other))
    }
}

impl Mul for Expr {
    type Output = Expr;

    fn mul(self, other: Expr) -> Expr {
        Expr::FuncAr2(FuncAr2::Times, bx(self), bx(other))
    }
}

impl Div for Expr {
    type Output = Expr;

    fn div(self, other: Expr) -> Expr {
        Expr::FuncAr2(FuncAr2::Div, bx(self), bx(other))
    }
}

impl From<i32> for Expr {
    fn from(i: i32) -> Self {
        Expr::Num(i as f32)
    }
}

impl From<f32> for Expr {
    fn from(i: f32) -> Self {
        Expr::Num(i)
    }
}

impl FromStr for Expr {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        parse_expr(s)
            .map(|(_, expr)| expr)
            .map_err(|_| format!("could not parse: {}", s))
    }
}

named!(parse_var<&str, Expr>,
    map!(char('x'), |_| Expr::Var));

named!(parse_num<&str, Expr>,
    map!(float, Expr::Num));

named!(parse_func_ar_1_name<&str, FuncAr1>,
    alt!(
        tag!("ln")  => { |_| FuncAr1::Ln  } |
        tag!("cos") => { |_| FuncAr1::Cos } |
        tag!("sin") => { |_| FuncAr1::Sin } |
        tag!("tan") => { |_| FuncAr1::Tan } |
        tag!("exp") => { |_| FuncAr1::Exp }
    ));

named!(parse_func_ar_2_name<&str, FuncAr2>,
    alt!(
        tag!("+") => { |_| FuncAr2::Plus  } |
        tag!("-") => { |_| FuncAr2::Minus } |
        tag!("*") => { |_| FuncAr2::Times } |
        tag!("/") => { |_| FuncAr2::Div } |
        tag!("^") => { |_| FuncAr2::Pow }
    ));

named!(parse_inner_func_ar_1<&str, Expr>,
    do_parse!(
        func_ar_1: parse_func_ar_1_name >>
        many1!(tag!(" ")) >>
        expr: parse_expr >>
        (Expr::FuncAr1(func_ar_1, bx(expr)))
    ));

named!(parse_inner_func_ar_2<&str, Expr>,
    do_parse!(
        func_ar_2: parse_func_ar_2_name >>
        many1!(tag!(" ")) >>
        expr1: parse_expr >>
        many1!(tag!(" ")) >>
        expr2: parse_expr >>
        (Expr::FuncAr2(func_ar_2, bx(expr1), bx(expr2)))
    ));


named!(parse_inner_func<&str, Expr>,
    alt!(parse_inner_func_ar_1 | parse_inner_func_ar_2));

named!(parse_func<&str, Expr>,
    delimited!(tag!("("), parse_inner_func, tag!(")")));

named!(parse_expr<&str,Expr>,
    alt!(parse_var | parse_num | parse_func));

fn expr_parser(input: &str) -> Expr {
    input
        .parse()
        .expect(&format!("we should never get bad expressions: {}", input))
}

fn pretty_print_func_ar_1(func_ar_1: FuncAr1) -> String {
    match func_ar_1 {
        FuncAr1::Cos => "cos".to_string(),
        FuncAr1::Sin => "sin".to_string(),
        FuncAr1::Tan => "tan".to_string(),
        FuncAr1::Exp => "exp".to_string(),
        FuncAr1::Ln  => "ln".to_string(),
    }
}

fn pretty_print_func_ar_2(func_ar_1: FuncAr2) -> String {
    match func_ar_1 {
        FuncAr2::Plus  => "+".to_string(),
        FuncAr2::Minus => "-".to_string(),
        FuncAr2::Times => "*".to_string(),
        FuncAr2::Div   => "/".to_string(),
        FuncAr2::Pow   => "^".to_string(),
    }
}

fn pretty_print_expr(expr: Expr) -> String {
    match expr {
        Expr::Var => "x".to_string(),
        Expr::Num(i) => i.to_string(),
        Expr::FuncAr1(func_ar_1, expr1) =>
            format!("({} {})",
                pretty_print_func_ar_1(func_ar_1),
                pretty_print_expr(*expr1)),
        Expr::FuncAr2(func_ar_2, expr1, expr2) =>
            format!("({} {} {})",
                pretty_print_func_ar_2(func_ar_2),
                pretty_print_expr(*expr1),
                pretty_print_expr(*expr2)),
    }
}

fn sin(expr: Expr) -> Expr {
    Expr::FuncAr1(FuncAr1::Sin, bx(expr))
}

fn cos(expr: Expr) -> Expr {
    Expr::FuncAr1(FuncAr1::Cos, bx(expr))
}

fn tan(expr: Expr) -> Expr {
    Expr::FuncAr1(FuncAr1::Tan, bx(expr))
}

fn exp(expr: Expr) -> Expr {
    Expr::FuncAr1(FuncAr1::Exp, bx(expr))
}

fn auto_diff_func_ar_1(func_ar_1: FuncAr1, expr: Expr) -> Expr {
    match func_ar_1 {
        FuncAr1::Cos => auto_diff(expr.clone()) * num(-1f32) * sin(expr.clone()),
        FuncAr1::Sin => auto_diff(expr.clone()) * cos(expr.clone()),
        // FuncAr1::Tan => (num(1f32) / pow(cos(expr.clone()), 2.into())) * auto_diff(expr),
        FuncAr1::Tan =>
            auto_diff(expr.clone()) * (num(1f32) + pow(tan(expr.clone()), 2.into())),
        FuncAr1::Exp => auto_diff(expr.clone()) * exp(expr.clone()),
        FuncAr1::Ln => auto_diff(expr.clone()) * (num(1f32) / expr.clone()),
    }
}

fn num(f: f32) -> Expr {
    Expr::Num(f)
}

fn pow(expr1: Expr, expr2: Expr) -> Expr {
    Expr::FuncAr2(FuncAr2::Pow, bx(expr1), bx(expr2))
}

fn auto_diff_func_ar_2(func_ar_2: FuncAr2, expr1: Expr, expr2: Expr) -> Expr {
    match func_ar_2 {
        FuncAr2::Plus => auto_diff(expr1) + auto_diff(expr2),
        FuncAr2::Minus => auto_diff(expr1) - auto_diff(expr2),
        FuncAr2::Times =>
            auto_diff(expr1.clone()) * expr2.clone() + expr1 * auto_diff(expr2),
        FuncAr2::Div =>
            (auto_diff(expr1.clone()) * expr2.clone() - expr1 * auto_diff(expr2.clone())) /
            pow(expr2, 2.into()),
        FuncAr2::Pow =>
            expr2.clone() * pow(expr1.clone(), expr2 - 1.into()) * auto_diff(expr1),
    }
}

fn auto_diff(expr: Expr) -> Expr {
    match expr {
        Expr::Var => 1.into(),
        Expr::Num(_) => 0.into(),
        Expr::FuncAr1(func_ar_1, expr1) =>
            auto_diff_func_ar_1(func_ar_1, *expr1),
        Expr::FuncAr2(func_ar_2, expr1, expr2) =>
            auto_diff_func_ar_2(func_ar_2, *expr1, *expr2),
    }
}

fn simplify_one_step(expr: Expr) -> Expr {
    // - The returned expression should not have unecessary 0 or 1 factors. For example it should not return (* 1 (+ x 1)) but simply the term (+ x 1) similarly it should not return (* 0 (+ x 1)) instead it should return just 0
    // - Results with two constant values such as for example (+ 2 2) should be evaluated and returned as a single value 4
    // - Any argument raised to the zero power should return 1 and if raised to 1 should return the same value or variable. For example (^ x 0) should return 1 and (^ x 1) should return x
    match expr {
        Expr::FuncAr2(FuncAr2::Plus, expr1, expr2) =>
            match (*expr1, *expr2) {
                (Expr::Num(num1), Expr::Num(num2)) => (num1 + num2).into(),
                (Expr::Num(0f32), f) => f,
                (e, Expr::Num(0f32)) => e,
                (e, f) => Expr::FuncAr2(FuncAr2::Plus, bx(e), bx(f)),
            }
        Expr::FuncAr2(FuncAr2::Minus, expr1, expr2) =>
            match (*expr1, *expr2) {
                (Expr::Num(num1), Expr::Num(num2)) => (num1 - num2).into(),
                // (Expr::Num(0), f) => f,
                (e, Expr::Num(0f32)) => e,
                (e, f) => Expr::FuncAr2(FuncAr2::Minus, bx(e), bx(f)),
            }
        Expr::FuncAr2(FuncAr2::Times, expr1, expr2) =>
            match (*expr1, *expr2) {
                (Expr::Num(num1), Expr::Num(num2)) => (num1 * num2).into(),
                (Expr::Num(0f32), _) => 0.into(),
                (_, Expr::Num(0f32)) => 0.into(),
                (Expr::Num(1f32), f) => f,
                (e, Expr::Num(1f32)) => e,
                (e, f) => Expr::FuncAr2(FuncAr2::Times, bx(e), bx(f)),
            }
        Expr::FuncAr2(FuncAr2::Div, expr1, expr2) =>
            match (*expr1, *expr2) {
                (Expr::Num(num1), Expr::Num(num2)) => (num1 / num2).into(),
                (Expr::Num(0f32), _) => 0.into(),
                // (_, Expr::Num(0)) => 0.into(),
                // (Expr::Num(1), f) => f,
                (e, Expr::Num(1f32)) => e,
                (e, f) => Expr::FuncAr2(FuncAr2::Div, bx(e), bx(f)),
            }
        Expr::FuncAr2(FuncAr2::Pow, expr1, expr2) =>
            match (*expr1, *expr2) {
                (Expr::Num(num1), Expr::Num(num2)) => num1.powf(num2).into(),
                (Expr::Num(0f32), _) => 0.into(),
                // (_, Expr::Num(0)) => 0.into(),
                // (Expr::Num(1), f) => f,
                (e, Expr::Num(1f32)) => e,
                (e, f) => Expr::FuncAr2(FuncAr2::Pow, bx(e), bx(f)),
            }
        e => e,
    }
}

fn on_bx<F, T>(t: Box<T>, f: F) -> Box<T> where
    F: FnOnce(T) -> T,
{
    bx(f(*t))
}

fn bx<I>(e: I) -> Box<I> {
    Box::new(e)
}

fn simplify_recurse(expr: Expr) -> Expr {
    match expr {
        Expr::Var => Expr::Var,
        Expr::Num(i) => Expr::Num(i),
        Expr::FuncAr2(func_ar_2, expr1, expr2) =>
            Expr::FuncAr2(func_ar_2, on_bx(expr1, simplify), on_bx(expr2, simplify)),
        Expr::FuncAr1(func_ar_1, expr1) =>
            Expr::FuncAr1(func_ar_1, on_bx(expr1, simplify)),
    }
}

fn simplify(expr: Expr) -> Expr {
    let mut last_expr = expr.clone();
    let mut res_expr = simplify_one_step(simplify_recurse(expr.clone()));

    while res_expr != last_expr.clone() {
        last_expr = res_expr;
        res_expr = simplify_one_step(simplify_recurse(last_expr.clone()));
    }

    res_expr
}

fn diff(expr: &str) -> String {
    // TODO: Implement from_string for expr
    pretty_print_expr(simplify(auto_diff(expr_parser(expr))))
}

fn parse_and_print(s: &str) -> String {
    pretty_print_expr(expr_parser(s))
}

// See https://doc.rust-lang.org/stable/rust-by-example/testing/unit_testing.html

#[cfg(test)]
mod tests {
    use super::*;

    fn test_simp(s: &str) -> String {
        pretty_print_expr(simplify(expr_parser(s)))
    }

    fn test_auto_diff_no_simp(s: &str) -> String {
        pretty_print_expr(auto_diff(expr_parser(s)))
    }

    #[test]
    fn test_parse_var() {
        assert_eq!(parse_var("x"), Ok(("", Expr::Var)));
        assert_eq!(parse_var("y"), Err(Err::Error(("y", ErrorKind::Char))));
    }

    #[test]
    fn test_parse_num() {
        assert_eq!(parse_num("123"), Ok(("", 123.into())));
        assert_eq!(parse_num("y"), Err(Err::Error(("y", ErrorKind::Float))));
    }

    #[test]
    fn test_expr_parser() {
        assert_eq!(parse_and_print("123"), "123");
        assert_eq!(parse_and_print("x"), "x");
        assert_eq!(parse_and_print("(ln 123)"), "(ln 123)");
        assert_eq!(parse_and_print("(ln (cos x))"), "(ln (cos x))");
        assert_eq!(parse_and_print("(+ (cos x) (- x (ln 3)))"), "(+ (cos x) (- x (ln 3)))");
    }

    #[test]
    fn test_parse_expr() {
        assert_eq!(parse_expr("123"), Ok(("", 123.into())));
        assert_eq!(parse_expr("x"), Ok(("", Expr::Var)));
        assert_eq!(parse_expr("y"), Err(Err::Error(("y", ErrorKind::Alt))));
    }

    #[test]
    fn test_simplify() {
        assert_eq!(test_simp("(+ 0 x)"), "x");
        assert_eq!(test_simp("(+ 0 (* (+ x 0) x))"), "(* x x)");
        assert_eq!(test_simp("(* 0 (* (+ x 0) x))"), "0");
        assert_eq!(test_simp("(* 2 0)"), "0");
    }

    #[test]
    fn test_simplify_1() {
        assert_eq!(test_simp("(* 1 x)"), "x");
    }

    #[test]
    fn test_simplify_2() {
        assert_eq!(test_simp("(* x 1)"), "x");
    }

    #[test]
    fn test_simplify_3() {
        assert_eq!(test_simp("(/ 1 2)"), "0.5");
    }

    #[test]
    fn test_auto_diff_no_simp_1() {
        assert_eq!(test_auto_diff_no_simp("(/ x 2)"), "(/ (- (* 1 2) (* x 0)) (^ 2 2))");
    }

    #[test]
    fn test_simplify_4() {
        assert_eq!(test_simp("(* x 0)"), "0");
        assert_eq!(test_simp("(* 1 2)"), "2");
        assert_eq!(test_simp("(+ (* 1 2) (* x 0))"), "2");
        assert_eq!(test_simp("(^ 2 2)"), "4");
    }

    #[test]
    fn test_simplify_5() {
        assert_eq!(test_simp("(/ 2 4)"), "0.5");
    }

    #[test]
    fn test_simplify_6() {
        assert_eq!(test_simp("(/ 2 4)"), "0.5");
    }

    // #[test]
    // fn test_auto_diff_no_simp_2() {
    //     assert_eq!(test_auto_diff_no_simp("(tan x)"), "(* (+ 1 (^ (tan x) 2)) 1)")
    // }

    #[test]
    fn test_simplify_7() {
        assert_eq!(test_simp("(* (+ 1 (^ (tan x) 2)) 1)"), "(+ 1 (^ (tan x) 2))")
    }

    // #[test]
    // fn test_auto_diff_no_simp_3() {
    //     assert_eq!(test_auto_diff_no_simp("(/ 2 (+ 1 x))"), "(/ (- (* 0 (+ 1 x)) (* 2 (+ 0 1))) (^ (+ 1 x) 2)2")
    // }
        // assert_eq!(diff("(/ 2 (+ 1 x))"), "(/ -2 (^ (+ 1 x) 2))");

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

