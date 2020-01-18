// https://www.codewars.com/kata/584daf7215ac503d5a0001ae/train/rust


use nom::branch::alt;
use nom::char::complete::char;

fn main() {
    
}

enum FuncAr2 {
    Plus,
    Minus,
    Times,
    Div,
    Pow,
}

enum FuncAr1 {
    Cos,
    Sin,
    Tan,
    Exp,
    Ln,
}

enum Expr {
    Var,
    Num(i32),
    FuncAr2(FuncAr2, Box<Expr>, Box<Expr>),
    FuncAr1(FuncAr1, Box<Expr>),
}

fn parse_var(input: &str) -> IResult<&str, Expr> {
    char('x')(expr)
}

fn parse_num(expr: &str) -> IResult<&str, Expr> {
}

fn parse_expr(expr: &str) -> Expr {
    parse_var(input)
    // alt((parse_var, parse_num , parse_func_ar_2, parse_func_ar_1))(input)
        .expect(format!("we should never get bad expressions: {}", str))
        .0
}

fn diff(expr: &str) -> String {
    expr.to_string()
}

// See https://doc.rust-lang.org/stable/rust-by-example/testing/unit_testing.html

#[cfg(test)]
mod tests {
    use super::*;

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

