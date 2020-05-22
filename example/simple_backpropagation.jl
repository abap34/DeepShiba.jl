using DeepShiba


Square(x) = func(
    x -> x^2,
    x -> 2x
)(x)

Exp(x) = func(
    x -> exp(x),
    x -> exp(x),
)(x)

Add(x1, x2) = func(
    (x1, x2) -> x1 + x2,
    x -> 1,
)(x1, x2)


x1 = variable(0.5)
x2 = variable(1.5)
y = Add(Square(x1), Exp(x2))
backward!(y)
