using Test
using DeepShiba


const e = 10e-5

@testset "NumericalDiffTest" begin
    f(x) = 2x^3 + x^2 + 3x + 2
    f′(x) = 6x^2 + 2x + 3
    for i in 0:0.1:1
        @test (f′(i) - e) <= (numerical_diff(f, i)) <= (f′(i) + e) 
    end
end

@testset "DiffTest" begin
    x = variable(0.5)
    f(x) = Func(
        x -> x^2,
        x -> 2x,
        nothing,
        nothing,
        0,
        "f"
    )(x)
    y = f(x)
    y.grad = 1
    backward!(y)
    @test 1.0 - e <= x.grad <= 1.0 + e
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
    @test ((0.5^2) + exp(1.5) - e) <= y.data <= ((0.5^2) + exp(1.5) + e)
end

