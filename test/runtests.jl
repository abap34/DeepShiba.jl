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
    x = Variable(0.5, nothing, nothing)
    f = Func(
        x -> x^2,
        x -> 2x,
        nothing,
        nothing
    )
    y = f(x)
    y.grad = 1
    backward!(y)
    @test 1.0 - e <= x.grad <= 1.0 + e
end
