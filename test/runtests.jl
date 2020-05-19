using Test
using DeepShiba
#include("../src/DeepShiba.jl")

const e = 10e-5
@testset "DiffTest" begin
    x = Variable(0.5, nothing, nothing)
    f = Func(
        x -> x ^ 2,
        x -> 2x,
        nothing,
        nothing
    )
    y = f(x)
    y.grad = 1
    backward!(y)
    @test 1.0 - e <= x.grad <= 1.0 + e
end
