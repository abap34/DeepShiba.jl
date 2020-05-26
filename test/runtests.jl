using Test
using DeepShiba



function isAbout(x, y, e=10e-5)
    return x - e <= x <= x + e
end

@testset "NumericalDiffTest" begin
    f(x) = 2x^3 + x^2 + 3x + 2
    f′(x) = 6x^2 + 2x + 3
    for i in 0:0.1:1
        num_diff = numerical_diff(f, i)
        @test isAbout(num_diff, f′(i)) 
    end
end


@testset "DiffTest" begin
    # ===================
    x = variable(0.5)
    f(x) = Func(
        x->x^2,
        x->2x,
        nothing,
        nothing,
        0,
        "f"
    )(x)
    y = f(x)
    y.grad = 1
    backward!(y)
    @test isAbout(x.grad, 1.0)
    # ===================
    
    
    # ===================
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
        (x1, x2) -> (1, 1)
    )(x1, x2)


    x1 = variable(0.5)
    x2 = variable(1.5)
    y = Add(Square(x1), Exp(x2))
    backward!(y)
    @test isAbout(y.data, (0.5^2) + exp(1.5))
    # ===================

    # ===================
    x = variable(2.0)

    Square(x) = func(
        x -> x^2,
        x -> 2x,
        "Square"
    )(x)


    Add(x1, x2) = func(
    (x1, x2) -> x1 + x2,
    (x1, x2) -> (1, 1),
    "Add"
    )(x1, x2)

    a = Square(x)
    y = Add(Square(a), Square(a))
    backward!(y)
    @test isAbout(y.data, 32)
    @test isAbout(x.grad, 54)
    # =====================

    # =====================
    sphere(x, y) = x^2 + y^2

    x = variable(1.0)
    y = variable(1.0)
    z = sphere(x, y)
    backward!(z)
    @test isAbout(x.grad, 2)
    @test isAbout(y.grad, 2)

    # ====================

end
