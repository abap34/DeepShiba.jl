using Test
using DeepShiba

function numerical_diff(f::Function, x::Real; e=10e-4)
    return (f(x + e) - f(x - e)) / 2e
end

function numerical_diff(f::Function, xs::AbstractArray; e=1e-4)
    grads = zeros(length(xs)) 
    for idx in 1:length(xs)
        tmp_val = xs[idx]
        xs[idx] = tmp_val + e
        fxh1 = f(xs...)
        xs[idx] = tmp_val - e
        fxh2 = f(xs...)
        grads[idx] = (fxh1 - fxh2) / 2e
        xs[idx] = tmp_val
    end
    return grads
end

function isAbout(x, y; e=10e-4)
    return y - e <= x <= y + e
end




@testset "NumericalDiffTest" begin
    f(x) = 2x^3 + x^2 + 3x + 2
    f′(x) = 6x^2 + 2x + 3
    for i in 0:0.1:1
        num_diff = numerical_diff(f, i)
        @test isAbout(num_diff, f′(i)) 
    end
end


@testset "OperatorTest" begin
    operators = [+, -, *, /]
    basic_func(a, b, c) = (a^2 - 4b^2 + 10) / c
    # This simple function contains all the basic operators.
    @testset "OperatorForwardTest" begin
        a, b = rand(2)
        shiba_a, shiba_b = variable.((a, b))
        for op in operators
            y = op(a, b)
            shiba_y = op(shiba_a, shiba_b)
            @test y == shiba_y.data
        end
    end

    @testset "OparatorBackwardTest" begin
        a, b = rand(2)
        shiba_a, shiba_b = variable.((a, b))
        for op in operators
            cleargrad!(shiba_a)
            cleargrad!(shiba_b)
            y = op(a, b)
            shiba_y = op(shiba_a, shiba_b)
            backward!(shiba_y)
            shiba_a_grad = shiba_a.grad.data
            shiba_b_grad = shiba_b.grad.data
            numerical_grads = numerical_diff(op, [a, b])
            @test isAbout(shiba_a_grad, numerical_grads[1])
            @test isAbout(shiba_b_grad, numerical_grads[2])
        end
    end


    @testset "BasicFunBackwardTest" begin
        a, b, c = rand(3)
        shiba_a, shiba_b, shiba_c = variable.((a, b, c))
        grads = numerical_diff(basic_func, [a, b, c])
        shiba_y = basic_func(shiba_a, shiba_b, shiba_c)
        backward!(shiba_y)
        for (shiba_grad, numerical_grad) in zip([shiba_a, shiba_b, shiba_c], [a, b, c])
            @test isAbout(numerical_grad, shiba_grad.data)
        end
    end

end


@testset "MathFuncTest" begin
    math_funcs = [sin, cos, tan, tanh]
    @testset "OneOrgMathFuncForwardTest" begin
        x = rand()
        shiba_x = variable(x)
        for func in math_funcs
            y = func(x)
            shiba_y = func(shiba_x)
            @test isAbout(y, shiba_y.data)
        end
    end
    @testset "OneOrgMathFuncBackwardTest" begin
        x = rand()
        shiba_x = variable(x)
        for func in math_funcs
            cleargrad!(shiba_x)
            shiba_y = func(shiba_x)
            backward!(shiba_y)
            numerical_grad = numerical_diff(func, x)
            @test isAbout(shiba_x.grad.data, numerical_grad)
        end
    end
end
