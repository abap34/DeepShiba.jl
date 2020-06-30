using Test
using DeepShiba



function isAbout(x, y; e=10e-5)
    return y - e <= x <= y + e
end

const operators = [+, -, *, /, ^]


basic_func(a, b, c) = (a^b - 4b^2 + 10) / c
# This simple function contains all the basic operators.


@testset "NumericalDiffTest" begin
    f(x) = 2x^3 + x^2 + 3x + 2
    f′(x) = 6x^2 + 2x + 3
    for i in 0:0.1:1
        num_diff = numerical_diff(f, i)
        @test isAbout(num_diff, f′(i)) 
    end
end

 

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
        @test isAbout(shiba_a_grad, numerical_grads[1], e=10e-4)
        @test isAbout(shiba_b_grad,numerical_grads[2], e=10e-4)
    end
end


@testset "BasicFunBackwardTest" begin
    a, b, c = rand(3)
    shiba_a, shiba_b, shiba_c = variable.((a, b, c))
    grads = numerical_diff(basic_func, [a, b, c], e=10e-4)
    shiba_y = basic_func(shiba_a, shiba_b, shiba_c)
    backward!(shiba_y)
    for (shiba_grad, numerical_grad) in zip([shiba_a, shiba_b, shiba_c], [a, b, c])
        @test isAbout(numerical_grad, shiba_grad.data, e=10e-4)
    end
end