using Test
using DeepShiba



function isAbout(x, y, e=10e-5)
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



