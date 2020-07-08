module DeepShiba
    include("core_structures.jl")
    include("core_functions.jl")
    include("utils.jl")
    include("functions.jl")
    export Variable, Func, variable, backward!, numerical_diff, cleargrad!, show, +, -, *, /, reshape, sin, cos, tanh, plot
end 
