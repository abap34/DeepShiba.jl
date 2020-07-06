module DeepShiba
    include("core.jl")
    include("utils.jl")
    include("functions.jl")
    export Variable, Func, variable, backward!, numerical_diff, cleargrad!, print, display, +, -, *, /, reshape, sin, cos, plot
end 
