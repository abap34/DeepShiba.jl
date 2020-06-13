module DeepShiba
    include("utils.jl")
    export Variable, Func, variable, func, backward!, numerical_diff, cleargrad!, print, display, +, -, *, /, plot
end 
