module DeepShiba
    include("utils.jl")
    export Variable, Func, variable, backward!, numerical_diff, cleargrad!, print, display, +, -, *, /, plot
end 
