module DeepShiba
    include("core/core_structures.jl")
    include("core/core_functions.jl")
    include("utils/output_txt.jl")
    include("utils/plot.jl")
    include("functions/math_functions.jl")
    include("functions/trans_functions.jl")
    export Variable, Func, variable, backward!, numerical_diff, cleargrad!, show, +, -, *, /, reshape, sin, cos, tanh, plot
end 
