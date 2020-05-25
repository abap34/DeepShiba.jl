module DeepShiba
    include("core_simple.jl")
    include("utils.jl")
    export  Variable, Func, variable, func, backward!, numerical_diff, cleargrad, print
end 
