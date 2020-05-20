module DeepShiba
    include("core_simple.jl")
    include("utils.jl")
    export  Variable,Func,backward!, numerical_diff
end 
