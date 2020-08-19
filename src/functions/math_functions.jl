@DeepShiba_Func mutable struct Log <: Func end

@DeepShiba_Func mutable struct Sin <: Func end

@DeepShiba_Func mutable struct Cos <: Func end

@DeepShiba_Func mutable struct Tan <: Func end

@DeepShiba_Func mutable struct Tanh <: Func end


_Log(x::Variable) = Log([x, ], nothing, 0)(x)
_Sin(x) = Sin([x, ],   nothing, 0)(x)
_Cos(x) = Cos([x, ], nothing, 0)(x)
_Tan(x) = Tan([x, ], nothing, 0)(x)
_Tanh(x) = Tanh([x, ], nothing, 0)(x)

forward(f::Log, x) = log(x)
forward(f::Sin, x) = sin(x)
forward(f::Cos, x) = cos(x)
forward(f::Tan, x) = tan(x)
forward(f::Tanh, x) = tanh(x)


function backward(f::Log)::AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    x = f.inputs[1]
    return [1 / x,] .* gys
end


function backward(f::Sin)::AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    x, = f.inputs
    return [cos(x),] .* gys
end

function backward(f::Cos)::AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    x, = f.inputs
    return [-sin(x),] .* gys
end


function backward(f::Tan)::AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    x, = f.inputs
    return [1 / cos(x)^2,] .* gys
end


function backward(f::Tanh)::AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    y = f.outputs[1]
    return [1 - y^2,] .* gys
end



