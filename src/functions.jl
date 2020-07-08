import Base


# ----MathFunc----


mutable struct Log <: Func
    inputs :: NullableVariableValueType
    outputs :: NullableVariableValueType
    generation :: Int
end

mutable struct Sin <: Func
    inputs :: NullableVariableValueType
    outputs :: NullableVariableValueType
    generation :: Int
end

mutable struct Cos <: Func
    inputs :: NullableVariableValueType
    outputs :: NullableVariableValueType
    generation :: Int
end

mutable struct Tan <: Func
    inputs :: NullableVariableValueType
    outputs :: NullableVariableValueType
    generation :: Int
end

mutable struct Tanh <: Func
    inputs :: NullableVariableValueType
    outputs :: NullableVariableValueType
    generation :: Int
end


_Log(x::Variable) = Log(nothing, nothing, 0)(x)
_Sin(x) = Sin(nothing, nothing, 0)(x)
_Cos(x) = Cos(nothing, nothing, 0)(x)
_Tan(x) = Tan(nothing, nothing, 0)(x)
_Tanh(x) = Tanh(nothing, nothing, 0)(x)

forward(f::Log, x) = log(x)
forward(f::Sin, x) = sin(x)
forward(f::Cos, x) = cos(x)
forward(f::Tan, x) = tan(x)
forward(f::Tanh, x) = tanh(x)


function backward(f::Log) :: AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    x = f.inputs[1]
    return [1 / x,] .* gys
end


function backward(f::Sin) :: AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    x, = f.inputs
    return [cos(x),] .* gys
end

function backward(f::Cos) :: AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    x, = f.inputs
    return [-sin(x),] .* gys
end


function backward(f::Tan) :: AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    x, = f.inputs
    return [1 / cos(x)^2,] .* gys
end


function backward(f::Tanh) :: AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    y = f.outputs[1]
    return [1 - y ^ 2,] .* gys
end



# ---- TransFunc -----

mutable struct Reshape <: Func
    inputs
    outputs
    generation
end

_Reshape(x, shape) = Reshape(nothing, nothing, nothing)(x, shape)

forward(f::Reshape, x, shape)  = reshape(x, shape)

function backward(f::Reshape)
    gys = f.outputs[1].grad
    return (reshape(gys, size(f.inputs[1])),)
end



Base.reshape(x::Variable, shape::Variable) = _Reshape(x, shape)
Base.reshape(x::Variable, shape) = _Reshape(x, variable(shape))
Base.reshape(x, shape::Variable) = _Reshape(variable(x), shape)

Base.sin(x::Variable) = _Sin(x)
Base.cos(x::Variable) = _Cos(x)
Base.tan(x::Variable) = _Tan(x)
Base.tanh(x::Variable) = _Tanh(x)