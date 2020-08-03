import Base


# ----MathFunc----


mutable struct Log <: Func
    inputs::NullableVariableArray
    outputs::NullableVariableArray
    generation::Int
end

mutable struct Sin <: Func
    inputs::NullableVariableArray
    outputs::NullableVariableArray
    generation::Int
end

mutable struct Cos <: Func
    inputs::NullableVariableArray
    outputs::NullableVariableArray
    generation::Int
end

mutable struct Tan <: Func
    inputs::NullableVariableArray
    outputs::NullableVariableArray
    generation::Int
end

mutable struct Tanh <: Func
    inputs::NullableVariableArray
    outputs::NullableVariableArray
    generation::Int
end


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



# ---- TransFunc -----

mutable struct Reshape <: Func
    inputs
    outputs
    generation
    shape
end



mutable struct Transpose <: Func
    inputs
    outputs
    generation
end

_Reshape(x, shape) = forward!(Reshape([x, ], nothing, nothing, shape))
_Transpose(x) = forward!(Transpose([x, ], nothing, nothing)(x))


forward(f::Reshape, x)  = reshape(x, f.shape)
forward(f::Transpose, x) = transpose(x)

function backward(f::Reshape)
    gys = get_gys(f.outputs)
    x_shape = size(f.inputs[1])  
    return reshape.(gys, Ref(x_shape))
end

function backward(f::Transpose)
    gys = get_gys(f.outputs)
    return transpose.(gys)
end




Base.sin(x::Variable) = _Sin(x)
Base.cos(x::Variable) = _Cos(x)
Base.tan(x::Variable) = _Tan(x)
Base.tanh(x::Variable) = _Tanh(x)

Base.reshape(x::Variable, shape::Variable) = _Reshape(x, shape.data)
Base.reshape(x::Variable, shape) = _Reshape(x, shape)
Base.reshape(x, shape::Variable) = _Reshape(variable(x), shape.data)
Base.transpose(x::Variable) = _Transpose(x)