import Base


# ----MathFunc----

mutable struct Sin <: Func
    inputs
    outputs
    generation
end

mutable struct Cos <: Func
    inputs
    outputs
    generation
end

_Sin(x) = Sin(nothing, nothing, nothing)(x)
_Cos(x) = Cos(nothing, nothing, nothing)(x)

forward(f::Sin, x) = sin(x)
forward(f::Cos, x) = cos(x)


function backward(f::Cos)
    gys = get_gys(f.outputs)
    x, = f.inputs
    return (-sin(x),) .* gys
end

function backward(f::Sin)
    gys = get_gys(f.outputs)
    x, = f.inputs
    return (cos(x),) .* gys
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