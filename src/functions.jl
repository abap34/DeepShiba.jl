import Base
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

