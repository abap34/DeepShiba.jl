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