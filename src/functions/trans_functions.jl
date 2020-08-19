@DeepShiba_Func mutable struct Reshape <: Func
    shape::Tuple{Int}
end


@DeepShiba_Func mutable struct Transpose <: Func end

_Reshape(x, shape) = Reshape([x, ], nothing, nothing, shape)(x)
_Transpose(x) = Transpose([x, ], nothing, nothing)(x)


forward(f::Reshape, x)  = reshape(x, f.shape)
forward(f::Transpose, x) = transpose(x)

function backward(f::Reshape)::AbstractArray{Variable}
    gys = get_gys(f.outputs)
    x_shape = size(f.inputs[1])  
    return reshape.(gys, Ref(x_shape))
end

function backward(f::Transpose)::AbstractArray{Variable}
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