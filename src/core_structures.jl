import Base
using DataStructures


abstract type ShibaObject end
abstract type Func <: ShibaObject end 


const NullableRealValueType = Union{Real,AbstractArray{<:Real},Nothing}
const NullableFuncType = Union{Func,Nothing}

mutable struct Variable <: ShibaObject 
    data::NullableRealValueType
    creator::NullableFuncType
    grad::Union{Variable,AbstractArray{Variable},Nothing}
    generation::Int
    name::String
end

const NullableVariableArray = Union{AbstractArray{Variable},Nothing}

mutable struct Add <: Func
    inputs::NullableVariableArray
    outputs::NullableVariableArray
    generation::Int
end


mutable struct Sub <: Func
    inputs::NullableVariableArray
    outputs::NullableVariableArray
    generation::Int
end


mutable struct Neg <: Func
    inputs::NullableVariableArray
    outputs::NullableVariableArray
    generation::Int
end


mutable struct Mul <: Func
    inputs::NullableVariableArray
    outputs::NullableVariableArray
    generation::Int
end


mutable struct Div <: Func
    inputs::NullableVariableArray
    outputs::NullableVariableArray
    generation::Int
end


mutable struct Pow <: Func
    inputs::NullableVariableArray
    outputs::NullableVariableArray
    generation::Int
    c::NullableRealValueType
end




_Add(x1, x2) =  forward!(Add([x1, x2], nothing, 0))

_Sub(x1, x2) = forward!(Sub([x1, x2], nothing, 0))

_Neg(x) = forward!(Neg([x, ], nothing, 0))

_Mul(x1, x2) = forward!(Mul([x1, x2], nothing, 0))

_Div(x1, x2) = forward!(Div([x1, x2], nothing, 0))

_Pow(x, c) = forward!(Pow([x, ], nothing, 0, c))


@inline forward(f::Add, x1, x2)  = x1 + x2
@inline forward(f::Sub, x1, x2) = x1 - x2
@inline forward(f::Neg, x) = -x
@inline forward(f::Mul, x1, x2) = x1 * x2
@inline forward(f::Div, x1, x2) = x1 / x2
@inline forward(f::Pow, x) = x^(f.c) 

function backward(f::Add)::AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    return [1, 1] .* gys
end

function backward(f::Sub)::AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    return [1, -1] .* gys
end

function backward(f::Neg)::AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    return -gys
end

function backward(f::Mul)::AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    x1, x2 = f.inputs
    return [x2, x1] .* gys
end

function backward(f::Div)::AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    x1, x2 = f.inputs
    return [1 / x2, -x1 / (x2^2)] .* gys
end

function backward(f::Pow)::AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    x, c = f.inputs[1], f.c    
    return [c * (x^(c - 1)),] .* gys
end



Base.:+(x1::Variable, x2::Variable) = _Add(x1, x2)
Base.:+(x1::Variable, x2) = _Add(x1, variable(x2))
Base.:+(x1, x2::Variable) = _Add(variable(x1), x2)


Base.:-(x1::Variable, x2::Variable) = _Sub(x1, x2)
Base.:-(x1::Variable, x2) = _Sub(x1, variable(x2))
Base.:-(x1, x2::Variable) = _Sub(variable(x1), x2)

Base.:*(x1::Variable, x2::Variable) = _Mul(x1, x2)
Base.:*(x1::Variable, x2) = _Mul(x1, variable(x2))
Base.:*(x1, x2::Variable) = _Mul(variable(x1), x2)

Base.:/(x1::Variable, x2::Variable) = _Div(x1, x2)
Base.:/(x1::Variable, x2) = _Div(x1, variable(x2))
Base.:/(x1, x2::Variable) = _Div(variable(x1), x2)

Base.:-(x::Variable) = _Neg(x)

Base.:^(x1::Variable, x2::Variable) = _Pow(x1, x2.data)
Base.:^(x1::Variable, x2) = _Pow(x1, x2)
Base.:^(x1, x2::Variable) = _Pow(variable(x1), x2.data)

Base.log(x::Variable) = _Log(x)
Base.size(x::Variable) = size(x.data)
Base.iterate(x::Variable) = iterate(x.data)

