import Base
using DataStructures


abstract type ShibaObject end
abstract type Func <: ShibaObject end 


NullableRealValueType = Union{Real,AbstractArray{<:Real}, Nothing}
NullableFuncType = Union{Func,Nothing}
NullableVariableValueType = Any


mutable struct Variable <: ShibaObject 
    data :: NullableRealValueType
    creator :: NullableFuncType
    grad :: NullableVariableValueType
    generation :: Int
    name :: String
end

NullableVariableValueType = Union{Variable,AbstractArray{Variable}, Nothing}

mutable struct Add <: Func
    inputs :: NullableVariableValueType
    outputs :: NullableVariableValueType
    generation :: Int
end


mutable struct Sub <: Func
    inputs :: NullableVariableValueType
    outputs :: NullableVariableValueType
    generation :: Int
end


mutable struct Neg <: Func
    inputs :: NullableVariableValueType
    outputs :: NullableVariableValueType
    generation :: Int
end


mutable struct Mul <: Func
    inputs :: NullableVariableValueType
    outputs :: NullableVariableValueType
    generation :: Int
end


mutable struct Div <: Func
    inputs :: NullableVariableValueType
    outputs :: NullableVariableValueType
    generation :: Int
end


mutable struct Pow <: Func
    inputs :: NullableVariableValueType
    outputs :: NullableVariableValueType
    generation :: Int
    c :: NullableRealValueType
end




_Add(x1, x2) =  Add(nothing, nothing, 0)(x1, x2)

_Sub(x1, x2) = Sub(nothing, nothing, 0)(x1, x2)

_Neg(x) = Neg(nothing, nothing, 0)(x)

_Mul(x1, x2) = Mul(nothing, nothing, 0)(x1, x2)

_Div(x1, x2) = Div(nothing, nothing, 0)(x1, x2)

_Pow(x, c) = Pow(nothing, nothing, 0, c)(x)



forward(f::Add, x1, x2)  = x1 + x2
forward(f::Sub, x1, x2) = x1 - x2
forward(f::Neg, x) = -x
forward(f::Mul, x1, x2) = x1 * x2
forward(f::Div, x1, x2) = x1 / x2
forward(f::Pow, x) = x^(f.c)

function backward(f::Add) :: AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    return gys .* [1, 1]
end

function backward(f::Sub) :: AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    return gys .* [1, -1]
end

function backward(f::Neg) :: AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    return -gys
end

function backward(f::Mul) :: AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    x1, x2 = f.inputs
    return [x2, x1] .* gys
end

function backward(f::Div) :: AbstractArray{Variable} 
    gys = get_gys(f.outputs)
    x1, x2 = f.inputs
    return [1 / x2, -x1 / (x2^2)] .* gys
end

function backward(f::Pow) :: AbstractArray{Variable} 
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



