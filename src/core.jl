import Base
using DataStructures


abstract type ShibaObject end

mutable struct Variable <: ShibaObject 
    data
    creator
    grad
    generation
    name
end


variable(data, name=nothing) = Variable(data, nothing, nothing, 0, name)

abstract type Func <: ShibaObject end 

mutable struct Add <: Func
    inputs
    outputs 
    generation 
end

mutable struct Sub <: Func
    inputs
    outputs
    generation
end


mutable struct Neg <: Func
    inputs
    outputs
    generation
end


mutable struct Mul <: Func
    inputs
    outputs
    generation
end


mutable struct Div <: Func
    inputs
    outputs
    generation
end


mutable struct Pow <: Func
    inputs
    outputs
    generation
    c
end

_Add(x1, x2) =  Add(nothing, nothing, nothing)(x1, x2)

_Sub(x1, x2) = Sub(nothing, nothing, nothing)(x1, x2)

_Neg(x) = Neg(nothing, nothing, nothing)(x)

_Mul(x1, x2) = Mul(nothing, nothing, nothing)(x1, x2)

_Div(x1, x2) = Div(nothing, nothing, nothing)(x1, x2)

_Pow(x, c) = Pow(nothing, nothing, nothing, c)(x, c)


forward(f::Add, x1, x2) = x1 + x2
forward(f::Sub, x1, x2) = x1 - x2
forward(f::Neg, x) = -x
forward(f::Mul, x1, x2) = x1 * x2
forward(f::Div, x1, x2) = x1 / x2
forward(f::Pow, x, c) = x^c

function get_gys(outputs)
    return [output.grad for output in outputs]
end

function backward(f::Add) 
    gys = get_gys(f.outputs)
    return gys .* [1, 1]
end

function backward(f::Sub) 
    gys = get_gys(f.outputs)
    return gys .* [1, -1]
end

function backward(f::Neg) 
    gys = get_gys(f.outputs)
    return (- f.inputs .* gys,)
end

function backward(f::Mul) 
    gys = get_gys(f.outputs)
    x1, x2 = f.inputs
    return (x2, x1) .* gys
end

function backward(f::Div) 
    gys = get_gys(f.outputs)
    x1, x2 = f.inputs
    return (1 / x1, -x1 / (x2^2),) .* gys
end

function backward(f::Pow)
    gys = get_gys(f.outputs)
    x1, x2 = f.inputs
    return (x2 * (x1^(x2 - 1)),) .* gys
end

function ones_like(x)
    shape = size(x)
    isempty(shape) ? 1 : ones(shape)
end


function as_tuple(x)
    typeof(x) <: Tuple ? x : tuple(x)
end

function as_variable(x)
    typeof(x) == Variable ? x : variable(x)
end

get_data(var::Variable) = var.data

get_generation(var::Variable) = var.generation

function cleargrad!(var::Variable)
    var.grad = nothing
end

function set_name(var::Variable, name)
    var.name = name
end


function (f::Func)(vars::Variable...)
    f.inputs = [vars...]
    xs = [x.data for x in vars]
    ys = forward(f, xs...)
    ys = as_tuple(ys)
    f.generation = minimum([x.generation for x in f.inputs])
    outputs = [Variable(y, f, nothing, f.generation - 1, nothing) for y in ys]  
    #= Why is the generation decimating?
        When using the Priority queue, the values are taken in order of decreasing priority.
        The larger the number of generations, the more we want to process it first, so the value is decremented. =#
    f.outputs = outputs
    return length(outputs)  == 1 ? outputs[1] : outputs
end


function backward!(var::Variable)
    (isnothing(var.grad)) && (var.grad = variable(ones_like(var.data)))
    funcs = PriorityQueue{Func,Int}()
    seen_set = Set{Func}()
    enqueue!(funcs, var.creator, 1)
    push!(seen_set, var.creator)
    while !(isempty(funcs))
        f = dequeue!(funcs)
        gxs = backward(f)
        for (x, gx) in zip(f.inputs, gxs)
            if x.grad === nothing
                x.grad = gx
            else
                x.grad = x.grad + gx
            end
            if x.creator !== nothing && (!(x.creator in seen_set))
                push!(seen_set, x.creator)
                enqueue!(funcs, x.creator, x.creator.generation)
            end
        end
    end
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

Base.:^(x1::Variable, x2::Variable) = _Pow(x1, x2)
Base.:^(x1::Variable, x2) = _Pow(x1, variable(x2))
Base.:^(x1, x2::Variable) = _Pow(variable(x1), x2)

Base.size(x::Variable) = size(x.data)
