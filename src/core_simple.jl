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

mutable struct Func <: ShibaObject
    forward
    backward
    inputs
    outputs
    generation
    name
end


variable(data, name = nothing) = Variable(data, nothing, nothing, 0, name)

func(forward, backward, name = nothing) = Func(forward, backward, nothing, nothing, 0, name)


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

function cleargrad(var::Variable)
    var.grad = nothing
end

function (f::Func)(vars::Variable...)
    f.inputs = vars
    xs = [x.data for x in vars]
    ys = f.forward(xs...)
    ys = as_tuple(ys)
    f.generation = maximum([x.generation for x in f.inputs])
    outputs = [Variable(y, f, nothing, f.generation - 1, nothing) for y in ys]  
    #= Why is the generation decimating?
        When using the Priority queue, the values are taken in order of decreasing priority.
        The larger the number of generations, the more we want to process it first, so the value is decremented.=#
    f.outputs = outputs
    length(outputs)  == 1 ? outputs[1] : outputs
end


function backward!(var::Variable, debug = false)
    (isnothing(var.grad)) && (var.grad = ones_like(var.data))
    funcs = PriorityQueue{Func,Int}()
    seen_set = Set{Func}()
    funcs[var.creator] = 1
    push!(seen_set, var.creator)
    while !(isempty(funcs))
        f = dequeue!(funcs)
        gys = [output.grad for output in f.outputs]
        inputs = get_data.(f.inputs)
        gxs =  f.backward(inputs...) .* gys
        gxs = tuple(gxs...)
        for (x, gx) in zip(f.inputs, gxs)
            if isnothing(x.grad) 
                x.grad = gx
            else
                x.grad = x.grad + gx
            end
            if (!(isnothing(x.creator))) && (!(x.creator in seen_set))
                push!(seen_set, x.creator)
                enqueue!(funcs, x.creator, x.creator.generation)
            end
        end
    end
end


Add(x1, x2) = func(
    (x1, x2) -> x1 + x2,
    (x1, x2) -> (1, 1)
)(x1, x2)

Sub(x1, x2) = func(
    (x1, x2) -> x1 - x2,
    x -> (1, -1)
)(x1, x2)

Neg(x) = func(
    x -> -x,
    x -> -x
)(x)

Mul(x1, x2) = func(
    (x1, x2) -> x1 * x2,
    x -> (x, x)
)(x1, x2)

Div(x1, x2) = func(
    (x1, x2) -> x1 / x2,
    (x1, x2) -> (1 / x1, -x1 / (x2 ^ 2))
)(x1, x2)

Pow(x1, x2) = func(
    (x1, x2) -> x1 ^ x2,
    (x1, x2) -> x2 * x1 ^  (x2 - 1)
)(x1, x2)

Base.:+(x1::Variable, x2::Variable) = Add(x1, x2)
Base.:+(x1::Variable, x2) = Add(x1, variable(x2))
Base.:+(x1, x2::Variable) = Add(variable(x1), x2)


Base.:-(x1::Variable, x2::Variable) = Sub(x1, x2)
Base.:-(x1::Variable, x2) = Sub(x1, variable(x2))
Base.:-(x1, x2::Variable) = Sub(variable(x1), x2)

Base.:*(x1::Variable, x2::Variable) = Mul(x1, x2)
Base.:*(x1::Variable, x2) = Mul(x1, variable(x2))
Base.:*(x1, x2::Variable) = Mul(variable(x1), x2)

Base.:/(x1::Variable, x2::Variable) = Div(x1, x2)
Base.:/(x1::Variable, x2) = Div(x1, variable(x2))
Base.:/(x1, x2::Variable) = Div(variable(x1), x2)

Base.:-(x::Variable) = Neg(x)

Base.:^(x1::Variable, x2::Variable) = Pow(x1, x2)
Base.:^(x1::Variable, x2) = Pow(x1, variable(x2))
Base.:^(x1, x2::Variable) = Pow(variable(x1), x2)