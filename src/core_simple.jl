using DataStructures

mutable struct Variable
    data
    creator
    grad
    generation
    name
end

mutable struct Func
    forward
    backward
    inputs
    outputs
    generation
    name
end


variable(data, name=nothing) = Variable(data, nothing, nothing, 0, name)

func(forward, backward, name=nothing) = Func(forward, backward, nothing, nothing, 0, name)


function ones_like(x)
    shape = size(x)
    isempty(shape) ? 1 : ones(shape)
end


function as_tuple(x)
    typeof(x) <: Tuple ? x : tuple(x)
end


get_data(var::Variable) = var.data

function cleargrad(var::Variable)
    var.grad = nothing
end

function (f::Func)(vars::Variable...)
    f.inputs = vars
    xs = [x.data for x in vars]
    ys = f.forward(xs...)
    ys = as_tuple(ys)
    outputs = [Variable(y, f, nothing, f.generation + 1, nothing) for y in ys]
    f.generation = maximum([x.generation for x in f.inputs])
    f.outputs = outputs
    length(outputs)  == 1 ? outputs[1] : outputs
end


function backward!(var::Variable)
    (isnothing(var.grad)) && (var.grad = ones_like(var.data))
    funcs = PriorityQueue{Func, Int}()
    seen_set = Set{Func}()
    funcs[var.creator] = 1
    while !(isempty(funcs))
        f = dequeue!(funcs)
        gys = [output.grad for output in f.outputs]
        inputs = get_data.(f.inputs)
        gxs =  f.backward.(inputs) .* gys
        gxs = tuple(gxs...)
        for (x, gx) in zip(f.inputs, gxs)
            if isnothing(x.grad) 
                x.grad = gx
            else
                x.grad = x.grad + gx
            end
            if (!(isnothing(x.creator))) 
                if !(x.creator in seen_set)
                    funcs[x.creator] = x.generation
                end
            end
        end
    end
end
