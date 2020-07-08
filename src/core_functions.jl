function (f::Func)(vars::Variable...)
    f.inputs = [vars...]
    xs = [x.data for x in vars]
    ys = forward(f, xs...)
    f.generation = minimum([x.generation for x in f.inputs])
    outputs = [Variable(y, f, nothing, f.generation - 1, "") for y in ys]   
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
            if (x.creator !== nothing) && (!(x.creator in seen_set))
                push!(seen_set, x.creator)
                enqueue!(funcs, x.creator, x.creator.generation)
            end
        end
    end
end


variable(data, name="") = Variable(data, nothing, nothing, 0, name)


function get_gys(outputs)
    return [output.grad for output in outputs]
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
