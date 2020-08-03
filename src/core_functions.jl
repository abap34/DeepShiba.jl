function (f::Func)(vars::Variable...)
    xs = Array{Float64}(undef, length(vars))
    xs[1] = vars[1].data
    min_generation = vars[1].generation
    for i in 2:length(vars)
        xs[i] = vars[i].data
        if vars[i].generation < min_generation
            min_generation = vars[i].generation
        end
    end
    ys = forward(f, xs...)
    f.generation = min_generation
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


function variable(data, creator=nothing, grad=nothing, generation=0; name="") 
    return Variable(data, creator, grad, generation, name)
end

function get_gys(outputs)
    return [output.grad for output in outputs]
end


function get_data(var::Variable)
    return var.data
end

function get_creator(var::Variable)
    return var.creator
end


function get_grad(var::Variable)
    return var.grad
end

function get_generation(var::Variable)
    return var.generation
end

function get_outputs(f::Func)
    return f.outputs
end



function ones_like(x)
    shape = size(x)
    return isempty(shape) ? 1 : ones(shape)
end

function as_tuple(x)
    typeof(x) <: Tuple ? x : tuple(x)
end

function as_variable(x)
    typeof(x) == Variable ? x : variable(x)
end

function cleargrad!(var::Variable)
    var.grad = nothing
end

function set_name(var::Variable, name)
    var.name = name
end
