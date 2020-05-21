mutable struct Variable
    data
    creator
    grad
end

mutable struct Func
    forward
    backward
    input
    output
end


function (f::Func)(var::Variable)
    f.input = var 
    y = f.forward(var.data)
    output = Variable(y, f, nothing)
    f.output = output
    return  output
end



function backward!(var::Variable)
    funcs = [var.creator]
    while !(isempty(funcs))
        f = pop!(funcs)
        x, y = f.input, f.output
        x.grad = f.backward(x.data) * y.grad
        (!(isnothing(x.creator))) && (push!(funcs, x.creator))
    end
end

