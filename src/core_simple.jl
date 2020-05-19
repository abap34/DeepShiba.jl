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


function (f::Func)(x::Variable)
    f.input = x 
    y = f.forward(x.data)
    output = Variable(y, f, nothing)
    f.output = output
    return  output
end



function backward!(var::Variable)
    f = var.creator
    if !(isnothing(f))
        x = f.input
        x.grad = f.backward(f.input.data) * var.grad
        backward!(x)
    end
end

