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


const nothing_var = Variable(nothing,nothing,nothing)

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

square_forward(x) = x ^ 2
square_backward(x) = 2 * x


exp_forward(x) = exp(x)
exp_backward(x) = exp(x)