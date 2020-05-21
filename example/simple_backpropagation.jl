include("../src/core_simple.jl")
square_forward(x) = x^2
square_backward(x) = 2 * x


exp_forward(x) = exp(x)
exp_backward(x) = exp(x)

a = Square = Func(
    square_forward,
    square_backward, 
    nothing,
    nothing
)
b = Func(
    exp_forward,
    exp_backward,
    nothing,
    nothing
)
c = Func(
    square_forward,
    square_backward, 
    nothing,
    nothing
)


x = Variable(
    0.5,
    nothing,
    nothing
)


A = a(x)
B = b(A)
C = c(B)
C.grad = 1
backward!(C)
println(x.grad)