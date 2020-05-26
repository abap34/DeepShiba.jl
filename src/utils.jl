include("core_simple.jl")

function Base.print(x::Variable, debug=true)
    println("Variable(")
    print("data:",x.data)
    if debug
        println()
        (x.grad !== nothing) && println("grad:",x.grad)
        if (x.creator !== nothing)
            (x.creator.name !== nothing) && (println("creator:", x.creator.name))
        end
    end
    println(")")
end

function Base.print(f::Func)
    println("Func[")
    print("input: ")
    println(f.inputs)
    print("output: ")
    println(f.outputs)
    println("]")
end


function numerical_diff(f::Function, x::Real, e=10e-4)
    return (f(x + e) - f(x - e)) / 2e
end