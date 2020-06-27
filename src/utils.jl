include("core.jl")


function numerical_diff(f::Function, x::Real; e=10e-4)
    return (f(x + e) - f(x - e)) / 2e
end

function numerical_diff(f::Function, x::AbstractArray; e=1e-4)
    grad = zeros(length(x)) 
    for idx in 1:length(x)
        tmp_val = x[idx]
        x[idx] = tmp_val + e
        fxh1 = f(x)
        x[idx] = tmp_val - e
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / 2e
        x[idx] = tmp_val
    end
    return grad
end

function get_output_str(var::Variable)
    output = "{DeepShiba.ShibaObject.Variable}:\n"
    if var.name !== nothing
        output *= "$(var.name):\n"
    end
    output *= "data: $(var.data)\n"

    if (var.grad !== nothing) 
        output *= "grad: $(var.grad.data)\n"
    end

    if (var.creator !== nothing)
        output *= "creator: $(typeof(var.creator))\n"
    else
        output *= "User-Defined\n"
    end
    return output
end

function Base.print(var::Variable)
    println(get_output_str(var))
end

function Base.println(var::Variable)
    println(get_output_str(var))
end

function Base.display(var::Variable)
    println(get_output_str(var))
end




function Base.print(f::Func)
    println("Func[")
    print("input: ")
    print(f.inputs)
    print("output: ")
    print(f.outputs)
    print("]")
end

function Base.println(f::Func)
    println("Func[")
    print("input: ")
    print(f.inputs)
    print("output: ")
    print(f.outputs)
    print("]")
end

function Base.display(f::Func)
    println("Func[")
    print("input: ")
    print(f.inputs)
    print("output: ")
    print(f.outputs)
    print("]")
end

function get_value_type(nest_var)
    if eltype(nest_var) <: Real
        return eltype(nest_var)
    else
        get_value_type(eltype(nest_var))
    end
end

function _dot_var(var::Variable)
    name = var.name === nothing ? "" : var.name * ":"
    if var.data !== nothing
        var_size = size(var.data)
        if isempty(var_size)
            try var.grad.data !== nothing
                name *= "$(var.grad.data)"
            catch
                name *= "nothing"
            end
        else    
            name *= "shape: $(var_size) \n type: $(get_value_type(var.data))"
        end
    end
    dot_var = "$(objectid(var)) [label=\"$name\", color=orange, style=filled]\n"
end


function _dot_func(f::Func)
    txt = "$(objectid(f)) [label=\"$(typeof(f))\", color=lightblue, style=filled, shape=box]\n"
    for x in f.inputs
        txt *= "$(objectid(x)) -> $(objectid(f))\n"
    end
    for y in f.outputs
        txt *= "$(objectid(f)) -> $(objectid(y))\n"
    end
    return txt
end

function get_dot_graph(var)
    txt = ""
    funcs = []
    seen_set = Set{Func}()
    push!(funcs, var.creator)
    txt = _dot_var(var)
    while !(isempty(funcs))
        f = pop!(funcs)
        txt *= _dot_func(f)
        for x in f.inputs
            txt *= _dot_var(x)
            if x.creator !== nothing && (!(x.creator in seen_set))
                push!(seen_set, x.creator)
                push!(funcs, x.creator)
            end
        end
    end
    return "digraph g {
                $txt 
            }"
end



function plot(var::Variable; to_file = "graph.png")
    dot_graph = get_dot_graph(var)
    tmp_dir = join([expanduser("~"),".DeepShiba"], "/")
    (!(ispath(tmp_dir))) && (mkdir(tmp_dir))
    dot_file_path = join([tmp_dir,"tmp_graph.dot"], "/")
    open(dot_file_path, "w") do io
        write(io, dot_graph)
    end
    extension = split(to_file, ".")[2]
    to_file = join([pwd(), to_file], "/") 
    cmd = `dot $(dot_file_path) -T $(extension) -o $(to_file)`
    run(cmd)
end


