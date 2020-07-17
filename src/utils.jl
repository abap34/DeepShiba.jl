function numerical_diff(f::Function, x::Real; e=10e-4)
    return (f(x + e) - f(x - e)) / 2e
end

function numerical_diff(f::Function, xs::AbstractArray; e=1e-4)
    grads = zeros(length(xs)) 
    for idx in 1:length(xs)
        tmp_val = xs[idx]
        xs[idx] = tmp_val + e
        fxh1 = f(xs...)
        xs[idx] = tmp_val - e
        fxh2 = f(xs...)
        grads[idx] = (fxh1 - fxh2) / 2e
        xs[idx] = tmp_val
    end
    return grads
end

function get_output_str(var::Variable)
    output = "{DeepShiba.ShibaObject.Variable}:\n"
    if var.name !== ""
        output *= "$(var.name) :\n"
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


function Base.show(io::IO, ::MIME"text/plain", var::Variable) 
    print(get_output_str(var))
end


function Base.show(io::IO, var::Variable)
    print(var.data)
end



function Base.show(io::IO, f::Func)
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
    name = var.name == "" ? "" : var.name * ":"
    if var.data !== nothing
        var_size = size(var.data)
        if isempty(var_size)
            try var.data !== nothing
                name *= "$(var.data)"
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
    f_type = typeof(f)
    if f_type == DeepShiba.Pow
        f_type = "$(f_type)($(f.c))"
    end
    txt = "$(objectid(f)) [label=\"$(f_type)\", color=lightblue, style=filled, shape=box]\n"
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


