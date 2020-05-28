include("core_simple.jl")

function Base.print(x::Variable, debug = true)
    if x.name !== nothing
        println(x.name, "(")
    else
        println("(")
    end
    print("data:", x.data)
    if debug
        println()
        (x.grad !== nothing) && println("grad:", x.grad)
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


function numerical_diff(f::Function, x::Real, e = 10e-4)
    return (f(x + e) - f(x - e)) / 2e
end


function _dot_var(var::Variable)
    name = var.name === nothing ? "" : var.name
    if var.data !== nothing
        var_size = isempty(size(var.data)) ? "scalar" : size(var.data)
        if var_size == "scalar"
            name *= ": $(var.data)"
        end
        name *= ": $(var_size) $(eltype(var.data))"
    end
    dot_var = "$(objectid(var)) [label=\"($name)\", color=orange, style=filled]\n"
end


function _dot_func(f::Func)
    txt = "$(objectid(f)) [label=\"$(f.name)\", color=lightblue, style=filled, shape=box]\n"
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
    funcs = PriorityQueue{Func,Int}()
    seen_set = Set{Func}()
    enqueue!(funcs, var.creator, 1)
    txt = _dot_var(var)
    while !(isempty(funcs))
        f = dequeue!(funcs)
        txt *= _dot_func(f)
        for x in f.inputs
            txt *= _dot_var(x)
            if x.creator !== nothing && (!(x.creator in seen_set))
                push!(seen_set, x.creator)
                enqueue!(funcs, x.creator, x.creator.generation)
            end
        end
    end
    return "digraph g {
                $txt 
            }"
end



function plot(var::Variable,to_file = "graph.png")
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


function mkdir_hoge(s)
    s = "mkdir $s"
    run(`$(split(s))`, wait=false)
end