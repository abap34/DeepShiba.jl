const tmp_dir = join([expanduser("~"),".DeepShiba"], "/")
const dot_file_path = join([tmp_dir,"tmp_graph.dot"], "/")

struct PNGContainer
    content
end

function Base.show(io::IO, ::MIME"image/png", c::PNGContainer)
    write(io, c.content)
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


function plot_tmp_dir(extension)
    extension = "png"
    to_file = join([tmp_dir, "graph.png"], "/")       
    cmd = `dot $(dot_file_path) -T $(extension) -o $(to_file)`
    run(cmd)
    return to_file
end

function plot(var::Variable; to_file = "")
    dot_graph = get_dot_graph(var)
    (!(ispath(tmp_dir))) && (mkdir(tmp_dir))
    open(dot_file_path, "w") do io
        write(io, dot_graph)
    end
    
    if to_file == ""
        if split(PROGRAM_FILE, "/")[end] == "kernel.jl"
            png_file_path = plot_tmp_dir(".png")
            c = open(png_file_path) do io
                PNGContainer(read(io))
            end
            display(c)
        else
            plot_tmp_dir(".png")

        end
    else
        extension = split(to_file, ".")[2]
        cmd = `dot $(dot_file_path) -T $(extension) -o $(to_file)`
        run(cmd)
    end
end