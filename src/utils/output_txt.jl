function get_output_str(var::Variable)
    output = "{DeepShiba.ShibaObject.Variable}:\n"
    if var.name != ""
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


function Base.show(io::IO, var::Variable)
    print(io, var.data)
end

function Base.show(io::IO, ::MIME"text/plain", var::Variable) 
    print(io, get_output_str(var))
end


function get_value_type(nested_var)
    if eltype(nested_var) <: Real
        return eltype(nested_var)
    else
        get_value_type(eltype(nested_var))
    end
end




