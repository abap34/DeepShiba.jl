module DeepShiba
    logo = raw"
 ____  _____ _____ ____  ____  _   _ ___ ____    _    
|  _ \| ____| ____|  _ \/ ___|| | | |_ _| __ )  / \    |  Documentation: https://xxxxxxxxxx
| | | |  _| |  _| | |_) \___ \| |_| || ||  _ \ / _ \   |  
| |_| | |___| |___|  __/ ___) |  _  || || |_) / ___ \  |  Version: v0.1 (yyyy-mm-dd)
|____/|_____|_____|_|   |____/|_| |_|___|____/_/   \_\ |
"
    
    (isinteractive()) && (println(logo))                                                  
    include("core/core_structures.jl")
    include("core/core_functions.jl")
    include("utils/output_txt.jl")
    include("utils/plot.jl")
    include("functions/math_functions.jl")
    include("functions/trans_functions.jl")
    export Variable, Func, variable, @variable, backward!, numerical_diff, cleargrad!, show, +, -, *, /, reshape, sin, cos, tanh, plot
end 


