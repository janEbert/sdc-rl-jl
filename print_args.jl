#!/usr/bin/env bash
#=
exec julia --project -O0 "${BASH_SOURCE[0]}" "$@"
=#

import JLD2

function main(filename)
    JLD2.@load filename args
    if args isa AbstractDict || args isa AbstractArray
        arg_iter = args
    elseif args isa Tuple
        arg_iter = zip(args...)
    end
    for (arg, value) in arg_iter
        print_arg(arg, value)
    end
end

function print_arg(arg, value)
    # println(arg, ": ", value)
    println(arg, ": ", lpad(value, 30 - length(string(arg))))
end

if abspath(PROGRAM_FILE) == @__FILE__
    @assert !isempty(ARGS) "no argument given"
    main(first(ARGS))
end
