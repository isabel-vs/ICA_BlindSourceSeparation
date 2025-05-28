module ICA_BlindSourceSeparation

# Write your package code here.
using Plots;

"""
    read_dataset(filename::String) -> Matrix{Float64}

    Reads a file containing numbers separated by spaces or tabs.
    Number of columns is detected by analyzing the first valid line.
    Returns a matrix of Float64.
"""
function read_dataset(filename::String)::Matrix{Float64}
    data = Float64[]
    ncols = 0;
    nrows = 0;

    open(filename, "r") do io
        for line in eachline(io)
            line = strip(line)
            # Skip empty lines or comments
            if isempty(line) || startswith(line, "#")
                continue 
            end

            values = split(line)
            if ncols == 0
                ncols = length(values)
                @assert ncols > 0 "No valid data found in the file"
            else
                @assert length(values) == ncols "Inconsistent number of columns on line $(nrows + 1)"
            end

            append!(data, parse.(Float64, values))
            nrows += 1
        end
    end

    return reshape(data, ncols, nrows)'
end

"""
    plot_matrix(data::Matrix{Float64})
    
    Plots each column of the dataset (except the first) against the first column
"""
function plot_matrix(data::Matrix{Float64})
    @assert size(data, 2) >= 2 "Data must have at least two columns (time + signal)"

    time = data[:, 1]
    signals = data[:, 2:end]
    nsignals = size(signals, 2)

    spacing = 1.2 * maximum(abs.(signals))  
    plt = plot(title="Estimated Source Signals", xlabel="Time (s)", yticks=false, legend=false)

    for i in 1:nsignals
        offset = spacing * (nsignals - i)
        trace = signals[:, i] .+ offset
        plot!(plt, time, trace, label="Signal $i")
    end
    
    display(plt)
end

end
