module ICA_BlindSourceSeparation

# Write your package code here.
using Plots
using LinearAlgebra
using Statistics

"""
    whiten(X::Vector{Float64}) -> Vector{Float64}

    Checks wether vector is empty.
    Applies whitening function to every column except the first (time).
    Returns a vector of Float64.
"""
function whiten(x::Vector{Float64})
    @assert !(isempty(x)) "Input vector must not be empty."

    # Mean center
    μ = mean(x)
    x_centered = x .- μ

    # Standard deviation (for unit variance)
    σ = std(x_centered)
    @assert !(σ ≈ 0) "Standard deviation is zero; cannot whiten constant vector."

    # Whitened vector: zero mean and unit variance
    x_whitened = x_centered ./ σ

    return x_whitened
end

"""
    whiten_dataset(X::Matrix{Float64}) -> Matrix{Float64}

    Checks wether dataset contains at least two columns.
    Applies whitening function to every column except the first (time).
    Returns a matrix of Float64.
"""
function whiten_dataset(X::Matrix{Float64})
    n_rows, n_cols = size(X)
    @assert n_cols ≥ 2 "Matrix must have at least two columns."

    # Copy to avoid modifying the original matrix
    X_whitened = copy(X)

    # Whiten each column except the first
    for j in 2:n_cols
        X_whitened[:, j] = whiten(X[:, j])
    end

    return X_whitened
end

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

function demo()
    plot_matrix(whiten_dataset(read_dataset("data/foetal_ecg.dat")))
end

end
