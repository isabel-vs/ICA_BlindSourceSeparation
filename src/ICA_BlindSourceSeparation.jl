module ICA_BlindSourceSeparation

# Write your package code here.
using Plots
using LinearAlgebra
using Statistics

mutable struct sensorData
    # Vector of size C containing timestamps
    time::Vector{Float64}
    # CxN Matrix containing data of N sensors corresponding to timestamps. 
    data::Matrix{Float64}
end

"""
    whiten(X::Vector{Float64}) -> Vector{Float64}

    Checks wether vector is empty.
    Applies whitening function to vector.
    Returns a vector of Float64.
"""
function whiten(x::Vector{Float64})
    @assert !(isempty(x)) "Input vector must not be empty."

    n = size(x,1)

    # sample mean
    μ = mean(x)
    x_centered = x .- μ

    #calculate sample covariance matrix
    test = (x_centered * x_centered')

    Sigma =  1/(n-1) * sum(x_centered * x_centered')

    #create whitening matrix
    U, Lambda, _ = svd(Sigma)
    W = Diagonal(1/sqrt.(Lambda .+ 1e-5)) * U'

    #whiten data
    x_whitened = W * x_centered

    return x_whitened
end

function whiten(X::Matrix{Float64})
    
    [T,n] = size(X)

    # sample mean
    μ = mean(x)
    x_centered = x .- μ

"""
    whiten_dataset(X::sensorData) -> sensorData

    Checks wether dataset contains at least one column.
    Applies whitening function to every column.
    Returns the whitened dataset.
"""
function whiten_dataset(X::sensorData)::sensorData
    n_rows, n_cols = size(X.data)
    @assert n_cols > 0 "Matrix must have at least one column."

    # Copy to avoid modifying the original matrix
    X_whitened = deepcopy(X.data)

    # Whiten each column of the dataset
    for j in 1:n_cols
        X_whitened[:, j] = whiten(X_whitened[:, j])
    end

    return sensorData(X.time, X_whitened)
end

"""
    read_dataset(filename::String) -> sensorData

    Reads a file containing numbers separated by spaces or tabs.
    Number of columns is detected by analyzing the first valid line.
    Returns an instance of sensorData.
"""
function read_dataset(filename::String)::sensorData 
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

    data = reshape(data, ncols, nrows)'
    time = data[:, 1]
    data = data[:, 2:end]

    return sensorData(time, data)
end

"""
    plot_matrix(dataset::sensorData)
    
    Plots each column of the dataset against the timestamp vector
"""
function plot_dataset(dataset::sensorData)
    @assert size(dataset.data, 2) >= 1 "Data must have at least one column"

    time = dataset.time
    signals = dataset.data
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
    plot_dataset(whiten_dataset(read_dataset("data/foetal_ecg.dat")))
end

export whiten, whiten_dataset, read_dataset, plot_matrix, demo, test

end
