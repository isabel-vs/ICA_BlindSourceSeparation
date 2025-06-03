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
    whiten_dataset(X::sensorData) -> sensorData

    Applies PCA whitening to TxN data matrix
    T: number of samples
    N: number of sensors
    Returns the whitened dataset.
"""
function whiten_dataset(dataset::sensorData)::sensorData
    n_rows, n_cols = size(dataset.data)
    @assert n_cols > 0 "Matrix must have at least two column."

    # Copy to avoid modifying the original matrix
    X = deepcopy(dataset.data)
    T = size(X,1)

    # center matrix
    μ = mean(X, dims=1)
    X_centered = X .- μ

    # calculate sample covariance matrix (NxN)
    Σ = cov(X, dims=1, corrected=false)

    # eigendecomposition of Σ
    eigenvals, U = eigen(Σ)
    
    # calculate whitening matrix
    W = Diagonal(1 ./sqrt.(eigenvals)) * U'

    data_white = X_centered * W'

    return sensorData(dataset.time, data_white)
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

function test()
    sensor = read_dataset("data/foetal_ecg.dat")
    data = sensor.data
    data_white = whiten_matrix(data)

    (T,N) = size(data_white)     # T time-samples, N sensors

    # center matrix
    μ = mean(data_white, dims=1)
    X_centered = data_white .- μ

    # calculate sample covariance matrix (NxN)
    Sigma = (X_centered' * X_centered)/T
    
    return Sigma

end

export whiten_dataset, read_dataset, plot_matrix, demo, test

end