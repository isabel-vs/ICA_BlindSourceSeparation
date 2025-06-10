module ICA_BlindSourceSeparation

# Write your package code here.
using Plots
using LinearAlgebra
using Statistics

include("SensorData.jl")
include("Picard.jl")
include("JADE.jl")
include("Shibbs.jl")

@enum ALGORITHM begin
    jade = 1
    picard = 2
    shibbs = 3
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
    if (length(dataset.time) != n_rows)
        throw("Mismatch between time length and signal lengths")
    end
    if (n_cols == 0)
        throw("Matrix must have at least two column.")
    end

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
                if (ncols <= 1)
                    throw("No valid data found in the file")
                end
            else
                if (length(values) != ncols)
                    throw("Inconsistent number of columns on line $(nrows + 1)")
                end
            end

            append!(data, parse.(Float64, values))
            nrows += 1
        end
    end

    if (nrows == 0)
        throw("The file contains no valid data rows")
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
    if (length(dataset.time) != size(dataset.data, 1)) 
        throw("Mismatch between time length and signal lengths!")
    end
    if (size(dataset.data, 2) == 0)
        throw("Data must have at least one column!")
    end

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

function perform_separation(dataset::sensorData, algo::ALGORITHM)::sensorData 
    if (algo == jade)
        return ica_jade(dataset)
    end 
    if (algo == picard)
        return ica_picard(dataset)
    end
    if (algo == shibbs)
        return ica_shibbs(dataset)
    end
end

export read_dataset, whiten_dataset, plot_dataset, demo, perform_separation

end