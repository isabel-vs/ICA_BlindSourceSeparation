module ICA_BlindSourceSeparation

using Plots: plot, display

using LinearAlgebra: eigen
using Statistics: cov, mean, Diagonal

include("SensorData.jl")

# define algorithm structs 
#TODO: Separate them into their respective file
struct Jade
    nSensors::Integer
end
struct Picard
end
struct Shibbs
    nSensors::Integer
end


"""
    whiten_dataset(X::sensorData) -> sensorData

    Applies PCA whitening to TxN data matrix
    T: number of samples
    N: number of sensors
    Returns the whitened dataset.
"""
function whiten_dataset(dataset::sensorData)
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
    whiten_dataset(X::sensorData, m::Int64) -> sensorData, W::Matrix{Float64}, iW::Matrix{Float64}

    Applies PCA whitening to TxN data matrix to decorrelate m sources
    T: number of samples
    n: number of sensors
    m: number of sources
    Returns the whitened dataset (Txm data matrix), whitening matrix W (mxn), pseudo-inverse whitening matrix iW (nxm)
"""
function whiten_dataset(dataset::sensorData, m::Int64)
    
    n_rows, n_cols = size(dataset.data)
    if (length(dataset.time) != n_rows)
        throw("Mismatch between time length and signal lengths")
    end
    if (n_cols == 0)
        throw("Matrix must have at least two column.")
    end
    
    # Copy to avoid modifying the original matrix
    X = dataset.data
    T = size(X,1)
    n = size(X,2)

    if (m>n)
        throw("More sources than sensors")
    end

    # center matrix
    μ = mean(X, dims=1)
    X_centered = X .- μ

    # calculate sample covariance matrix (NxN)
    Σ = cov(X_centered, dims=1, corrected=false)

    # eigendecomposition of Σ
    eigenvals, U = eigen(Σ)

    # select m largest eigenvalues
    rangeW = n-m+1:n
    scales = sqrt.(eigenvals[rangeW])
        
    # calculate whitening matrix
    W = Diagonal(1 ./scales) * U[:, rangeW]'
    iW = U[:, rangeW] * Diagonal(scales)

    data_white = X_centered * W'

    return sensorData(dataset.time, data_white), W, iW
end

"""
    read_dataset(filename::String) -> sensorData

    Reads a file containing numbers separated by spaces or tabs.
    Number of columns is detected by analyzing the first valid line.
    Returns an instance of sensorData.
"""
function read_dataset(filename::String)
    data = Float64[]
    ncols = 0
    nrows = 0

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

#TODO: add doc string
function demo()
    plot_dataset(whiten_dataset(read_dataset("data/foetal_ecg.dat")))
end

include("Shibbs.jl")
include("Picard.jl")
include("JADE.jl")

perform_separation(dataset, algo::Jade) = ica_jade(dataset, algo.nSensors)
perform_separation(dataset, algo::Picard) = ica_picard(dataset)
perform_separation(dataset, algo::Shibbs) = ica_shibbs(dataset, algo.nSensors)
perform_separation(dataset, algo) = error("$algo is not a valid algorithm")

export read_dataset, whiten_dataset, plot_dataset, demo, perform_separation, ALGORITHM

end