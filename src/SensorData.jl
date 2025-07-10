
struct sensorData{
    T_time <: AbstractVector{<:Real},
    T_data <: AbstractMatrix{<:Real}
}
    time::T_time  # Vector of size C containing timestamps
    data::T_data  # CxN Matrix containing data of N sensors corresponding to timestamps.

    function sensorData(time::T_time, data::T_data) where {
        T_time<:AbstractVector{<:Real},
        T_data<:AbstractMatrix{<:Real}
    } 
        Base.require_one_based_indexing(time)
        Base.require_one_based_indexing(data)
        new{T_time, T_data}(time, data)
    end
end

"""
    whiten_dataset(X::sensorData)

Applies PCA whitening to TxN data matrix.
T: number of samples
N: number of sensors
Returns the whitened dataset.
"""
function whiten_dataset(dataset::sensorData)

    n = size(dataset.data,2)

    data_white, _, _ = whiten_dataset(dataset, n)

    return data_white
end
"""
    whiten_dataset(X::sensorData, m::Int)

Applies PCA whitening to a dataset, reducing its dimensionality to m components.
Returns the whitened dataset (Txm), whitening matrix W (mxn), pseudo-inverse whitening matrix iW (nxm)
"""
function whiten_dataset(dataset::sensorData, m::Int)
    n_rows, n_cols = size(dataset.data)
    if (m>n_cols)
        throw(ArgumentError("More sources than sensors"))
    end
    if (length(dataset.time) != n_rows)
        throw(DimensionMismatch("Mismatch between time length and signal lengths"))
    end

    T_type = eltype(dataset.data)
    
    # Copy to avoid modifying the original matrix
    X = dataset.data
    T = size(X,1)
    n = size(X,2)

    # center matrix
    μ = mean(X, dims=1)
    X_centered = X .- μ

    # calculate sample covariance matrix (NxN)
    Σ = Matrix{T_type}(cov(X_centered, dims=1, corrected=false))

    # eigendecomposition of Σ
    eigenvals, U = eigen(Symmetric(Σ))

    # select m largest eigenvalues
    rangeW = n-m+1:n
    scales = T_type.(sqrt.(eigenvals[rangeW]))
        
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
                    throw(ArgumentError("No valid data found in the file"))
                end
            else
                if (length(values) != ncols)
                    throw(DimensionMismatch("Inconsistent number of columns on line $(nrows + 1)"))
                end
            end

            append!(data, parse.(Float64, values))
            nrows += 1
        end
    end

    if (nrows == 0)
        throw(ArgumentError("The file contains no valid data rows"))
    end

    data = reshape(data, ncols, nrows)'
    time = data[:, 1]
    data = data[:, 2:end]

    return sensorData(time, data)
end

"""
    plot_dataset(dataset::sensorData)
    
Plots each column of the dataset against the timestamp vector
"""
function plot_dataset(dataset::sensorData)
    if (length(dataset.time) != size(dataset.data, 1)) 
        throw(DimensionMismatch("Mismatch between time length and signal lengths!"))
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
    
    return plt
end

"""
    demo()

Compares Jade, Shibbs, Picard algorithms for separating ECG data.
"""
function demo()
    root = pkgdir(ICA_BlindSourceSeparation)
    path = joinpath(root, "data", "foetal_ecg.dat")
    x = read_dataset(path)
    y1, _ = perform_separation(x,Jade(2))
    y2, _ = perform_separation(x,Shibbs(2, 1000))
    y3, _ = perform_separation(x,Picard(2, 3, 200, 1e-6, 1e-2, 10, false))
    p1 = plot_dataset(y1)
    p2 = plot_dataset(y2)
    p3 = plot_dataset(y3)
    plt = plot(p1, p2, p3, layout = (3,1))
    display(plt)
end