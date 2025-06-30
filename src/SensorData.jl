struct sensorData{T<:Real}
    # Vector of size C containing timestamps
    time::Vector{T}
    # CxN Matrix containing data of N sensors corresponding to timestamps. 
    data::Matrix{T}
end