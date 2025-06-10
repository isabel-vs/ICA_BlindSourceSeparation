mutable struct sensorData
    # Vector of size C containing timestamps
    time::Vector{Float64}
    # CxN Matrix containing data of N sensors corresponding to timestamps. 
    data::Matrix{Float64}
end