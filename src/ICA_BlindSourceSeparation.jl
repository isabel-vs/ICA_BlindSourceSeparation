module ICA_BlindSourceSeparation

using Plots: plot, plot!, display

using LinearAlgebra: eigen, I, sqrt, pinv, Diagonal
using Statistics: cov, mean, norm

include("SensorData.jl")
include("JnC_commons.jl")
include("Shibbs.jl")
include("Picard.jl")
include("JADE.jl")

export read_dataset, whiten_dataset, plot_dataset, demo, perform_separation, Jade, Picard, Shibbs, sensorData

end