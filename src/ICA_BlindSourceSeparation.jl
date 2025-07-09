module ICA_BlindSourceSeparation

using Plots: plot, plot!, display

using LinearAlgebra: eigen, I, sqrt, pinv, Diagonal, Symmetric
using Statistics: cov, mean, norm

include("SensorData.jl")
include("JnC_commons.jl")
include("Shibbs.jl")
include("Picard.jl")
include("JADE.jl")

function profile_test(x,n)
    for i = 1:n
        algo = Jade(2)
        y, _ = perform_separation(x, algo)
    end
end

export read_dataset, whiten_dataset, plot_dataset, demo, perform_separation, Jade, Picard, Shibbs, sensorData, profile_test

end