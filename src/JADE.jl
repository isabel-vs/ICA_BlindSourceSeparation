#=------------------------------------------------------------
JADE algorithm.
Implements blind source separation by Joint Approximation Diagonalization of Eigen-matrices.
Author: Isabel von Stebut
Adapted from Jean-François Cardoso's MATLAB version
=------------------------------------------------------------=#
"""
    ica_jade(dataset::sensorData, m::Int)

Separates m mixed sources with JADE algorithm.
Returns separated sources and transformation matrix V.

See also [`whiten_dataset`](@ref), [`estimate_cumulants`](@ref), [`joint_diagonalize`](@ref)
"""
function ica_jade(dataset::sensorData, m::Int)

    # whitening & projection onto signal subspace
    d_white, W, iW = whiten_dataset(dataset, m)

    # estimation of cumulant matrices
    CM = estimate_cumulants(d_white.data')

    # joint diagonalization of cumulant matrices
    T = size(d_white.data, 1)
    V, _ = joint_diagonalize(CM, 0.01 / sqrt(T), (2^63)-1)

    # source estimation
    X_white = d_white.data
    S = X_white * V

    #TODO: order according to "most energetically significant" (as in matlab code)

    return sensorData(dataset.time, S), V
end

struct Jade
    nSensors::Int
end

perform_separation(dataset, algo::Jade) = ica_jade(dataset, algo.nSensors)