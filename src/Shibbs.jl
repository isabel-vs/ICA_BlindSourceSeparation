#=------------------------------------------------------------
SHIBBS algorithm.
Implements blind source separation by cumulant joint diagonalization.
Author: Erik Felgendreher
Adapted from Jean-François Cardoso's MATLAB version
=------------------------------------------------------------=#

"""
    ica_shibbs(dataset::sensorData, m::Integer, maxSteps::Integer, thresh::Real = 1)

Outer loop of the shibbs algorithm. Whitens data and loops untile diagonalization result is within threshold.
Returns the dataset combined with transformation matrix as well as the transformation Matrix

See also [`whiten_dataset`](@ref), [`estimate_cumulants`](@ref), [`joint_diagonalize`](@ref)
"""
function ica_shibbs(dataset::sensorData, m::Integer, maxSteps::Integer, thresh::Real = 1)
    d_white, B, iW = whiten_dataset(dataset, m)
    X = Matrix(d_white.data')

    T = size(d_white.data, 1)

    seuil = (0.01 / sqrt(T)) / thresh

    OneMoreStep = true
    nSteps = 0
    while OneMoreStep && nSteps < maxSteps
        nSteps += 1

        CM = estimate_cumulants(X)

        V, rot_size = joint_diagonalize(CM, seuil, 100000)

        # Update
        X = V' * X
        B = V * B

        # Check convergence
        OneMoreStep = rot_size >= (m * seuil)
    end

    # Sort components by energy
    B = sort_by_energy(B)

    # Fix signs: first column non-negative
    b = B[:, 1]
    signs = sign.(sign.(b) .+ 0.1)
    B = Diagonal(signs) * B

    S = B * Matrix(dataset.data')

    return sensorData(dataset.time, Matrix(S')), B
end

struct Shibbs
    nSensors::Integer
    maxSteps::Integer
    threshold::Real
end

perform_separation(dataset, algo::Shibbs) = ica_shibbs(dataset, algo.nSensors, algo.maxSteps, algo.threshold)