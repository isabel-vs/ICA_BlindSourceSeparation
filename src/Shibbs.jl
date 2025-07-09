#=------------------------------------------------------------
SHIBBS algorithm.
Implements blind source separation by cumulant joint diagonalization.
Author: Erik Felgendreher
Adapted from Jean-François Cardoso's MATLAB version
=------------------------------------------------------------=#

# !Moved to JnC commons!
# """
#     estimate_cumulants(X::AbstractMatrix)

# Returns cumulant matrix.
# """
# function estimate_cumulants(X::AbstractMatrix)
#     m, T = size(X)
#     nbcm = m
#     CM = zeros(m, m * nbcm)
#     R = I(m)

#     for k in 1:m
#         xk = X[k, :]
#         Xk = X .* xk'
#         Rk = Xk * Xk' / T - R
#         Rk[k, k] -= 2
#         CM[:, ((k - 1) * m + 1):(k * m)] .= Rk
#     end

#     return CM
# end
"""
    ica_shibbs(dataset::sensorData, m::Integer, maxSteps::Integer)

Outer loop of the shibbs algorithm. Whitens data and loops untile diagonalization result is within threshold.
Returns the dataset combined with transformation matrix as well as the transformation Matrix
"""
function ica_shibbs(dataset::sensorData, m::Integer, maxSteps::Integer)
    d_white, B, iW = whiten_dataset(dataset, m)
    X = Matrix(d_white.data')

    T = size(d_white.data, 1)
    n = size(d_white.data, 2)

    seuil = 0.01 / sqrt(T)

    if m > n
        error("shibbs -> Do not ask for more sources than sensors.")
    end

    V = zeros(2,2)
    
    # === Outer loop ===
    OneMoreStep = true
    nSteps = 0
    while OneMoreStep && nSteps < maxSteps
        # println("$nSteps")
        nSteps += 1
        # Estimate cumulants
        CM = estimate_cumulants(X)

        # Joint diagonalization
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
end

perform_separation(dataset, algo::Shibbs) = ica_shibbs(dataset, algo.nSensors, algo.maxSteps)