#=------------------------------------------------------------
SHIBBS algorithm.
Implements blind source separation by cumulant joint diagonalization.
Author: Erik Felgendreher
Adapted from Jean-François Cardoso's MATLAB version
=------------------------------------------------------------=#


"""
    estimate_cumulants(X::AbstractMatrix)

Returns cumulant matrix.
"""
function estimate_cumulants(X::AbstractMatrix)
    m, T = size(X)
    nbcm = m
    println("$m")
    CM = zeros(m, m * nbcm)
    R = I(m)

    for k in 1:m
        xk = X[k, :]
        Xk = X .* xk'
        Rk = Xk * Xk' / T - R
        Rk[k, k] -= 2
        CM[:, ((k - 1) * m + 1):(k * m)] .= Rk
    end

    return CM
end
"""
    joint_diagonalize(CM::AbstractMatrix, seuil::Float64)

Returns diagonalization matrix and rotation size
"""
function joint_diagonalize(CM::AbstractMatrix, threshold::Float64)
    m, _ = size(CM)
    nbcm = div(size(CM, 2), m)
    V = Matrix{Float64}(I, m, m)
    nbrs = 1
    max_iters = 5000
    nIter = 0 

    while nbrs != 0 && nIter < max_iters
        nIter += 1
        nbrs = 0
        for p in 1:m-1
            for q in p+1:m
                Ip = p:m:m*nbcm
                    Iq = q:m:m*nbcm

                    # Computation of Givens angle
                    g = [CM[p, Ip] - CM[q, Iq]; CM[p, Iq] + CM[q, Ip]]
                    gg = g * g'
                    ton = gg[1, 1] - gg[2, 2]
                    toff = gg[1, 2] + gg[2, 1]
                    theta = 0.5 * atan(toff, ton + sqrt(ton^2 + toff^2))

                    # Givens update
                    if abs(theta) > threshold
                        nbrs += 1
                        c = cos(theta)
                        s = sin(theta)
                        G = [c -s; s c]

                        pair = [p, q]
                        V[:, pair] = V[:, pair] * G
                        CM[pair, :] = G' * CM[pair, :]
                        CM[:, [Ip; Iq]] = [c * CM[:, Ip] + s * CM[:, Iq] -s * CM[:, Ip] + c * CM[:, Iq]]
                    end
            end
        end
    end

    rot_size = norm(V - I)
    return V, rot_size
end
"""
    sort_by_energy(B::AbstractMatrix)

Sort rows of B to put most energetic sources first
Returns sorted matrix
"""
function sort_by_energy(B::AbstractMatrix)
    A = pinv(B)
    energies = sum(abs2, A; dims=1)
    sorted_indices = sortperm(vec(energies), rev=true)
    B_sorted = B[sorted_indices, :]

    # Flip sign to make first column positive
    signs = sign.(B_sorted[:, 1] .+ 0.1)
    B_flipped = Diagonal(signs) * B_sorted
    return B_flipped
end

"""
    ica_shibbs(dataset::sensorData, m::Int64)

Outer loop of the shibbs algorithm. Whitens data and loops untile diagonalization result is within threshold.
Returns transformation matrix applicable to X
"""
function ica_shibbs(dataset::sensorData, m::Int64)
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
    maxSteps = 1
    while OneMoreStep && nSteps < maxSteps
        println("$nSteps")
        nSteps += 1
        # Estimate cumulants
        CM = estimate_cumulants(X)

        # Joint diagonalization
        V, rot_size = joint_diagonalize(CM, seuil)

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

    S = X' * V

    return sensorData(dataset.time, S)
end