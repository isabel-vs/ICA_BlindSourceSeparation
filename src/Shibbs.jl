#=------------------------------------------------------------
SHIBBS algorithm.
Implements blind source separation by cumulant joint diagonalization.
Author: Erik Felgendreher
Adapted from Jean-François Cardoso's MATLAB version
=------------------------------------------------------------=#

"""
    estimate_cumulants(X::Matrix{Float64}) -> Matrix{Float64}

    Returns cumulant matrix.
"""
function estimate_cumulants(X::Matrix{Float64})::Matrix{Float64}
    m, T = size(X)
    nbcm = m
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
    joint_diagonalize(CM::Matrix{Float64}; threshold=1e-6, max_iterations=Inf) -> Matrix{Float64}
    Joint diagonalization using Givens rotation
    threshold: target size for the rotation size
    max_iterations: if threshold is not undercut, finish anyway after max iterations
    Returns diagonalization matrix
"""
function joint_diagonalize(CM::Matrix{Float64}; threshold=1e-6, max_iterations=Inf)::Matrix{Float64}
    m, _ = size(CM)
    V = Matrix{Float64}(I, m, m)
    nbcm = div(size(CM, 2), m)
    sweep = 0

    while true
        sweep += 1
        rotations = 0

        for p in 1:m-1
            for q in p+1:m
                Ip = collect(p:m:m * nbcm)
                Iq = collect(q:m:m * nbcm)

                g1 = CM[p, Ip] .- CM[q, Iq]
                g2 = CM[p, Iq] .+ CM[q, Ip]
                g = vcat(g1, g2)

                gg = g * g'
                ton = gg[1, 1] - gg[2, 2]
                toff = gg[1, 2] + gg[2, 1]

                theta = 0.5 * atan(toff / (ton + sqrt(ton^2 + toff^2)))

                if abs(theta) > threshold
                    c = cos(theta)
                    s = sin(theta)
                    G = [c -s; s c]
                    pair = [p, q]
                    V[:, pair] .= V[:, pair] * G

                    CM_pair_p = CM[p, :]
                    CM_pair_q = CM[q, :]

                    CM[p, :] .= c .* CM_pair_p .+ s .* CM_pair_q
                    CM[q, :] .= -s .* CM_pair_p .+ c .* CM_pair_q

                    for k in 1:nbcm
                        Ip_k = (k - 1) * m + p
                        Iq_k = (k - 1) * m + q
                        tmp_p = CM[:, Ip_k]
                        tmp_q = CM[:, Iq_k]

                        CM[:, Ip_k] .= c .* tmp_p .+ s .* tmp_q
                        CM[:, Iq_k] .= -s .* tmp_p .+ c .* tmp_q
                    end

                    rotations += 1
                end
            end
        end

        rot_size = norm(V - I(m), Inf)
        if (rot_size < m * threshold || sweep >= max_iterations)
            break
        end
    end

    return V
end
"""
    sort_by_energy(B::Matrix{Float64}) -> Matrix{Float64}
    Sort rows of B to put most energetic sources first
    Returns sorted matrix
"""
function sort_by_energy(B::Matrix{Float64})::Matrix{Float64}
    A = pinv(B)
    energies = sum(abs2, A; dims=1)
    sorted_indices = sortperm(vec(energies), rev=true)
    B_sorted = B[sorted_indices, :]

    # Flip sign to make first column positive
    signs = sign.(B_sorted[:, 1] .+ 0.1)
    B_flipped = Diagonal(signs) * B_sorted
    return B_flipped
end

# Main SHIBBS function
function shibbs_core(X::Matrix{Float64}, m::Int = size(X, 1))
    n, T = size(X)
    if (m > n)
        throw("Cannot extract more sources than sensors.")
    end

    # Estimate cumulants and perform joint diagonalization
    CM = estimate_cumulants(X)
    V = joint_diagonalize(CM; threshold=0.01 / sqrt(T), max_iterations=Inf)

    # Final separation matrix
    B_final = Matrix(V') #* B
    B_final = sort_by_energy(B_final)
    return B_final
end

function ica_shibbs(dataset::sensorData)::sensorData
    # Preprocessing
    d = whiten_dataset(dataset)
    
    X = Matrix(d.data')
    B = shibbs_core(X)
    
    S=B*X
    S = Matrix(S')
    return sensorData(dataset.time, S)
end