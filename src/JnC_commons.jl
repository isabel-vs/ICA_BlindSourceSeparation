"""
    estimate_cumulants(X::AbstractMatrix)

Returns cumulant matrix.
"""
function estimate_cumulants(X::AbstractMatrix)
    m, T = size(X)
    nbcm = m
    T_e = eltype(X)
    CM = zeros(T_e, m, m * nbcm)
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
    joint_diagonalize(CM_in::AbstractMatrix, thresh::Real, max_iters::Integer)

Returns diagonalization matrix and rotation size
"""
function joint_diagonalize(CM_in::AbstractMatrix, thresh::Real, max_iters::Integer)
    CM = copy(CM_in)
    T = eltype(CM_in)
    m, _ = size(CM)
    num_cumm = div(size(CM, 2), m)
    V = Matrix{T}(I, m, m)
    nIter = 0 

    repeat = true
    sweep = 0
    updates = 0
    while repeat && nIter < max_iters
        nIter += 1
        repeat = false
        sweep += 1

        for p = 1:m-1
            for q = p+1:m
                Ip = p:m:(m*num_cumm)
                Iq = q:m:(m*num_cumm)

                #compute givens angle
                row1 = (@view(CM[p,Ip]) - @view(CM[q,Iq]))'
                row2 = (@view(CM[p,Iq]) + @view(CM[q,Ip]))'
                g = [row1 ; row2]
                gg = g*g'
                ton = gg[1,1] - gg[2,2]
                toff = gg[1,2] + gg[2,1]
                theta = 0.5 * atan(toff, ton+sqrt(ton^2 + toff^2))

                # givens update
                if abs(theta) > thresh
                    repeat = true
                    updates += 1
                    c = cos(T(theta))
                    s = sin(T(theta))
                    G = T[c -s ; s c]

                    pair = [p , q]
                    V[:,pair] = V[:,pair]*G
                    CM[pair,:] = G' * CM[pair,:]
                    all_indices = vcat(Ip, Iq)
                    CM[:,all_indices] = [c*CM[:,Ip]+s*CM[:,Iq] -s*CM[:,Ip]+c*CM[:,Iq]]
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
    T = eltype(B)
    A = pinv(B)
    energies = sum(abs2, A; dims=1)
    sorted_indices = sortperm(vec(energies), rev=true)
    B_sorted = B[sorted_indices, :]

    # Flip sign to make first column positive
    signs = sign.(B_sorted[:, 1] .+ T(0.1))
    B_flipped = Diagonal(signs) * B_sorted
    return B_flipped
end