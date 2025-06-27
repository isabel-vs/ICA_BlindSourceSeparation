#=------------------------------------------------------------
JADE algorithm.
Implements blind source separation by Joint Approximation Diagonalization of Eigen-matrices.
Author: Isabel von Stebut
Adapted from Jean-François Cardoso's MATLAB version
=------------------------------------------------------------=#
"""
    JADE Algorithm for ICA source separation
"""
function ica_jade(dataset::sensorData, m::Int64)::sensorData

    # whitening & projection onto signal subspace
    d_white, W, iW = whiten_dataset(dataset, m)

    # estimation of cumulant matrices
    CM = estimate_cumulants(d_white, m)

    # joint diagonalization of cumulant matrices
    T = size(d_white.data, 1)
    V = joint_diag(T, m, CM)

    return sensorData(dataset.time, V)

end

"""
    Estimation of cumulant matrices
"""
function estimate_cumulants(dataset::sensorData, m::Int64)
    X = (dataset.data)'
    T = size(X,2)

    println(size(X))
    println(T)
    println(m)

    dimsymm = (m*(m+1))÷2   # Dimension of space of real symm. matrices
    num_cumm = dimsymm      # TODO: skip??
    CM = zeros(m, m*num_cumm)   # storage for cumulant matrices
    R = I(m)
    scale = ones(m)/T
    
    col_idx = 1
    for i = 1:m
        current_range = col_idx:(col_idx+m-1)
        Xi = @view X[i,:]
        Qij = ((scale * (Xi.*Xi)') .* X) * X' - R - 2*R[:,i]*R[:,i]'
        CM[:, current_range] = Qij
        col_idx += m
        for j in 1:i-1
            current_range = col_idx:(col_idx+m-1)
            Xj = @view X[j,:]
            Qij = ((scale * (Xi.*Xj)') .* X) * X' - R[:,i]*R[:,j]' - R[:,j]*R[:,i]'
            CM[:,current_range] = sqrt(2)*Qij
            col_idx += m
        end
    end
    
    return CM

end


"""
Joint diagonalization of the cumulant matrices
"""
function joint_diag(T::Int64, m::Int64, CM_in::Matrix{Float64})
    CM = copy(CM_in)
    V = Matrix{Float64}(I, m, m)
    thresh = 1/(100 * sqrt(T))
    num_cumm = size(CM, 2) ÷ m

    repeat = true
    sweep = 0
    updates = 0
    while repeat
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
                    c = cos(theta)
                    s = sin(theta)
                    G = [c -s ; s c]

                    pair = [p , q]
                    V[:,pair] = V[:,pair]*G
                    CM[pair,:] = G' * CM[pair,:]
                    all_indices = vcat(Ip, Iq)
                    CM[:,all_indices] = [c*CM[:,Ip]+s*CM[:,Iq] -s*CM[:,Ip]+c*CM[:,Iq]]
                end
            end
        end
    end
    return V
end
