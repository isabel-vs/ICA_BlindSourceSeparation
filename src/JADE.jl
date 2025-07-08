#=------------------------------------------------------------
JADE algorithm.
Implements blind source separation by Joint Approximation Diagonalization of Eigen-matrices.
Author: Isabel von Stebut
Adapted from Jean-François Cardoso's MATLAB version
=------------------------------------------------------------=#
"""
    ica_jade(dataset::sensorData, m::Integer)

Separates m mixed sources with JADE algorithm.

Whitens data, estimates cumulants and performs joint diagonalization.

See also [`whiten_dataset`](@ref), [`estimate_cumulants`](@ref), [`joint_diagonalize`](@ref)
"""
function ica_jade(dataset::sensorData, m::Integer)

    # whitening & projection onto signal subspace
    d_white, W, iW = whiten_dataset(dataset, m)

    # estimation of cumulant matrices
    CM = estimate_cumulants(d_white, m)

    # joint diagonalization of cumulant matrices
    T = size(d_white.data, 1)
    #V = joint_diag(T, m, CM)
    V, _ = joint_diagonalize(CM, 0.01 / sqrt(T), (2^63)-1)

    # source estimation
    X_white = d_white.data
    S = X_white * V

    #TODO: order according to "most energetically significant" (as in matlab code)

    return sensorData(dataset.time, S)
end

"""
    estimate_cumulants(dataset::sensorData, m::Integer)

Estimates cumulant matrices.
"""
function estimate_cumulants(dataset::sensorData, m::Integer)
    X = (dataset.data)'
    T = size(X,2)

    num_cumm = (m*(m+1))÷2      # Dimension of space of real symm. matrices
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

struct Jade
    nSensors::Integer
end

perform_separation(dataset, algo::Jade) = ica_jade(dataset, algo.nSensors)