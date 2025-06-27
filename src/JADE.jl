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

    return sensorData(dataset.time, CM)

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
