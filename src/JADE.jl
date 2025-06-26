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
    #---------------


    # joint diagonalization of cumulant matrices
    #---------------

end

"""
    Estimation of cumulant matrices
"""
function estimate_cumulants(dataset::sensorData, m::Int64)
    # TODO: understand theory
    X = (dataset.data)'
    T = size(X,1)
    n = size(X,2)

    dimsymm = (m*(m+1))/2   # Dimension of space of real symm. matrices
    num_cumm = dimsymm      # TODO: skip??
    CM = zeros(m, m*num_cumm)   # storage for cumulant matrices
    R = eye(m)
    scale = ones(m,1)/T

    Range = 1:m
    for i = 1:m
        Xi = X[i,:]
        Qij = ((scale * (Xi.*Xi)) .* X) * X' - R - 2*R[:,i]*R[:,i]'
        CM[:,Range] = Qij
        Range = Range + m
        for j in 1:i-1
            Xj = X[j,:]
            Qij = ((scale * (Xi.*Xj)) .* X) * X' - R[:,i]*R[:,j]' - R[:,j]*R[:,i]'
            CM[:,Range] = sqrt(2)*Qij
            Range = Range + m
        end
    end
    
    return CM

end
