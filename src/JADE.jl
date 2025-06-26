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
    d = whiten_dataset(dataset, m)

    # estimation of cumulant matrices
    #---------------


    # joint diagonalization of cumulant matrices
    #---------------

end
