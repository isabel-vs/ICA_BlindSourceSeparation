using ICA_BlindSourceSeparation
using Test
using Statistics
using LinearAlgebra

@testset "ICA_BlindSourceSeparation.jl" begin
    # Write your tests here.

    # test if whitening dataset results in unitary diagonal covariance matrix
    root = pkgdir(ICA_BlindSourceSeparation)
    path = joinpath(root, "data", "foetal_ecg.dat")
    dataset_white = whiten_dataset(read_dataset(path))
    N = size(dataset_white.data, 2)
    Sigma = cov(dataset_white.data, dims=1, corrected=false)
    @test isapprox(Sigma, I(N))
    
end
