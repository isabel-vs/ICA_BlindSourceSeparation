using ICA_BlindSourceSeparation
using Test
using Statistics: cov
using LinearAlgebra: I

@testset "ICA_BlindSourceSeparation.jl" begin
    # Write your tests here.

    # test if whitening dataset results in unitary diagonal covariance matrix
    root = pkgdir(ICA_BlindSourceSeparation)
    path = joinpath(root, "data", "foetal_ecg.dat")
    dataset_white = whiten_dataset(read_dataset(path))
    N = size(dataset_white.data, 2)
    Sigma = cov(dataset_white.data, dims=1, corrected=false)
    @test isapprox(Sigma, I(N))

    # test if reading functions recognize edgecases correctly
    test_files = ["empty.dat", "misshaped.dat", "timecolumn.dat"]
    for i in test_files
        path = joinpath(root, "data", i)
        test_val = false
        try
            read_dataset(path)
        catch
            test_val = true
        end
        @test test_val
    end
    
    path = joinpath(root, "data", "with_comments.dat")
    test_val = true
    try
        read_dataset(path)
    catch
        test_val = false
    end
    @test test_val
    
    # testidea: all algorithms should create roughly the same output

    # other testidea: compare Jade and Shibbs result with C or Matlab result from JnS
    # JADE
    
    path = joinpath(root, "data", "foetal_ecg.dat")
    x = read_dataset(path)
    algo = Jade(2)
    x = perform_separation(x, algo)
    n, m = size(x.data)
    @test (n == 2500) && (m == 2)

    # Shibbs
    path = joinpath(root, "data", "foetal_ecg.dat")
    x = read_dataset(path)
    algo = Shibbs(2)
    x, _ = perform_separation(x, algo)
    n, m = size(x.data)
    @test (n == 2500) && (m == 2)

    # Picard
    #=
    path = joinpath(root, "data", "foetal_ecg.dat")
    x = read_dataset(path)
    algo = Picard(???)
    x = perform_separation(x, algo)
    n, m = size(x.data)
    @test (n == 2500) && (m == 2)
    =#
end
