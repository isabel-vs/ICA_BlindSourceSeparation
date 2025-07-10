using ICA_BlindSourceSeparation
using Test: @test, @test_throws, @testset
using Statistics: cov
using LinearAlgebra: I
using Plots

include("data_tests.jl")

@testset "ICA_BlindSourceSeparation.jl" begin
    @testset "whitening" begin
        # test if whitening dataset results in unitary diagonal covariance matrix
        root = pkgdir(ICA_BlindSourceSeparation)
        path = joinpath(root, "data", "foetal_ecg.dat")
        dataset_white = whiten_dataset(read_dataset(path))
        N = size(dataset_white.data, 2)
        Sigma = cov(dataset_white.data, dims=1, corrected=false)
        @test isapprox(Sigma, I(N))
    end
    @testset "read_dataset" begin
        # test if reading functions recognize edgecases correctly
        root = pkgdir(ICA_BlindSourceSeparation)
        @test_throws ArgumentError read_dataset(joinpath(root, "data", "empty.dat"))
        @test_throws DimensionMismatch read_dataset(joinpath(root, "data", "misshaped.dat"))
        @test_throws ArgumentError read_dataset(joinpath(root, "data", "timecolumn.dat"))
        
        test_val = true
        test_data = read_dataset(joinpath(root, "data", "with_comments.dat")).data
        if test_data[3,2] != 0.5404
            test_val = false
        end        
        @test test_val
    end
    @testset "Algorithms" begin
        @testset "Jade" begin
            # JADE
            root = pkgdir(ICA_BlindSourceSeparation)
            path = joinpath(root, "data", "foetal_ecg.dat")
            x = read_dataset(path)
            algo = Jade(2)
            x, _ = perform_separation(x, algo)
            n, m = size(x.data)
            @test (n == 2500) && (m == 2)

            @test test_algo(Jade(2), 1)
        end
        @testset "Shibbs" begin
            # Shibbs
            root = pkgdir(ICA_BlindSourceSeparation)
            path = joinpath(root, "data", "foetal_ecg.dat")
            x = read_dataset(path)
            algo = Shibbs(2, 1000)
            x, _ = perform_separation(x, algo)
            n, m = size(x.data)
            @test (n == 2500) && (m == 2)

            @test test_algo(Shibbs(2, 1000), 100)
        end
        @testset "Picard" begin
            # Picard
            root = pkgdir(ICA_BlindSourceSeparation)
            path = joinpath(root, "data", "foetal_ecg.dat")
            x = read_dataset(path)
            algo = Picard(3, 200, 1e-6, 1e-2, 10, false)
            x, _ = perform_separation(x, algo)
            n, m = size(x.data)
            @test (n == 2500) && (m == 8)

            #@test test_algo(Picard(2, 200, 1e-6, 1e-2, 10, false), 1)
        end
    end
    @testset "Plotting" begin
        root = pkgdir(ICA_BlindSourceSeparation)
        path = joinpath(root, "data", "foetal_ecg.dat")
        x = read_dataset(path)
        plt = plot_dataset(x)
        @test plt isa Plots.Plot
        sp = plt[1]
        xaxis_obj = sp.attr[:xaxis]
        yaxis_obj = sp.attr[:yaxis]
        @test sp.attr[:title] == "Estimated Source Signals"
    end

    demo()

    @testset "Error Handling" begin
        root = pkgdir(ICA_BlindSourceSeparation)
        path = joinpath(root, "data", "foetal_ecg.dat")
        x = read_dataset(path)
        @test_throws ArgumentError whiten_dataset(x, 10)

        time = [0, 1, 2]
        data = zeros(2,2)
        @test_throws DimensionMismatch whiten_dataset(sensorData(time, Matrix(data)), 2)
        
        @test_throws DimensionMismatch plot_dataset(sensorData(time, Matrix(data)))

        # Perform separation without implemented algorithm
        #=
        not_a_real_algo = 5
        test_val = false
        try
            perform_separation(x, not_a_real_algo)
        catch
            test_val = true
        end
        @test test_val
        =#
    end
end
