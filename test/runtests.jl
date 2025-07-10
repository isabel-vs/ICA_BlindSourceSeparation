using ICA_BlindSourceSeparation
using Test
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
            test_data = read_dataset(path).data
            if test_data[3,2] != 0.5404
                test_val = false
            end        
        catch
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
            x = perform_separation(x, algo)
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

        @testset "Error Handling" begin
            # Test case for time/data dimension mismatch
            invalid_time_data = sensorData(
                [1.0, 2.0], # 2 time points
                [1.0 4.0; 2.0 5.0; 3.0 6.0] # 3 rows of data
            )
            @test_throws DimensionMismatch plot_dataset(invalid_time_data)

            # Test case for data with no columns
            empty_col_data = sensorData(
                [1.0, 2.0, 3.0],
                zeros(3, 0) # 3 rows, 0 columns
            )
            @test_throws ArgumentError plot_dataset(empty_col_data)
        end
    end
end
