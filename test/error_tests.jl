function error_tests()
    # Whiten dataset m > n error
    test_val = false
    try
        y = whiten_dataset(x, 10)
    catch
        test_val = true
    end
    @test test_val

    # Plot dataset mismatch between time length and signal length
    time = [0, 1, 2]
    data = zeros(2,2)
    test_val = false
    try
        plot_dataset(sensorData(time, Matrix(data)))
    catch
        test_val = true
    end
    @test test_val

    # Whiten dataset mismatch between time length and signal length
    test_val = false
    try
        whiten_dataset(sensorData(time, Matrix(data)), 2)
    catch
        test_val = true
    end
    @test test_val

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