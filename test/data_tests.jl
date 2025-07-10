function compare_waves(A::AbstractVector, B::AbstractVector)
    if (length(A) != length(B))
        return false
    end
    n = length(A)
    correct_c = true
    correct_d = true
    for i = 1:n
        a = A[i]
        b = B[i]
        c = isapprox(a, b; atol=0.1)
        d = isapprox(a, b * -1; atol=0.1)
        if c == false
            #println("$a == $b = $c")
            correct_c = false
        end
        if d == false
            #println("$a == $b = $c")
            correct_d = false
        end
    end
    return correct_c || correct_d
end

function test_algo(algo, iterations::Integer)
    n_sources = 2
    n_samples = 1000
    # Create a time vector for our signals
    time = collect(range(0, stop=10, length=n_samples))
    # Source 1: A sine wave
    s1 = sin.(0.8 * 2 * π * time)' # Note the transpose ' to make it a 1x1000 row vector
    # Source 2: A square wave
    s2 = sign.(sin.(1.5 * 2 * π * time))' # Transpose to a 1x1000 row vector
    S_true = vcat(s1, s2)
    # mixing matrix
    A = [0.8 0.5; 0.3 0.9] # Using a fixed matrix
    # Mix the sources
    nonsep = sensorData(time, S_true')

    X = Matrix((A * S_true)')
    separated = sensorData(time, X)

    for i = 1:iterations
        separated, _ = perform_separation(separated, algo)

        correct_aa = compare_waves(nonsep.data[:, 1], separated.data[:, 1])
        correct_ab = compare_waves(nonsep.data[:, 1], separated.data[:, 2])
        correct_ba = compare_waves(nonsep.data[:, 2], separated.data[:, 1])
        correct_bb = compare_waves(nonsep.data[:, 2], separated.data[:, 2])

        c0 = correct_aa || correct_bb
        c1 = correct_ab || correct_ba

        if c0 || c1
            return true
        end
    end
    return false
end

function data_test_all_algos()
    @test test_algo(Jade(2), 1)
    @test test_algo(Shibbs(2, 1000), 100)
    @test test_algo(Picard(2, 200, 1e-6, 1e-2, 10, false), 1)
end