using LinearAlgebra

function ica_picard(dataset::sensorData)
    
    tol = 1e-7
    lambda_min = 0.01
    ls_tries = 10
    verbose = false
    m = 7
    max_iter = 1000

    X = deepcopy(dataset.data)
    T, N = size(X)
    W = Matrix{Float64}(I, N, N)
    Y = X

    s_list = Vector{Matrix{Float64}}()
    y_list = Vector{Matrix{Float64}}()
    r_list = Vector{Float64}()

    signs = ones(N)
    current_loss = loss(Y, signs)
    requested_tolerance = false
    sign_change = false

    old_signs = copy(signs)
    G_old = zeros(N, N)
    direction = zeros(N, N)

    for n in 1:max_iter
        psiY       = tanh.(Y)
        psidY_mean = mean(1 .- psiY.^2, dims=1)[:]
        g          = psiY' * Y / T

        K     = psidY_mean .* diag(g)
        signs = sign.(K)
        if n > 1
            sign_change = any(signs .!= old_signs)
        end
        old_signs = signs

        g .*= reshape(signs, N, 1)
        psidY_mean .*= signs

        G = (g - g') / 2

        diagonal = psidY_mean * ones(1, N)
        off_diag  = diag(g) * ones(1, N)
        h         = 0.5 .* (diagonal .+ diagonal' .- off_diag .- off_diag')
        h = regularize_hessian(h, lambda_min)

        gradient_norm = maximum(abs.(G))
        if gradient_norm < tol
            requested_tolerance = true
            break
        end

        if n > 1
            push!(s_list, direction)
            ΔG = G .- G_old
            push!(y_list, ΔG)
            push!(r_list, 1.0 / sum(direction .* ΔG))
            if length(s_list) > m
                popfirst!(s_list)
                popfirst!(y_list)
                popfirst!(r_list)
            end
        end
        G_old = copy(G)

        if sign_change
            current_loss = nothing
            empty!(s_list)
            empty!(y_list)
            empty!(r_list)
        end

        direction = l_bfgs_direction(G, h, s_list, y_list, r_list, true)

        converged, Y_new, W_new, new_loss, direction =
            line_search(Y, W, log_lik_tanh, direction, signs, current_loss;
                        ls_tries=ls_tries, verbose=verbose, ortho=true, extended=false)
        if !converged
            direction .= -G
            empty!(s_list)
            empty!(y_list)
            empty!(r_list)
            _, Y_new, W_new, new_loss, direction =
                line_search(Y, W, log_lik_tanh, direction, signs, current_loss;
                            ls_tries=10, verbose=false, ortho=true, extended=false)
        end

        Y = Y_new
        W = expm(direction) * W
        current_loss = new_loss

        if verbose
            println("iteration $(n), gradient norm = $(gradient_norm), loss = $(current_loss)")
        end
    end

    return sensorData(dataset.time, Y)
end

"""
    core(Y::AbstractArray{<:Real})
    Apply the hyperbolic tangent elementwise to each entry of `Y`.
    # Arguments
    - `Y::AbstractArray{<:Real}`: input array (vector, matrix, or higher‑dimensional array) of real numbers.
    # Returns
    - an array with the same shape as `Y`, where each element is `tanh(y)`.
"""
function score(Y::AbstractArray{<:Real})
    return tanh.(Y)
end

"""
    core_der(psiY::AbstractMatrix{<:Real})
    Compute the average derivative of the hyperbolic tangent nonlinearity for each row of `psiY`.
    # Arguments
    - `psiY::AbstractMatrix{<:Real}`: input array of size `N×T`, where each row is a signal component over `T` observations.
    # Returns
    - an `N×1` array in which each entry  represents the average derivative of the `tanh` nonlinearity, evaluated at each value in the corresponding row of `psiY`.
"""
function score_der(psiY::AbstractMatrix{<:Real})
    return -mean(psiY .^ 2, dims=2) .+ 1.0
end

function gradient(Y, psiY)
    T = size(Y, 2)
    return (psiY * transpose(Y)) / T
end

function proj_hessian_approx(Y, psidY_mean, G)
    N = size(Y, 1)
    diagonal = psidY_mean * ones(1, N)
    off_diag  = repeat(diag(G), 1, N)
    hess = 0.5 * (diagonal + transpose(diagonal) - off_diag - transpose(off_diag))
    return hess
end

function regularize_hessian(h, lambda_min)
    return max.(h, lambda_min)
end

function solve_hessian(G, h)
    return G ./ h
end

function loss(Y, signs)
    N, T = size(Y)
    total = 0.0
    for i in 1:N
        y = Y[i, :]
        s = signs[i, :]
        term = abs.(y) .+ log1p.(exp.(-2 .* abs.(y)))
        total += sum(s .* term) / T
    end
    return total
end

function l_bfgs_direction(G, h, s_list, y_list, r_list)
    q = copy(G)
    a_list = Vector{Float64}()
    m = length(s_list)
    for i in m:-1:1
        s = s_list[i]
        y = y_list[i]
        r = r_list[i]
        alpha = r * sum(s .* q)
        push!(a_list, alpha)
        q .-= alpha .* y
    end
    z = solve_hessian(q, h)
    for i in 1:m
        s = s_list[i]
        y = y_list[i]
        r = r_list[i]
        alpha = a_list[m - i + 1]
        beta = r * sum(y .* z)
        z .+= (alpha - beta) .* s
    end
    return -z
end

function line_search(Y, direction, signs, current_loss; ls_tries)
    alpha = 1.0
    loss0 = isfinite(current_loss) ? current_loss : loss(Y, signs)
    Y_new = Y
    new_loss = loss0
    for _ in 1:ls_tries
        Y_candidate = exp(alpha * direction) * Y
        cand_loss = loss(Y_candidate, signs)
        if cand_loss < loss0
            return true, Y_candidate, cand_loss, alpha
        end
        alpha /= 2
    end
    return false, Y_new, new_loss, alpha
end

struct Picard
end

perform_separation(dataset, algo::Picard) = ica_picard(dataset)