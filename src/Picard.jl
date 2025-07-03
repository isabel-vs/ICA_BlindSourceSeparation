using LinearAlgebra

function ica_picard(dataset::sensorData, m::Int, maxiter::Int, tol::Real, lambda_min::Real, ls_tries::Int; verbose::Bool=false)

    X = transpose(dataset.data)
    N, T = size(X)
    W = Matrix{Float64}(I, N, N)
    Y = copy(X)

    s_list = Vector{Matrix{Float64}}()
    y_list = Vector{Matrix{Float64}}()
    r_list = Vector{Float64}()

    #signs = ones(N)
    current_loss = Inf
    #requested_tolerance = false
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
    applies the hyperbolic tangent elementwise to each entry of `Y`.
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
    computes the average derivative of the hyperbolic tangent nonlinearity for each row of `psiY`.
    # Arguments
    - `psiY::AbstractMatrix{<:Real}`: input array of size `N×T`, where each row is a signal component over `T` observations.
    # Returns
    - an `N×1` array in which each entry  represents the average derivative of the `tanh` nonlinearity, evaluated at each value in the corresponding row of `psiY`.
"""
function score_der(psiY::AbstractMatrix{<:Real})
    return -mean(psiY .^ 2, dims=2) .+ 1.0
end

"""
    gradient(Y, psiY)
    computes the gradient of the contrast function with respect to the input signals.
    # Arguments
    - `Y::AbstractMatrix{<:Real}`: an `N×T` matrix where each row is a signal component over `T` samples.
    - `psiY::AbstractMatrix{<:Real}`: the elementwise derivative of the contrast function, same size as `Y`.
    # Returns
    - An `N×N` matrix representing the relative gradient
"""
function gradient(Y::AbstractMatrix{<:Real}, psiY::AbstractMatrix{<:Real})
    T = size(Y, 2)
    return (psiY * transpose(Y)) / T
end

"""
    proj_hessian_approx(Y, psidY_mean, G)
    computes an approximation of the projected Hessian matrix.
    # Arguments
    - `Y::AbstractMatrix{<:Real}`: an `N×T` matrix of current signal components.
    - `psidY_mean::AbstractVector{<:Real}`: a length-`N` vector containing the average of the second derivative (or negative squared derivative) of the contrast function for each component.
    - `G::AbstractMatrix{<:Real}`: the gradient matrix of size `N×N`.
    # Returns
    - An `N×N` symmetric matrix approximating the projected Hessian.
"""
function proj_hessian_approx(Y::AbstractMatrix{<:Real}, psidY_mean::AbstractVector{<:Real}, G::AbstractMatrix{<:Real})
    N = size(Y, 1)
    diagonal = psidY_mean * ones(1, N)
    off_diag  = repeat(diag(G), 1, N)
    hess = 0.5 * (diagonal + transpose(diagonal) - off_diag - transpose(off_diag))
    return hess
end

"""
    regularize_hessian(h, lambda_min)
    clips the diagonal values of the Hessian approximation from below, ensuring all values are at least `lambda_min`.
    # Arguments
    - `h::AbstractMatrix{<:Real}`: a diagonal matrix, where diagonal elements approximate eigenvalues.
    - `lambda_min::Real`: minimum allowed eigenvalue
    # Returns
    - A matrix of the same size as `h`, with all values less than `lambda_min` replaced by `lambda_min`
"""
function regularize_hessian(h::AbstractMatrix{<:Real}, lambda_min::Real)
    return max.(h, lambda_min)
end

"""
    solve_hessian(G, h)
    computes the product of the inverse Hessian approximation with the gradient.
    # Arguments
    - `G::AbstractMatrix{<:Real}`: the gradient matrix.
    - `h::AbstractMatrix{<:Real}`: diagonal approximation of the Hessian.
    # Returns
    - A matrix of same size as `G`, where each element is `G[i,j] / h[i,j]`.
"""
function solve_hessian(G, h)
    return G ./ h
end

"""
    loss(Y, signs)
    computes the total loss for a set of signals.
    # Arguments
    - `Y::AbstractMatrix{<:Real}`: matrix of shape `N×T`, where each row is a signal component over `T` samples.
    - `signs::AbstractMatrix{<:Real}`: matrix of the same shape as `Y`, containing signs or weights for each signal value.
    # Returns
    - A scalar representing the average contrast-based loss across all components and time steps.
"""
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

"""
    l_bfgs_direction(G, h, s_list, y_list, r_list)
    computes a search direction using the limited-memory BFGS (L-BFGS) algorithm.
    # Arguments
    - `G::AbstractVector{<:Real}`: the current gradient.
    - `h::AbstractVector{<:Real}`: a diagonal approximation to the Hessian.
    - `s_list::Vector{AbstractVector{<:Real}}`: list of previous update vectors.
    - `y_list::Vector{AbstractVector{<:Real}}`: list of previous gradient differences.
    - `r_list::Vector{Float64}`: list of scalars for each pair `(s, y)`.
    # Returns
    - the descent direction computed using the L-BFGS two-loop recursion.
"""
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

"""
    function line_search(Y, direction, signs, current_loss; ls_tries)
    Perform a backtracking line search using a matrix exponential update.
    # Arguments
    - `Y::AbstractMatrix{<:Real}`: current signal matrix (`N×T`).
    - `direction::AbstractMatrix{<:Real}`: descent direction matrix of the same size as `Y`.
    - `signs::AbstractMatrix{<:Real}`: sign weights for the loss, same size as `Y`.
    - `current_loss::Real`: current loss value, or `Inf` to force recomputation.
    - `ls_tries::Integer` (keyword): maximum number of backtracking steps.
    # Returns
    - A tuple `(converged, Y_new, new_loss, alpha)` where
    - `converged::Bool` indicates whether a successful step was found,
    - `Y_new::AbstractMatrix{<:Real}` is the updated signal matrix (or original if no step succeeded),
    - `new_loss::Real` is the loss at `Y_new`,
    - `alpha::Real` is the final step size.
"""
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