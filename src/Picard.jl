using LinearAlgebra
#using Printf

function ica_picard(dataset::sensorData, m::Int, maxiter::Int, tol::Real, lambda_min::Real, ls_tries::Int; verbose::Bool=false)

    X = transpose(dataset.data)
    N, T = size(X)
    W = Matrix{Float64}(I, N, N)    
    Y = copy(X)

    # vectors for L-BFGS
    s_list = Vector{Matrix{Float64}}()
    y_list = Vector{Matrix{Float64}}()
    r_list = Vector{Float64}()

    current_loss = Inf
    sign_change = false # whether signs flipped in current iteration
    old_signs = zeros(Int, N)
    G_old = zeros(N, N)
    direction = zeros(N, N)

    for n in 1:maxiter

        # score function and derivative
        psiY = score(Y)
        psidY_mean = score_der(psiY)

        g = gradient(Y, psiY)

        K = psidY_mean .- diag(g)
        signs = sign.(K)
        if n > 1
            sign_change = any(signs .!= old_signs) # sign flip
        end
        old_signs = signs

        g = Diagonal(signs) * g
        psidY_mean .*= signs

        G = (g - g') / 2

        gradient_norm = maximum(abs.(G))
        if gradient_norm < tol
            break
        end

        # update the L-BFGS memory
        if n > 1
            push!(s_list, direction)
            y = G .- G_old
            push!(y_list, y)
            push!(r_list, 1.0 / sum(direction .* y))
            if length(s_list) > m
                popfirst!(s_list)
                popfirst!(y_list)
                popfirst!(r_list)
            end
        end
        G_old = copy(G)

        # flush the memory if there is a sign change
        if sign_change
            current_loss = Inf
            empty!(s_list)
            empty!(y_list)
            empty!(r_list)
        end

        # hssian approximation
        h = proj_hessian_approx(Y, psidY_mean, g)
        h = regularize_hessian(h, lambda_min)

        # find the L-BFGS direction
        direction = l_bfgs_direction(G, h, s_list, y_list, r_list)

        converged, new_Y, new_loss, alpha = line_search(Y, direction, signs, current_loss; ls_tries=ls_tries)
        if !converged
            direction = -G
            empty!(s_list)
            empty!(y_list)
            empty!(r_list)
            _, new_Y, new_loss, alpha = line_search(Y, direction, signs, current_loss; ls_tries=ls_tries)
        end

        direction .*= alpha
        Y = new_Y
        W = exp(direction) * W
        current_loss = new_loss

        if verbose
            println("iteration ", n, ", gradient norm = ", round(gradient_norm; sigdigits=4))
        end
    end

    return sensorData(dataset.time, transpose(Y))
end

"""
    score(Y::AbstractArray{<:Real})

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
    score_der(psiY::AbstractMatrix{<:Real})

 Computes the average derivative of the hyperbolic tangent nonlinearity for each row of `psiY`.
   
 # Arguments
 - `psiY::AbstractMatrix{<:Real}`: input array of size `N×T`, where each row is a signal component over `T` observations.

 # Returns
 - a vebtor in which each entry  represents the average derivative of the `tanh` nonlinearity, evaluated at each value in the corresponding row of `psiY`.
"""
function score_der(psiY::AbstractMatrix{<:Real})
    return vec(1 .- mean(psiY .^ 2, dims=2))
end

"""
    gradient(Y, psiY)
    
Compute the gradient of the contrast function with respect to the input signals.
    
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

Compute an approximation of the projected Hessian matrix.

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
    
Clip the diagonal values of the Hessian approximation from below, ensuring all values are at least `lambda_min`.
    
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
    
Compute the product of the inverse Hessian approximation with the gradient.
    
# Arguments
- `G::AbstractMatrix{<:Real}`: the gradient matrix.
- `h::AbstractMatrix{<:Real}`: diagonal approximation of the Hessian.
    
# Returns
- A matrix of same size as `G`, where each element is `G[i,j] / h[i,j]`.
"""
function solve_hessian(G::AbstractMatrix{<:Real}, h::AbstractMatrix{<:Real})
    return G ./ h
end

"""
    loss(Y, signs)

Compute the total loss for a set of signals.
    
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
    
Compute a search direction using the limited-memory BFGS (L-BFGS) algorithm.
    
# Arguments
- `G::AbstractMatrix{<:Real}: the current gradient.
- `h::AbstractMatrix{<:Real}`: a diagonal approximation to the Hessian.
- `s_list::Vector{AbstractMatrix{<:Real}}`: list of previous update vectors.
- `y_list::Vector{AbstractMatrix{<:Real}}`: list of previous gradient differences.
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
- `signs::AbstractVector{<:Real}`: sign weights for the loss, same size as `Y`.
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
    m::Int
    maxiter::Int
    tol::Real
    lambda_min::Real
    ls_tries::Int
    verbose::Bool
end

perform_separation(dataset, algo::Picard) = ica_picard(
    dataset,
    algo.m,
    algo.maxiter,
    algo.tol,
    algo.lambda_min,
    algo.ls_tries;
    verbose = algo.verbose
    )