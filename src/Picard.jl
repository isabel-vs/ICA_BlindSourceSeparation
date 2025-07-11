#=------------------------------------------------------------
Picard algorithm.
Performs blind source separation using contrast function optimization with L-BFGS.
Author: Mihail Todorov
Adapted from Pierre Ablin, Jean-François Cardoso, Alexandre Gramfort's MATLAB version
=------------------------------------------------------------=#

"""
    ica_picard(dataset::sensorData, sig::Int, m::Int, maxiter::Int, tol::Real, lambda_min::Real, ls_tries::Int; verbose::Bool=false)

Perform Independent Component Analysis (ICA) using the Picard algorithm with limited-memory BFGS optimization.

# Arguments
- `dataset::sensorData`: Input data with timestamps and a matrix of mixed signals
- `sig::Int`: Number of mixed sources to be separated
- `m::Int`: Size of L-BFGS's memory. Typical values are in the range 3-15
- `maxiter::Int`: Maximum number of iterations
- `tol::real`: tolerance for the stopping criterion. 
    Iterations stop when the norm of the projected gradient gets smaller than `tol`.
- `lambda_min::Real`: Minimum eigenvalue for regularizing the Hessian approximation.
- `ls_tries::Int`: Maximum number of attempts in the backtracking line search.
    When that number is exceeded, the direction is discarded and the gradient is used instead.
- `verbose::Bool=false`: If true, prints the informations about the algorithm.

# Returns
- `sensorData`: A new `sensorData` object containing the unmixed signals.
- `W`: The unmixing matrix.
"""
function ica_picard(dataset::sensorData, sig::Int, m::Int, maxiter::Int, tol::Real, lambda_min::Real, ls_tries::Int; verbose::Bool=false)

    w_dataset, Wm, iWm = whiten_dataset(dataset, sig) # whitening the dataset
    X = transpose(w_dataset.data) # transposed data part of the dataset
    N, T = size(X) # saving the sizes, N rows for N signals, T columns for T points in time
    W = Matrix{Float64}(I, N, N) # unmixing matrix, initially identity matrix   
    Y = copy(X) # copying the whitened data

    # vectors for L-BFGS
    s_list = Vector{Matrix{Float64}}() # stores weight updates
    y_list = Vector{Matrix{Float64}}() # stores gradient updates
    r_list = Vector{Float64}() # stores inverse curvature estimates

    current_loss = Inf # initial value for the loss
    sign_change = false # whether any source signal changed sign in current iteration

    # initializing variables used in the for-loop
    old_signs = zeros(Int, N)
    G_old = zeros(N, N)
    direction = zeros(N, N)

    for n in 1:maxiter

        # score function and its average derivative
        psiY = score(Y)
        psidY_mean = score_der(psiY)

        # Compute relative gradient
        g = gradient(Y, psiY)

        K = psidY_mean .- diag(g) # curvature approximation to determine sign flipping
        signs = sign.(K) # extracting signs
        if n > 1
            sign_change = any(signs .!= old_signs) # sign flip
        end
        old_signs = signs # store signs for next iteration

        g = Diagonal(signs) * g # update gradient
        psidY_mean .*= signs # update derivative

        G = (g - g') / 2 # compute gradient matrix

        gradient_norm = maximum(abs.(G))

        # Stop if the gradient norm is below the convergence threshold
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
        G_old = copy(G) # Save current G for the next iteration

        # flush the memory if there is a sign change
        if sign_change
            current_loss = Inf
            empty!(s_list)
            empty!(y_list)
            empty!(r_list)
        end

        # hessian approximation
        h = proj_hessian_approx(Y, psidY_mean, G)
        h = regularize_hessian(h, lambda_min)

        # find the L-BFGS direction
        direction = l_bfgs_direction(G, h, s_list, y_list, r_list)

        # Perform line search along the computed direction
        converged, new_Y, new_loss, alpha = line_search(Y, direction, signs, current_loss; ls_tries=ls_tries)

        # If line search fails, reset direction to steepest descent and flush memory
        if !converged
            direction = -G
            empty!(s_list)
            empty!(y_list)
            empty!(r_list)
            # Retry line search with steepest descent:
            _, new_Y, new_loss, alpha = line_search(Y, direction, signs, current_loss; ls_tries=ls_tries)
        end

        direction .*= alpha # Scale update direction
        Y = new_Y 
        W = exp(direction) * W # Update unmixing matrix
        current_loss = new_loss

        # Optionally print progress
        if verbose
            println("iteration ", n, ", gradient norm = ", round(gradient_norm; sigdigits=4))
        end
    end

    return sensorData(dataset.time, Matrix(transpose(Y))), W
end

"""
    score(Y::AbstractArray{<:Real})

Apply the hyperbolic tangent elementwise to each entry of `Y`.

# Arguments
- `Y::AbstractArray{<:Real}`: input array (vector, matrix, or higher‑dimensional array) of real numbers.

# Returns
- an array with the same shape as `Y`, where each element is `tanh(y)`, with `y` being an element of `Y`.
"""
function score(Y::AbstractArray{<:Real})
    return tanh.(Y)
end

"""
    score_der(psiY::AbstractMatrix{<:Real})

Compute the average derivative of the `tanh` nonlinearity for each row of `psiY`.
   
# Arguments
- `psiY::AbstractMatrix{<:Real}`: An `N×T` matrix, typically computed as `tanh.(Y)`.

# Returns
- a vector in which each entry represents the average derivative of the `1 - tanh(y)^2` over each row of `psiY`, i.e. the average derivative of the `tanh` nonlinearity per component.
"""
function score_der(psiY::AbstractMatrix{<:Real})
    return vec(1 .- mean(psiY .^ 2, dims=2))
end

"""
    gradient(Y, psiY)
    
Compute the gradient of the contrast function with respect to the input signals.
    
# Arguments
- `Y::AbstractMatrix{<:Real}`: An `N×T` matrix where each row represents one estimated source over `T` time points.
- `psiY::AbstractMatrix{<:Real}`: An `N×T` matrix, typically computed as `tanh.(Y)`.
    
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
- `psidY_mean::AbstractVector{<:Real}`: a length-`N` vector with average score derivatives per component.
- `G::AbstractMatrix{<:Real}`: the gradient matrix of size `N×N`.

# Returns
- An `N×N` symmetric matrix approximating the projected Hessian.
"""
function proj_hessian_approx(Y::AbstractMatrix{<:Real}, psidY_mean::AbstractVector{<:Real}, G::AbstractMatrix{<:Real})
    N = size(Y, 1) # Number of signals
    diagonal = psidY_mean * ones(1, N) # Мatrix where each row is psidY_mean
    off_diag  = repeat(diag(G), 1, N) # Matrix where each row is the diagonal of G
    hess = 0.5 * (diagonal + transpose(diagonal) - off_diag - transpose(off_diag)) # Symmetric Hessian approximation
    return hess
end

"""
    regularize_hessian(h, lambda_min)
    
Clip the diagonal values of the Hessian approximation from below, ensuring all values are at least `lambda_min`.
    
# Arguments
- `h::AbstractMatrix{<:Real}`: a diagonal matrix, where diagonal elements approximate eigenvalues.
- `lambda_min::Real`: minimum allowed eigenvalue
    
# Returns
- Regularized matrix with no values below `lambda_min`
"""
function regularize_hessian(h::AbstractMatrix{<:Real}, lambda_min::Real)
    return max.(h, lambda_min)
end

"""
    solve_hessian(G, h)
    
Compute the product of the inverse Hessian approximation with the gradient.
    
# Arguments
- `G::AbstractMatrix{<:Real}`: the gradient matrix of size `N×N`.
- `h::AbstractMatrix{<:Real}`: symmetric Hessian approximation of the same size as `G`.
    
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
- `Y::AbstractMatrix{<:Real}`: matrix of shape `N×T`, where each row is a signal component over `T` time points.
- `signs::AbstractVector{<:Real}`: a vector of length `N`, containing signs for each signal value.
    
# Returns
- A scalar representing the average loss across all components and time points.
"""
function loss(Y, signs)
    N, T = size(Y)
    total = 0.0
    for i in 1:N
        y = Y[i, :] # Extract i-th signal row
        s = signs[i] # Corresponding sign row
        term = abs.(y) .+ log1p.(exp.(-2 .* abs.(y))) # smooth loss term
        total += sum(s .* term) / T # average loss for this component with sign applied
    end
    return total
end

"""
    l_bfgs_direction(G, h, s_list, y_list, r_list)
    
Compute a search direction using the limited-memory BFGS (L-BFGS) algorithm.
    
# Arguments
- `G::AbstractMatrix{<:Real}: the current gradient matrix.
- `h::AbstractMatrix{<:Real}`: a diagonal approximation to the Hessian matrix.
- `s_list::Vector{AbstractMatrix{<:Real}}`: list of previous update matrices.
- `y_list::Vector{AbstractMatrix{<:Real}}`: list of previous gradient differences.
- `r_list::Vector{Float64}`: list of scalars for each pair `(s, y)`.
    
# Returns
- the descent direction computed using the L-BFGS two-loop recursion.
"""
function l_bfgs_direction(G, h, s_list, y_list, r_list)
    q = copy(G) # Initialize q with the current gradient
    a_list = Vector{Float64}()
    m = length(s_list)
    for i in m:-1:1 # iterate from most recent to oldest update
        s = s_list[i]
        y = y_list[i]
        r = r_list[i]
        alpha = r * sum(s .* q)
        push!(a_list, alpha)
        q .-= alpha .* y
    end
    z = solve_hessian(q, h)
    for i in 1:m # iterate from oldest to most recent update
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
- `Y::AbstractMatrix{<:Real}`: current signal matrix of size`N×T`.
- `direction::AbstractMatrix{<:Real}`: descent direction matrix of size `N×N`.
- `signs::AbstractVector{<:Real}`: vector of length `N` containing sign weights for each signal component.
- `current_loss::Real`: current loss value.
- `ls_tries::Integer` (keyword): maximum number of backtracking steps.
    
# Returns
- A tuple `(converged, Y_new, new_loss, alpha)` where
    - `converged::Bool` indicates whether a successful step was found,
    - `Y_new::AbstractMatrix{<:Real}` is the updated signal matrix (or original if no step succeeded),
    - `new_loss::Real` is the loss at `Y_new`,
    - `alpha::Real` is the final step size.
"""
function line_search(Y, direction, signs, current_loss; ls_tries)
    alpha = 1.0 # initial step size
    loss0 = isfinite(current_loss) ? current_loss : loss(Y, signs) # compute current loss if needed
    Y_new = Y 
    new_loss = loss0
    for _ in 1:ls_tries
        Y_candidate = exp(alpha * direction) * Y # propose new Y
        cand_loss = loss(Y_candidate, signs) # evaluate the loss
        if cand_loss < loss0 
            return true, Y_candidate, cand_loss, alpha
        end
        alpha /= 2 # if candidate loss in not a solution, reduce step size
    end
    return false, Y_new, new_loss, alpha
end

struct Picard
    sig::Int
    m::Int
    maxiter::Int
    tol::Real
    lambda_min::Real
    ls_tries::Int
    verbose::Bool
end

perform_separation(dataset, algo::Picard) = ica_picard(
    dataset,
    algo.sig,
    algo.m,
    algo.maxiter,
    algo.tol,
    algo.lambda_min,
    algo.ls_tries;
    verbose = algo.verbose
    )