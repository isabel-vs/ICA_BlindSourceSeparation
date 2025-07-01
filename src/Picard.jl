using LinearAlgebra

function ica_picard(dataset::sensorData)::sensorData
    
    tol=1e-7
    lambda_min=0.01
    ls_tries=10
    verbose=false
    m=7
    max_iter=1000

    X = deepcopy(dataset.data) # taking the data part of the parameter dataset
    T, N = size(X) # saving the size of the dataset  - T rows for T points in time and N columns for N signals
    W = Matrix{Float64}(I, N, N) # unmixing matrix, initialy identity matrix
    Y=X # copying the data

    # arrays for L-BFGS
    s_list = Vector{Matrix{Float64}}()
    y_list = Vector{Matrix{Float64}}()
    r_list = Vector{Float64}()

    # signs array
    signs = ones(N)

    # computing the loss
    current_loss = compute_loss(Y, W, log_lik_tanh, signs; ortho=false, extended=true)

    # flags
    requested_tolerance = false # whether stopping tolerance has been reached
    sign_change = false # whether signs flipped in current iteration

    gradient_norm = 1.0 # initialising gradient norm

    covariance = X' * X / T
    C = copy(covariance)

    old_signs = copy(signs)
    G_old = zeros(N, N)
    direction = zeros(N, N)

    for n in 1:max_iter
        
        # Score function and its derivative
        psiY = tanh.(Y)
        psidY = 1 .- psiY .^ 2
        
        G = Y' * psiY / T # Relative gradient

        Y_square = Y .^ 2 # Squared signals

        K = mean(psidY, dims=1)[:] .* diag(C)
        K .-= diag(G)
        signs = sign.(K)

        if n > 1
            sign_change = any(signs .!= old_signs) # Sign flip
        end

        old_signs = signs

        G .= G .* signs # Apply signs to gradient
        psidY .= psidY .* signs

        G .+= C
        psidY .+= 1

        h_off = ones(N)

        # Hessian approximation diagonal (h)
        
        h = psidY' * Y_square ./ T
        h = regularize_hessian(h, h_off, lambda_min)

        G .-= Matrix{Float64}(I, N, N)

        # Stopping criterion
        gradient_norm = maximum(abs.(G))
        if gradient_norm < tol
            requested_tolerance = true
            break
        end

        # Update the L-BFGS memory
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

        # Flush the memory if there is a sign change
        if sign_change
            current_loss = nothing
            empty!(s_list)
            empty!(y_list)
            empty!(r_list)
        end

        # Find the L-BFGS direction
        direction = l_bfgs_direction(G, h, h_off, s_list, y_list, r_list, ortho=false)

         # Do a line-search in that direction
        converged, Y_new, W_new, new_loss, direction = line_search(Y, W, log_lik_tanh, direction, signs, current_loss; ls_tries=ls_tries, verbose=verbose, ortho=false, extended=true)

        if !converged
            direction .= -G

            # flush the memory
            empty!(s_list)
            empty!(y_list)
            empty!(r_list)

            _, Y_new, W_new, new_loss, direction = line_search( Y, W, log_lik_tanh, direction, signs, current_loss; ls_tries=10, verbose=false, ortho=false, extended=true)
        end

        Y = Y_new
        W = W_new
        current_loss = new_loss

        # Update covariance
        C = W * covariance * W'

        # Verbose logging
        if verbose
            println("iteration $(n), gradient norm = $(gradient_norm), loss = $(current_loss)")
        end
    end

    infos = Dict(
        :converged     => requested_tolerance,
        :gradient_norm => gradient_norm,
        :n_iterations  => n
    )

    infos[:signs] = signs

    new_dataset = sensorData(dataset.time, Y)
    return new_dataset
end

function compute_loss(Y::Matrix{Float64}, W::Matrix{Float64}, log_lik_tanh, signs::Vector{Float64}; ortho::Bool, extended::Bool)::Float64
    T, N = size(Y)
    loss = ortho ? 0.0 : -log(abs(det(W)))

    for i in 1:N
        y = Y[:, i]
        s = signs[i]
        loss += s * mean(log_lik_tanh.(y))
        if extended && !ortho
            loss += 0.5 * mean(y .^ 2)
        end
    end
    return loss
end

function log_lik_tanh(x)
    return -log.(cosh.(x))
end

function regularize_hessian(h::Matrix{Float64}, h_off::Vector{Float64}, lambda_min::Float64)::Matrix{Float64}
    N = size(h, 1)
    discr = sqrt.((h .- h').^2 .+ 4 .* (h_off * h_off'))

    eigmat = 0.5 .* (h .+ h' .- discr)

    mask = eigmat .< lambda_min
    for i in 1:N
        mask[i,i] = false
    end

    for (i,j) in CartesianIndices(mask)
        if mask[i,j]
            h[i,j] += (lambda_min - eigmat[i,j])
        end
    end
    return h
end

function l_bfgs_direction(G::Matrix{Float64}, h::Matrix{Float64}, h_off::Vector{Float64}, s_list::Vector{Matrix{Float64}}, y_list::Vector{Matrix{Float64}}, r_list::Vector{Float64}, ortho::Bool)::Matrix{Float64}

    q = copy(G)
    a_list = Float64[]
    for (s, y, r) in Iterators.reverse(zip(s_list, y_list, r_list))
        α = r * sum(s .* q)
        push!(a_list, α)
        q .-= α .* y
    end

    if ortho
        z = q ./ h
        z .= (z .- z') ./ 2
    else
        z = h \ q
    end

    for ((s, y, r), α) in zip(zip(s_list, y_list, r_list), Iterators.reverse(a_list))
        β = r * sum(y .* z)
        z .+= (α - β) .* s
    end

    return -z
end

function line_search(Y::AbstractMatrix{Float64}, W::AbstractMatrix{Float64}, log_lik::Function, direction::AbstractMatrix{Float64}, signs::AbstractVector{Float64}, current_loss::Union{Nothing,Float64}, ls_tries::Int, verbose::Bool, ortho::Bool, extended::Bool)
    
    N = size(W, 1)
    α = 1.0

    if current_loss === nothing
        current_loss = compute_loss(Y, W, log_lik, signs; ortho=ortho, extended=extended)
    end

    Y_new = similar(Y)
    W_new = similar(W)
    new_loss = current_loss

    for _ in 1:ls_tries
        if ortho
            transform = exp(α * direction)
        else
            transform = I + α * direction
        end

        # пробен ъпдейт
        Y_new .= transform * Y
        W_new .= transform * W

        new_loss = compute_loss(Y_new, W_new, log_lik, signs; ortho=ortho, extended=extended)

        if new_loss < current_loss
            return true, Y_new, W_new, new_loss, α .* direction
        end

        α /= 2
    end

    if verbose
        @warn "line search failed, falling back to gradient"
    end

    return false, Y_new, W_new, new_loss, α .* direction
end