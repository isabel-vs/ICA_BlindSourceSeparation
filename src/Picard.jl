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

function regularize_hessian(h::Matrix{Float64}, lambda_min::Float64)::Matrix{Float64}
    for i in eachindex(h)
        if h[i] < lambda_min
            h[i] = lambda_min
        end
    end
    return h
end

function score(Y::AbstractMatrix{Float64})
    return tanh.(Y)
end

function score_der(psiY::AbstractMatrix{Float64})
    return -mean(psiY .^ 2, dims=2) .+ 1.0
end

function gradient(Y::AbstractMatrix{Float64}, psiY::AbstractMatrix{Float64})
    T = size(Y, 2)
    return psiY * Y' ./ T
end

function proj_hessian_approx(Y::AbstractMatrix{Float64}, psidY_mean::AbstractVector{Float64}, G::AbstractMatrix{Float64})
    N = size(Y, 2)
    diagonal = psidY_mean * ones(1, N)
    off_diag  = diag(G) * ones(1, N)
    return 0.5 .* (diagonal .+ diagonal' .- off_diag .- off_diag')
end

function solve_hessian(G::AbstractMatrix{Float64}, h::AbstractMatrix{Float64})
    return h \ G
end

function loss(Y::AbstractMatrix{Float64}, signs::AbstractVector{Float64})
    N, T = size(Y)
    total = 0.0
    for i in 1:N
        y = Y[i, :]
        total += signs[i] * mean(abs.(y) .+ log1p.(exp.(-2.0 .* abs.(y))))
    end
    return total
end

function l_bfgs_direction(G::AbstractMatrix{Float64}, h::AbstractMatrix{Float64}, s_list::Vector{Matrix{Float64}}, y_list::Vector{Matrix{Float64}}, r_list::Vector{Float64}, ortho::Bool)
    q = copy(G)
    a_list = Float64[]
    for (s, y, r) in Iterators.reverse(zip(s_list, y_list, r_list))
        α = r * sum(s .* q)
        push!(a_list, α)
        q .-= α .* y
    end
    z = if ortho
        q ./ h
    else
        h \ q
    end
    for (s, y, r, α) in zip(s_list, y_list, r_list, Iterators.reverse(a_list))
        β = r * sum(y .* z)
        z .+= (α - β) .* s
    end
    return -z
end

function line_search(Y::AbstractMatrix{Float64}, W::AbstractMatrix{Float64}, log_lik::Function, direction::AbstractMatrix{Float64}, signs::AbstractVector{Float64}, current_loss::Union{Nothing, Float64}; ls_tries::Int, verbose::Bool, ortho::Bool, extended::Bool)
    α = 1.0
    if current_loss === nothing
        current_loss = loss(Y, signs)
    end
    Y_new = similar(Y)
    W_new = similar(W)
    for _ in 1:ls_tries
        transform = ortho ? expm(α .* direction) : I + α .* direction
        Y_new .= transform * Y
        W_new .= transform * W
        new_loss = loss(Y_new, signs)
        if new_loss < current_loss
            return true, Y_new, W_new, new_loss, α .* direction
        end
        α /= 2
    end
    return false, Y_new, W_new, current_loss, α .* direction
end

struct Picard
end

perform_separation(dataset, algo::Picard) = ica_picard(dataset)