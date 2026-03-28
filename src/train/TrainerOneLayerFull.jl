module TrainerFull

using Random

using ..AD: Dual, seed!, value, deriv
using ..Inner: PWLinearNodal
using ..Layers: KALayer
using ..Model: OneLayerModel, loss_batch, accuracy
using ..Train: SGD, sgd_step!, reset!
using ..Utils: RunLogger, new_run!, log_epoch!, write_summary!, tic!, toc!

export train_one_layer_full!

# Dual constante con derivadas en cero
const_dual(x::T, P::Int) where {T<:Real} = Dual{T}(x, zeros(T, P))

"""
Cuenta total de parámetros entrenables en fase full:
P = |A| + |B| + sum_{i,q,p} |values(i,q,p)|, con |values| = K+1
"""
function nparams_full(layer::KALayer{T}) where {T<:Real}
    P_A = length(layer.A)
    P_B = length(layer.B)
    P_phi = layer.m_out * layer.Q * layer.n_in * (layer.K + 1)
    return P_A + P_B + P_phi, P_A, P_B, P_phi
end

"""
Construye una KALayer dualizada para entrenar A,B y los valores nodales de φ.
- knots quedan constantes (deriv=0)
- values se siembran con base canónica
"""
function lift_layer_full_dual(layer::KALayer{T}) where {T<:Real}
    P, P_A, P_B, P_phi = nparams_full(layer)
    m_out, Q = size(layer.A)
    n_in = layer.n_in
    K = layer.K

    A_dual = Matrix{Dual{T}}(undef, m_out, Q)
    B_dual = Matrix{Dual{T}}(undef, m_out, Q)

    idx = 0
    @inbounds for i in 1:m_out, q in 1:Q
        idx += 1
        A_dual[i, q] = seed!(layer.A[i, q], idx, P)
    end
    @inbounds for i in 1:m_out, q in 1:Q
        idx += 1
        B_dual[i, q] = seed!(layer.B[i, q], idx, P)
    end

    # phi: knots constantes; values entrenables
    phi_dual = Array{PWLinearNodal{Dual{T}}, 3}(undef, m_out, Q, n_in)

    @inbounds for i in 1:m_out, q in 1:Q, p in 1:n_in
        pw = layer.phi[i, q, p]
        knots_d = [const_dual(k, P) for k in pw.knots]
        vals_d  = Vector{Dual{T}}(undef, K+1)
        for k in 1:(K+1)
            idx += 1
            vals_d[k] = seed!(pw.values[k], idx, P)
        end
        phi_dual[i, q, p] = PWLinearNodal{Dual{T}}(knots_d, vals_d)
    end

    @assert idx == P

    layer_d = KALayer{Dual{T}}(layer.n_in, layer.m_out, layer.Q, layer.K,
                               A_dual, B_dual, phi_dual)
    return layer_d, P, P_A, P_B, P_phi
end

"""
Desempaqueta gradiente g (longitud P) a:
- gA::Matrix
- gB::Matrix
- gphi::Array (m_out, Q, n_in, K+1) en el mismo orden de siembra
"""
function unpack_grads_full(g::Vector{T}, layer::KALayer{T}) where {T<:Real}
    P, P_A, P_B, P_phi = nparams_full(layer)
    @assert length(g) == P

    gA = reshape(copy(g[1:P_A]), size(layer.A))
    gB = reshape(copy(g[P_A+1:P_A+P_B]), size(layer.B))

    # gphi en orden i,q,p,k
    gphi = Array{T, 4}(undef, layer.m_out, layer.Q, layer.n_in, layer.K+1)
    idx = P_A + P_B
    @inbounds for i in 1:layer.m_out, q in 1:layer.Q, p in 1:layer.n_in, k in 1:(layer.K+1)
        idx += 1
        gphi[i, q, p, k] = g[idx]
    end
    @assert idx == P
    return gA, gB, gphi
end

"""
Aplica actualización SGD a A,B y values de phi usando grads.
Se implementa como bloques vectorizados (A,B,phi_values).
"""
function sgd_apply_full!(opt::SGD{T}, layer::KALayer{T}, gA::Matrix{T}, gB::Matrix{T}, gphi::Array{T,4}) where {T<:Real}
    # Vectorizar params
    A_vec = copy(vec(layer.A))
    B_vec = copy(vec(layer.B))

    # phi values vector en el mismo orden i,q,p,k
    phi_vec = Vector{T}(undef, length(A_vec) * 0 + length(B_vec) * 0 + layer.m_out*layer.Q*layer.n_in*(layer.K+1))
    idx = 0
    @inbounds for i in 1:layer.m_out, q in 1:layer.Q, p in 1:layer.n_in, k in 1:(layer.K+1)
        idx += 1
        phi_vec[idx] = layer.phi[i, q, p].values[k]
    end

    # Vectorizar grads
    gA_vec = collect(vec(gA))
    gB_vec = collect(vec(gB))
    gphi_vec = Vector{T}(undef, length(phi_vec))
    idx = 0
    @inbounds for i in 1:layer.m_out, q in 1:layer.Q, p in 1:layer.n_in, k in 1:(layer.K+1)
        idx += 1
        gphi_vec[idx] = gphi[i, q, p, k]
    end

    params = [A_vec, B_vec, phi_vec]
    grads  = [gA_vec, gB_vec, gphi_vec]

    sgd_step!(opt, params, grads)

    # Copiar de vuelta
    layer.A .= reshape(params[1], size(layer.A))
    layer.B .= reshape(params[2], size(layer.B))

    idx = 0
    @inbounds for i in 1:layer.m_out, q in 1:layer.Q, p in 1:layer.n_in, k in 1:(layer.K+1)
        idx += 1
        layer.phi[i, q, p].values[k] = params[3][idx]
    end

    return nothing
end

"""
Computa loss y gradientes full (A,B,phi-values) por forward-AD.
"""
function grads_full(model::OneLayerModel{T}, X::AbstractMatrix{T}, y::AbstractVector{<:Integer}) where {T<:Real}
    layer_d, P, P_A, P_B, P_phi = lift_layer_full_dual(model.layer)
    model_d = OneLayerModel(layer_d)

    Ld = loss_batch(model_d, X, y)
    g = deriv(Ld)
    gA, gB, gphi = unpack_grads_full(g, model.layer)
    return value(Ld), gA, gB, gphi
end

"""
Entrena A,B y nudos y_k (values) de todas las φ.
"""
function train_one_layer_full!(model::OneLayerModel{T}, split;
                               epochs::Int=300,
                               lr::Real=0.02,
                               seed::Int=0,
                               run_tag::AbstractString="iris_one_layer_full",
                               runs_root::AbstractString="runs") where {T<:Real}

    rng = MersenneTwister(seed)

    Xtr, ytr = split.X_train, split.y_train
    Xva, yva = split.X_val, split.y_val
    Xte, yte = split.X_test, split.y_test

    lg = RunLogger(runs_root=runs_root)
    run_dir = new_run!(lg; tag=run_tag, meta=Dict(
        "seed" => seed,
        "epochs" => epochs,
        "lr" => lr,
        "n_in" => model.layer.n_in,
        "m_out" => model.layer.m_out,
        "Q" => model.layer.Q,
        "K" => model.layer.K,
        "phase" => "affine_plus_inner_values"
    ))

    opt = SGD(Float64(lr); momentum=0.0)
    reset!(opt)

    best_val_acc = -1.0
    best_epoch = 0

    for epoch in 1:epochs
        tic!(lg)

        # Gradientes (train)
        Ltr, gA, gB, gphi = grads_full(model, Xtr, ytr)

        # Paso SGD
        sgd_apply_full!(opt, model.layer, gA, gB, gphi)

        # Métricas completas
        acc_tr = accuracy(model, Xtr, ytr)
        acc_va = accuracy(model, Xva, yva)
        acc_te = accuracy(model, Xte, yte)

        Lva = value(loss_batch(model, Xva, yva))
        Lte = value(loss_batch(model, Xte, yte))

        t_epoch = toc!(lg)

        log_epoch!(lg;
            epoch=epoch,
            loss_train=Ltr, acc_train=acc_tr,
            loss_val=Lva,   acc_val=acc_va,
            loss_test=Lte,  acc_test=acc_te,
            time_epoch_s=t_epoch
        )

        if acc_va > best_val_acc
            best_val_acc = acc_va
            best_epoch = epoch
        end
    end

    write_summary!(lg, "best_val_acc=$(best_val_acc)\nbest_epoch=$(best_epoch)\nrun_dir=$(run_dir)")
    return lg.run_dir
end

end # module TrainerFull