module Trainer

using Random

using ..AD: Dual, seed!, value, deriv
using ..Inner: PWLinearNodal

using ..Layers: KALayer, forward_batch
using ..Model: OneLayerModel, loss_batch, accuracy
using ..Train: SGD, sgd_step!, reset!
using ..Utils: RunLogger, new_run!, log_epoch!, write_summary!, tic!, toc!

export train_one_layer_affine_only!

# ---- Helpers: dualización de parámetros (solo A,B) ----

# Dual constante con derivadas en cero
const_dual(x::T, P::Int) where {T<:Real} = Dual{T}(x, zeros(T, P))

"""
    lift_layer_affine_dual(layer::KALayer{Float64})

Crea una capa KALayer{Dual{Float64}} donde:
- A y B se siembran con base canónica (P = |A| + |B|)
- phi (knots/values) se convierte a Dual constante (deriv = 0), porque NO se entrena inner en fase 1
"""
function lift_layer_affine_dual(layer::KALayer{T}) where {T<:Real}
    m_out, Q = size(layer.A)
    P_A = length(layer.A)
    P_B = length(layer.B)
    P = P_A + P_B

    # Dual A y B con semillas
    A_dual = Matrix{Dual{T}}(undef, m_out, Q)
    B_dual = Matrix{Dual{T}}(undef, m_out, Q)

    # A: índices 1..P_A
    idx = 0
    @inbounds for i in 1:m_out, q in 1:Q
        idx += 1
        A_dual[i, q] = seed!(layer.A[i, q], idx, P)
    end
    # B: índices P_A+1 .. P
    @inbounds for i in 1:m_out, q in 1:Q
        idx += 1
        B_dual[i, q] = seed!(layer.B[i, q], idx, P)
    end
    @assert idx == P

    # phi: convertir a Dual constante (deriv=0)
    n_in = layer.n_in
    phi_dual = Array{typeof(layer.phi[1,1,1]),3}(undef, m_out, Q, n_in) # placeholder type
    # Necesitamos PWLinearNodal{Dual{T}}, así que reconstruimos cada uno
    #phi_dual = Array{IKAM.Inner.PWLinearNodal{Dual{T}}, 3}(undef, m_out, Q, n_in)
    phi_dual = Array{PWLinearNodal{Dual{T}}, 3}(undef, m_out, Q, n_in)

    @inbounds for i in 1:m_out, q in 1:Q, p in 1:n_in
        pw = layer.phi[i, q, p]
        knots_d = [const_dual(k, P) for k in pw.knots]
        vals_d  = [const_dual(v, P) for v in pw.values]
        #phi_dual[i, q, p] = IKAM.Inner.PWLinearNodal{Dual{T}}(knots_d, vals_d)
        phi_dual[i, q, p] = PWLinearNodal{Dual{T}}(knots_d, vals_d)
    end

    # Construimos layer Dual
    #layer_d = IKAM.Layers.KALayer{Dual{T}}(layer.n_in, layer.m_out, layer.Q, layer.K,
    #                                      A_dual, B_dual, phi_dual)
    layer_d = KALayer{Dual{T}}(layer.n_in, layer.m_out, layer.Q, layer.K,
                                          A_dual, B_dual, phi_dual)

    return layer_d, P, P_A, P_B
end

"""
    grads_affine_only(model, X, y)

Retorna:
- loss_val::Float64
- gA::Matrix{Float64} mismo size que A
- gB::Matrix{Float64} mismo size que B
"""
function grads_affine_only(model::OneLayerModel{T}, X::AbstractMatrix{T}, y::AbstractVector{<:Integer}) where {T<:Real}
    # Dualizamos capa
    layer_d, P, P_A, P_B = lift_layer_affine_dual(model.layer)
    model_d = OneLayerModel(layer_d)

    # Pérdida dual (promedio batch)
    Ld = loss_batch(model_d, X, y)
    g = deriv(Ld)
    @assert length(g) == P

    # Desempaquetar gradientes a matrices
    gA_vec = g[1:P_A]
    gB_vec = g[P_A+1:end]

    gA = reshape(copy(gA_vec), size(model.layer.A))
    gB = reshape(copy(gB_vec), size(model.layer.B))

    return value(Ld), gA, gB
end

# ---- Entrenamiento full-batch: fase 1 (solo A,B) ----

"""
    train_one_layer_affine_only!(model, split; epochs, lr, seed, run_tag, runs_root)

Entrena SOLO parámetros afines (A,B) de una capa.
Registra por época:
loss/acc en train, val, test + tiempo por época.
"""
function train_one_layer_affine_only!(model::OneLayerModel{T}, split;
                                      epochs::Int=200,
                                      lr::Real=0.05,
                                      seed::Int=0,
                                      run_tag::AbstractString="iris_one_layer_affine_only",
                                      runs_root::AbstractString="runs") where {T<:Real}

    rng = MersenneTwister(seed)

    Xtr, ytr = split.X_train, split.y_train
    Xva, yva = split.X_val, split.y_val
    Xte, yte = split.X_test, split.y_test

    # Logger
    lg = RunLogger(runs_root=runs_root)
    run_dir = new_run!(lg; tag=run_tag, meta=Dict(
        "seed" => seed,
        "epochs" => epochs,
        "lr" => lr,
        "n_in" => model.layer.n_in,
        "m_out" => model.layer.m_out,
        "Q" => model.layer.Q,
        "K" => model.layer.K,
        "phase" => "affine_only"
    ))

    # Optimizer (SGD)
    opt = SGD(Float64(lr); momentum=0.0)
    reset!(opt)

    best_val_acc = -1.0
    best_epoch = 0

    for epoch in 1:epochs
        tic!(lg)

        # 1) Gradientes por forward-AD (solo train)
        Ltr, gA, gB = grads_affine_only(model, Xtr, ytr)

        # 2) Paso SGD (operamos con bloques vectorizados y re-asignamos)
        A_vec = copy(vec(model.layer.A))
        B_vec = copy(vec(model.layer.B))
        gA_vec = vec(gA)
        gB_vec = vec(gB)

        params = [A_vec, B_vec]
        grads  = [collect(gA_vec), collect(gB_vec)]
        sgd_step!(opt, params, grads)

        # Copiar de vuelta
        model.layer.A .= reshape(params[1], size(model.layer.A))
        model.layer.B .= reshape(params[2], size(model.layer.B))

        # 3) Métricas completas (train/val/test) con el modelo float actual
        acc_tr = accuracy(model, Xtr, ytr)
        acc_va = accuracy(model, Xva, yva)
        acc_te = accuracy(model, Xte, yte)

        Lva = loss_batch(model, Xva, yva) |> value
        Lte = loss_batch(model, Xte, yte) |> value

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

end # module Trainer