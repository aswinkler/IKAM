module Trainer2Affine

using Random

using ..AD: Dual, seed!, value, deriv
using ..Inner: PWLinearNodal
using ..Layers: KALayer
using ..Model2: TwoLayerModel, loss_batch, accuracy
using ..Train: SGD, sgd_step!, reset!
using ..Utils: RunLogger, new_run!, log_epoch!, write_summary!, tic!, toc!

export train_two_layer_affine_only!

const_dual(x::T, P::Int) where {T<:Real} = Dual{T}(x, zeros(T, P))

# Dualiza SOLO A,B de una capa, dejando phi constante
function lift_layer_affine_dual(layer::KALayer{T}, offset::Int, P::Int) where {T<:Real}
    m_out, Q = size(layer.A)
    n_in = layer.n_in

    A_dual = Matrix{Dual{T}}(undef, m_out, Q)
    B_dual = Matrix{Dual{T}}(undef, m_out, Q)

    idx = offset
    @inbounds for i in 1:m_out, q in 1:Q
        idx += 1
        A_dual[i, q] = seed!(layer.A[i, q], idx, P)
    end
    @inbounds for i in 1:m_out, q in 1:Q
        idx += 1
        B_dual[i, q] = seed!(layer.B[i, q], idx, P)
    end

    # phi constante
    phi_dual = Array{PWLinearNodal{Dual{T}}, 3}(undef, layer.m_out, layer.Q, layer.n_in)
    @inbounds for i in 1:layer.m_out, q in 1:layer.Q, p in 1:layer.n_in
        pw = layer.phi[i, q, p]
        knots_d = [const_dual(k, P) for k in pw.knots]
        vals_d  = [const_dual(v, P) for v in pw.values]
        phi_dual[i, q, p] = PWLinearNodal{Dual{T}}(knots_d, vals_d)
    end

    layer_d = KALayer{Dual{T}}(layer.n_in, layer.m_out, layer.Q, layer.K, A_dual, B_dual, phi_dual)
    return layer_d, idx
end

function grads_two_layer_affine(model::TwoLayerModel{T}, X::AbstractMatrix{T}, y::AbstractVector{<:Integer}) where {T<:Real}
    # Parámetros entrenables: A,B de layer1 y layer2
    P1 = length(model.layer1.A) + length(model.layer1.B)
    P2 = length(model.layer2.A) + length(model.layer2.B)
    P = P1 + P2

    # Dualizar ambas capas
    idx = 0
    l1d, idx = lift_layer_affine_dual(model.layer1, idx, P)
    l2d, idx = lift_layer_affine_dual(model.layer2, idx, P)
    @assert idx == P

    model_d = TwoLayerModel(l1d, l2d)
    Ld = loss_batch(model_d, X, y)
    g = deriv(Ld)
    @assert length(g) == P

    # Unpack grads
    g1 = g[1:P1]
    g2 = g[P1+1:end]

    P1A = length(model.layer1.A)
    g1A = reshape(copy(g1[1:P1A]), size(model.layer1.A))
    g1B = reshape(copy(g1[P1A+1:end]), size(model.layer1.B))

    P2A = length(model.layer2.A)
    g2A = reshape(copy(g2[1:P2A]), size(model.layer2.A))
    g2B = reshape(copy(g2[P2A+1:end]), size(model.layer2.B))

    return value(Ld), g1A, g1B, g2A, g2B
end

function train_two_layer_affine_only!(model::TwoLayerModel{T}, split;
                                      epochs::Int=300,
                                      lr::Real=0.05,
                                      seed::Int=0,
                                      run_tag::AbstractString="iris_two_layer_affine_only",
                                      runs_root::AbstractString="runs") where {T<:Real}

    Xtr, ytr = split.X_train, split.y_train
    Xva, yva = split.X_val, split.y_val
    Xte, yte = split.X_test, split.y_test

    lg = RunLogger(runs_root=runs_root)
    run_dir = new_run!(lg; tag=run_tag, meta=Dict(
        "seed"=>seed, "epochs"=>epochs, "lr"=>lr,
        "phase"=>"two_layer_affine_only",
        "n_in"=>model.layer1.n_in, "d"=>model.layer1.m_out, "m_out"=>model.layer2.m_out,
        "Q1"=>model.layer1.Q, "Q2"=>model.layer2.Q, "K"=>model.layer1.K
    ))

    opt = SGD(Float64(lr); momentum=0.0)
    reset!(opt)

    best_val_acc = -1.0
    best_epoch = 0

    for epoch in 1:epochs
        tic!(lg)

        Ltr, g1A, g1B, g2A, g2B = grads_two_layer_affine(model, Xtr, ytr)

        # Update as blocks: A1,B1,A2,B2
        pA1 = copy(vec(model.layer1.A)); pB1 = copy(vec(model.layer1.B))
        pA2 = copy(vec(model.layer2.A)); pB2 = copy(vec(model.layer2.B))
        params = [pA1, pB1, pA2, pB2]
        grads  = [collect(vec(g1A)), collect(vec(g1B)), collect(vec(g2A)), collect(vec(g2B))]
        sgd_step!(opt, params, grads)

        model.layer1.A .= reshape(params[1], size(model.layer1.A))
        model.layer1.B .= reshape(params[2], size(model.layer1.B))
        model.layer2.A .= reshape(params[3], size(model.layer2.A))
        model.layer2.B .= reshape(params[4], size(model.layer2.B))

        acc_tr = accuracy(model, Xtr, ytr)
        acc_va = accuracy(model, Xva, yva)
        acc_te = accuracy(model, Xte, yte)

        Lva = value(loss_batch(model, Xva, yva))
        Lte = value(loss_batch(model, Xte, yte))

        t_epoch = toc!(lg)

        log_epoch!(lg; epoch=epoch,
            loss_train=Ltr, acc_train=acc_tr,
            loss_val=Lva,   acc_val=acc_va,
            loss_test=Lte,  acc_test=acc_te,
            time_epoch_s=t_epoch)

        if acc_va > best_val_acc
            best_val_acc = acc_va
            best_epoch = epoch
        end
    end

    write_summary!(lg, "best_val_acc=$(best_val_acc)\nbest_epoch=$(best_epoch)\nrun_dir=$(run_dir)")
    return lg.run_dir
end

end # module Trainer2Affine