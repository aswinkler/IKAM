module Trainer2Full

using Random
using LinearAlgebra

using IKAM

using ..AD: Dual, seed!, value, deriv
using ..Inner: PWLinearNodal
using ..Layers: KALayer
using ..Model2: TwoLayerModel, loss_batch, accuracy
using ..InterLayer: InterLayerMinMax, fit_interlayer_minmax, transform_interlayer
using ..Train: SGD, sgd_step!, reset!
using ..Utils: RunLogger, new_run!, log_epoch!, write_summary!, tic!, toc!

export train_two_layer_full!

const_dual(x::T, P::Int) where {T<:Real} = Dual{T}(x, zeros(T, P))

nparams_layer_full(layer::KALayer) =
    length(layer.A) + length(layer.B) + layer.m_out * layer.Q * layer.n_in * (layer.K+1)

function lift_layer_full_dual(layer::KALayer{T}, offset::Int, P::Int) where {T<:Real}
    m_out, Q = size(layer.A)
    n_in = layer.n_in
    K = layer.K

    A_dual = Matrix{Dual{T}}(undef, m_out, Q)
    B_dual = Matrix{Dual{T}}(undef, m_out, Q)

    idx = offset
    @inbounds for i in 1:m_out, q in 1:Q
        idx += 1
        A_dual[i,q] = seed!(layer.A[i,q], idx, P)
    end
    @inbounds for i in 1:m_out, q in 1:Q
        idx += 1
        B_dual[i,q] = seed!(layer.B[i,q], idx, P)
    end

    phi_dual = Array{PWLinearNodal{Dual{T}},3}(undef, m_out, Q, n_in)
    @inbounds for i in 1:m_out, q in 1:Q, p in 1:n_in
        pw = layer.phi[i,q,p]
        knots_d = [const_dual(k, P) for k in pw.knots]
        vals_d  = Vector{Dual{T}}(undef, K+1)
        for k in 1:(K+1)
            idx += 1
            vals_d[k] = seed!(pw.values[k], idx, P)
        end
        phi_dual[i,q,p] = PWLinearNodal{Dual{T}}(knots_d, vals_d)
    end

    return KALayer{Dual{T}}(layer.n_in, layer.m_out, layer.Q, layer.K, A_dual, B_dual, phi_dual), idx
end

function clip_by_norm!(g::AbstractVector, maxnorm::Real)
    n = norm(g)
    if n > maxnorm
        g .*= (maxnorm / n)
    end
    return g
end

function apply_full_update!(opt::SGD{T}, layer::KALayer{T}, g::Vector{T}) where {T<:Real}
    # Empaqueta params en el mismo orden: A, B, phi-values (i,q,p,k)
    A_vec = copy(vec(layer.A))
    B_vec = copy(vec(layer.B))

    phi_len = layer.m_out * layer.Q * layer.n_in * (layer.K+1)
    phi_vec = Vector{T}(undef, phi_len)
    idx = 0
    @inbounds for i in 1:layer.m_out, q in 1:layer.Q, p in 1:layer.n_in, k in 1:(layer.K+1)
        idx += 1
        phi_vec[idx] = layer.phi[i,q,p].values[k]
    end

    # grads particionados
    P_A = length(A_vec)
    P_B = length(B_vec)
    gA = g[1:P_A]
    gB = g[P_A+1:P_A+P_B]
    gφ = g[P_A+P_B+1:end]

    params = [A_vec, B_vec, phi_vec]
    grads  = [collect(gA), collect(gB), collect(gφ)]
    sgd_step!(opt, params, grads)

    layer.A .= reshape(params[1], size(layer.A))
    layer.B .= reshape(params[2], size(layer.B))

    idx = 0
    @inbounds for i in 1:layer.m_out, q in 1:layer.Q, p in 1:layer.n_in, k in 1:(layer.K+1)
        idx += 1
        layer.phi[i,q,p].values[k] = params[3][idx]
    end
    return nothing
end

function grads_two_layer_full(model, X::AbstractMatrix{T}, y::AbstractVector{<:Integer}) where {T<:Real}
    P1 = nparams_layer_full(model.layer1)
    P2 = nparams_layer_full(model.layer2)
    P = P1 + P2

    idx = 0
    l1d, idx = lift_layer_full_dual(model.layer1, idx, P)
    l2d, idx = lift_layer_full_dual(model.layer2, idx, P)
    @assert idx == P

    model_d = 
        if hasproperty(model, :scaler)
            # TwoLayerModelScaled(l1, l2, scaler)
            IKAM.Model2.TwoLayerModelScaled(l1d, l2d, getproperty(model, :scaler))
        else
            # TwoLayerModel(l1,l2)
            IKAM.Model2.TwoLayerModel(l1d,l2d)
        end 
    Ld = loss_batch(model_d, X, y)
    g = deriv(Ld)
    @assert length(g) == P

    return value(Ld), g[1:P1], g[P1+1:end]
end

function train_two_layer_full!(model, split;
                               epochs::Int=400,
                               lr1::Real=0.001,
                               lr2::Real=0.01,
                               clip_norm::Real=1.0,
                               seed::Int=0,
                               run_tag::AbstractString="iris_two_layer_full",
                               runs_root::AbstractString="runs")

    Xtr, ytr = split.X_train, split.y_train
    Xva, yva = split.X_val, split.y_val
    Xte, yte = split.X_test, split.y_test

    lg = RunLogger(runs_root=runs_root)
    run_dir = new_run!(lg; tag=run_tag, meta=Dict(
        "seed"=>seed, "epochs"=>epochs, "lr1"=>lr1, "lr2" => lr2,
        "phase"=>"two_layer_full",
        "n_in"=>model.layer1.n_in, "d"=>model.layer1.m_out, "m_out"=>model.layer2.m_out,
        "Q1"=>model.layer1.Q, "Q2"=>model.layer2.Q, "K"=>model.layer1.K
    ))

    opt1 = SGD(Float64(lr1); momentum=0.0)
    opt2 = SGD(Float64(lr2); momentum=0.0)
    
    reset!(opt1)
    reset!(opt2)

    best_val_acc = -1.0
    best_epoch = 0

    for epoch in 1:epochs


        if epoch == 1 || epoch == 50
            s = model.scaler
            println("scaler stored: mins[1:3]=", s.mins[1:3], "  maxs[1:3]=", s.maxs[1:3])

            # Diagnóstico: recomputar el scaler "si se refiteara"
            Znow = IKAM.Layers.forward_batch(model.layer1, Xtr)
            s_now = IKAM.InterLayer.fit_interlayer_minmax(Znow)

            println("scaler refit : mins[1:3]=", s_now.mins[1:3], "  maxs[1:3]=", s_now.maxs[1:3])
            println("delta mins norm=", norm(s_now.mins .- s.mins),
                    "  delta maxs norm=", norm(s_now.maxs .- s.maxs))
        end
        if epoch == 1
            L0 = value(loss_batch(model, Xtr, ytr))
            Ltr, g1, g2 = grads_two_layer_full(model, Xtr, ytr)

            println("L0=", L0, "  Ldual=", Ltr,
                    "  ||g1||=", norm(g1), "  ||g2||=", norm(g2))

            # prueba: aplica UN paso y mide
            apply_full_update!(opt1, model.layer1, g1)
            apply_full_update!(opt2, model.layer2, g2)
            L1 = value(loss_batch(model, Xtr, ytr))
            println("L1(after 1 step)=", L1)
            # return lg.run_dir   # corta aquí para no desperdiciar epochs
        end
        tic!(lg)

        Ltr, g1, g2 = grads_two_layer_full(model, Xtr, ytr)

        clip_by_norm!(g1, clip_norm)
        clip_by_norm!(g2, clip_norm)

        apply_full_update!(opt1, model.layer1, g1)
        apply_full_update!(opt2, model.layer2, g2)

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

end # module Trainer2Full