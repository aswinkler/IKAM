# src/train/TrainTwoInterlayerFullABPhiC.jl
module InterTrainerTwoLayerFullABPhiC

using Random

using ..AD: Dual, value, deriv, seed!
using ..Inner: PWLinearNodal
using ..InterLayers
using ..TwoInterModel: TwoLayerInterModel, fit_hidden_scaler, forward_batch, loss_batch
using ..Train: SGD, reset!, sgd_step!
using ..Utils: RunLogger, new_run!, log_epoch!, write_summary!, tic!, toc!
using ..Metrics: accuracy_from_logits, cross_entropy_from_logits
using ..Data

export train_two_interlayer_full_AB_phi_C!

function clip_by_value!(g::AbstractVector{T}, clip::T) where {T<:Real}
    @inbounds for i in eachindex(g)
        if g[i] > clip
            g[i] = clip
        elseif g[i] < -clip
            g[i] = -clip
        end
    end
    return g
end

function _segment_index(knots::Vector{T}, x) where {T<:Real}
    xv = value(x)
    if xv <= knots[1]
        return 1
    elseif xv >= knots[end]
        return length(knots) - 1
    else
        i = searchsortedlast(knots, T(xv))
        return clamp(i, 1, length(knots)-1)
    end
end

function evaluate_with_values(pw::PWLinearNodal{T}, x, vals_override) where {T<:Real}
    k  = _segment_index(pw.knots, x)
    x0 = pw.knots[k]
    x1 = pw.knots[k+1]
    y0 = vals_override[k]
    y1 = vals_override[k+1]
    t  = (x - x0) / (x1 - x0)
    return y0 + (y1 - y0) * t
end

@inline function _clamp01_same(u)
    uv = value(u)
    if uv < 0
        return zero(u)
    elseif uv > 1
        return one(u)
    else
        return u
    end
end

function transform_hidden_vec(s::Data.MinMaxScaler{T}, z::AbstractVector; clamp01::Bool=true, eps::Real=1e-12) where {T<:Real}
    @assert length(z) == length(s.mins) == length(s.maxs)
    out = Vector{typeof(z[1])}(undef, length(z))
    @inbounds for i in eachindex(z)
        mn = s.mins[i]
        mx = s.maxs[i]
        den = mx - mn
        if den == zero(T)
            out[i] = zero(z[i])
        else
            v = (z[i] - mn) / (den + T(eps))
            out[i] = clamp01 ? _clamp01_same(v) : v
        end
    end
    return out
end

function build_phi_C_indices(layer)
    Kp1 = layer.K + 1

    phi_idxs = Vector{NTuple{4,Int}}()
    sizehint!(phi_idxs, layer.m_out * layer.Q * layer.n_in * Kp1)
    @inbounds for i in 1:layer.m_out, q in 1:layer.Q, p in 1:layer.n_in, k in 1:Kp1
        push!(phi_idxs, (i,q,p,k))
    end

    C_idxs = Vector{NTuple{4,Int}}()
    sizehint!(C_idxs, layer.m_out * layer.Q * length(layer.I2))
    @inbounds for i in 1:layer.m_out, q in 1:layer.Q
        for (p,r) in layer.I2
            push!(C_idxs, (i,q,p,r))
        end
    end

    return phi_idxs, C_idxs
end

function pack_phi(layer::InterLayers.KAInterLayer{T}, phi_idxs)::Vector{T} where {T<:Real}
    θ = Vector{T}(undef, length(phi_idxs))
    @inbounds for t in 1:length(phi_idxs)
        (i,q,p,k) = phi_idxs[t]
        θ[t] = layer.phi[i,q,p].values[k]
    end
    return θ
end

function unpack_phi!(layer::InterLayers.KAInterLayer{T}, phi_idxs, θ::AbstractVector{T}) where {T<:Real}
    @assert length(θ) == length(phi_idxs)
    @inbounds for t in 1:length(phi_idxs)
        (i,q,p,k) = phi_idxs[t]
        layer.phi[i,q,p].values[k] = θ[t]
    end
    return nothing
end

function pack_C(layer::InterLayers.KAInterLayer{T}, C_idxs)::Vector{T} where {T<:Real}
    θ = Vector{T}(undef, length(C_idxs))
    @inbounds for t in 1:length(C_idxs)
        (i,q,p,r) = C_idxs[t]
        θ[t] = layer.C[i,q,p,r]
    end
    return θ
end

function unpack_C!(layer::InterLayers.KAInterLayer{T}, C_idxs, θ::AbstractVector{T}) where {T<:Real}
    @assert length(θ) == length(C_idxs)
    @inbounds for t in 1:length(C_idxs)
        (i,q,p,r) = C_idxs[t]
        layer.C[i,q,p,r] = θ[t]
    end
    return nothing
end

function smoothness_penalty(phi_vec, layer)::Any
    Kp1 = layer.K + 1
    z0 = phi_vec[1] - phi_vec[1]
    acc = z0
    nb = length(phi_vec) ÷ Kp1
    @inbounds for b in 0:nb-1
        base = b*Kp1
        for k in 1:Kp1-1
            d = phi_vec[base+k+1] - phi_vec[base+k]
            acc += d*d
        end
    end
    return acc
end

function l2_penalty(C_vec)::Any
    z0 = C_vec[1] - C_vec[1]
    acc = z0
    @inbounds for j in eachindex(C_vec)
        acc += C_vec[j]*C_vec[j]
    end
    return acc
end

function forward_with_ABphiCvec(layer::InterLayers.KAInterLayer{T},
                                x::AbstractVector,
                                A_vec,
                                B_vec,
                                phi_vec,
                                C_vec,
                                λ::T) where {T<:Real}

    z0 = phi_vec[1] - phi_vec[1]
    Tout = typeof(z0)

    out   = Vector{Tout}(undef, layer.m_out)
    φvals = Vector{Tout}(undef, layer.n_in)

    Kp1    = layer.K + 1
    Lpairs = length(layer.I2)

    @inline linAQ(i,q) = i + (q-1)*layer.m_out

    @inbounds for i in 1:layer.m_out
        acc_i = z0
        for q in 1:layer.Q
            s = z0
            for p in 1:layer.n_in
                base_phi = (((i-1)*layer.Q + (q-1))*layer.n_in + (p-1))*Kp1
                vals_override = view(phi_vec, base_phi+1:base_phi+Kp1)
                vp = evaluate_with_values(layer.phi[i,q,p], x[p], vals_override)
                φvals[p] = vp
                s += vp
            end

            a = A_vec[linAQ(i,q)]
            b = B_vec[linAQ(i,q)]
            acc_i += a * s + b

            if λ != zero(T)
                inter = z0
                base_C = ((i-1)*layer.Q + (q-1))*Lpairs
                for t in 1:Lpairs
                    (p,r) = layer.I2[t]
                    inter += C_vec[base_C + t] * φvals[p] * φvals[r]
                end
                if Lpairs > 0
                    inter /= T(Lpairs)
                end
                acc_i += λ * inter
            end
        end
        out[i] = acc_i
    end

    return out
end

function hidden_scaler_from_layer(layer1::InterLayers.KAInterLayer{T},
                                  Xtr::AbstractMatrix{T},
                                  λe::T) where {T<:Real}
    layer1_eval = InterLayers.with_lambda(layer1, λe)
    return fit_hidden_scaler(layer1_eval, Xtr)
end

function train_two_interlayer_full_AB_phi_C!(m::TwoLayerInterModel{T}, split;
    epochs::Int=200,
    lr::Real=0.03,
    seed::Int=123,
    run_tag::String="iris_two_interlayer_full_AB_phi_C",
    runs_root::String="runs",
    λ_max::Real=20.0,
    warmup_epochs::Int=20,
    alpha_smooth::Real=1e-3,
    beta_C::Real=1e-3,
    clip_value::Real=3.0
) where {T<:Real}

    rng = MersenneTwister(seed)

    lg = RunLogger(runs_root=runs_root)
    new_run!(lg; tag=run_tag, meta=Dict(
        "seed"=>seed, "epochs"=>epochs, "lr"=>lr,
        "phase"=>"two_layer_full_AB_phi_C",
        "lambda_max"=>λ_max, "warmup_epochs"=>warmup_epochs,
        "alpha_smooth"=>alpha_smooth, "beta_C"=>beta_C,
        "clip_value"=>clip_value,
        "layer1_n_in"=>m.layer1.n_in, "layer1_m_out"=>m.layer1.m_out, "layer1_Q"=>m.layer1.Q, "layer1_K"=>m.layer1.K,
        "layer1_I2_size"=>length(m.layer1.I2),
        "layer2_n_in"=>m.layer2.n_in, "layer2_m_out"=>m.layer2.m_out, "layer2_Q"=>m.layer2.Q, "layer2_K"=>m.layer2.K,
        "layer2_I2_size"=>length(m.layer2.I2)
    ))

    Xtr, ytr = split.X_train, split.y_train
    Xva, yva = split.X_val,   split.y_val
    Xte, yte = split.X_test,  split.y_test

    layer1 = m.layer1
    layer2 = m.layer2

    phi1_idxs, C1_idxs = build_phi_C_indices(layer1)
    phi2_idxs, C2_idxs = build_phi_C_indices(layer2)

    function pack_theta_layer(layer, phi_idxs, C_idxs)::Vector{T}
        vA   = vec(copy(layer.A))
        vB   = vec(copy(layer.B))
        vphi = pack_phi(layer, phi_idxs)
        vC   = pack_C(layer, C_idxs)
        return vcat(vA, vB, vphi, vC)
    end

    function unpack_theta_layer!(layer, phi_idxs, C_idxs, θ::Vector{T})
        PA = length(layer.A)
        PB = length(layer.B)
        Pphi = length(phi_idxs)
        PC = length(C_idxs)

        @assert length(θ) == PA + PB + Pphi + PC

        layer.A .= reshape(view(θ, 1:PA), size(layer.A))
        layer.B .= reshape(view(θ, PA+1:PA+PB), size(layer.B))

        off = PA + PB
        unpack_phi!(layer, phi_idxs, collect(view(θ, off+1:off+Pphi)))
        unpack_C!(layer, C_idxs,   collect(view(θ, off+Pphi+1:off+Pphi+PC)))
        return nothing
    end

    function pack_theta_all()::Vector{T}
        return vcat(pack_theta_layer(layer1, phi1_idxs, C1_idxs),
                    pack_theta_layer(layer2, phi2_idxs, C2_idxs))
    end

    function unpack_theta_all!(θ::Vector{T})
        P1 = length(pack_theta_layer(layer1, phi1_idxs, C1_idxs))
        θ1 = collect(view(θ, 1:P1))
        θ2 = collect(view(θ, P1+1:length(θ)))
        unpack_theta_layer!(layer1, phi1_idxs, C1_idxs, θ1)
        unpack_theta_layer!(layer2, phi2_idxs, C2_idxs, θ2)
        return nothing
    end

    opt = SGD(T(lr); momentum=0.0)
    reset!(opt)

    best_val_acc = -1.0
    best_epoch   = 0

    λ0 = zero(T)
    scaler0 = hidden_scaler_from_layer(layer1, Xtr, λ0)
    model0 = TwoLayerInterModel(InterLayers.with_lambda(layer1, λ0),
                                InterLayers.with_lambda(layer2, λ0),
                                scaler0)

    logits_tr0 = forward_batch(model0, Xtr)
    logits_va0 = forward_batch(model0, Xva)
    logits_te0 = forward_batch(model0, Xte)

    acc_tr0 = accuracy_from_logits(logits_tr0, ytr)
    acc_va0 = accuracy_from_logits(logits_va0, yva)
    acc_te0 = accuracy_from_logits(logits_te0, yte)

    Ltr0 = value(loss_batch(model0, Xtr, ytr))
    Lva0 = value(loss_batch(model0, Xva, yva))
    Lte0 = value(loss_batch(model0, Xte, yte))

    log_epoch!(lg; epoch=0,
        loss_train=Ltr0, acc_train=acc_tr0,
        loss_val=Lva0,   acc_val=acc_va0,
        loss_test=Lte0,  acc_test=acc_te0,
        time_epoch_s=0.0
    )
    best_val_acc = acc_va0
    best_epoch   = 0

    θ1_0 = pack_theta_layer(layer1, phi1_idxs, C1_idxs)
    θ2_0 = pack_theta_layer(layer2, phi2_idxs, C2_idxs)

    P1 = length(θ1_0)
    P2 = length(θ2_0)
    Ptot = P1 + P2

    PA1   = length(layer1.A)
    PB1   = length(layer1.B)
    Pphi1 = length(phi1_idxs)
    PC1   = length(C1_idxs)

    PA2   = length(layer2.A)
    PB2   = length(layer2.B)
    Pphi2 = length(phi2_idxs)
    PC2   = length(C2_idxs)

    scaler_update_every::Int=1
    λ0 = zero(T)
    scaler_hidden = hidden_scaler_from_layer(layer1, Xtr, λ0)

    for epoch in 1:epochs
        tic!(lg)

        λe = T(λ_max) * T(min(1.0, epoch / max(warmup_epochs, 1)))
        #if epoch == 1 || ((epoch - 1) % scaler_update_every == 0)
            scaler_hidden = hidden_scaler_from_layer(layer1, Xtr, λe)
        #end

        θ0 = pack_theta_all()
        @assert length(θ0) == Ptot

        θd = Vector{Dual{T}}(undef, Ptot)
        @inbounds for j in 1:Ptot
            θd[j] = seed!(T(θ0[j]), j, Ptot)
        end

        θ1d = view(θd, 1:P1)
        θ2d = view(θd, P1+1:Ptot)

        A1d   = view(θ1d, 1:PA1)
        B1d   = view(θ1d, PA1+1:PA1+PB1)
        phi1d = view(θ1d, PA1+PB1+1:PA1+PB1+Pphi1)
        C1d   = view(θ1d, PA1+PB1+Pphi1+1:P1)

        A2d   = view(θ2d, 1:PA2)
        B2d   = view(θ2d, PA2+1:PA2+PB2)
        phi2d = view(θ2d, PA2+PB2+1:PA2+PB2+Pphi2)
        C2d   = view(θ2d, PA2+PB2+Pphi2+1:P2)

        N = size(Xtr, 2)
        Lacc = zero(phi1d[1])

        @inbounds for n in 1:N
            xn = @view Xtr[:, n]

            z1  = forward_with_ABphiCvec(layer1, xn, A1d, B1d, phi1d, C1d, λe)
            z1n = transform_hidden_vec(scaler_hidden, z1; clamp01=true)
            ŷn  = forward_with_ABphiCvec(layer2, z1n, A2d, B2d, phi2d, C2d, λe)

            Lacc += cross_entropy_from_logits(ŷn, ytr[n])
        end

        Ldata = Lacc / N

        Lsmooth =
            smoothness_penalty(phi1d, layer1) +
            smoothness_penalty(phi2d, layer2)

        LC =
            l2_penalty(C1d) +
            l2_penalty(C2d)

        L = Ldata + T(alpha_smooth)*Lsmooth + T(beta_C)*LC

        g = deriv(L)
        @assert length(g) == Ptot

        clip_by_value!(g, T(clip_value))

        θ = θ0
        params = [θ]
        grads  = [g]
        sgd_step!(opt, params, grads)
        unpack_theta_all!(params[1])

        layer1_eval = InterLayers.with_lambda(layer1, λe)
        layer2_eval = InterLayers.with_lambda(layer2, λe)
        scaler_eval = hidden_scaler_from_layer(layer1, Xtr, λe)

        m_eval = TwoLayerInterModel(layer1_eval, layer2_eval, scaler_eval)

        logits_tr = forward_batch(m_eval, Xtr)
        logits_va = forward_batch(m_eval, Xva)
        logits_te = forward_batch(m_eval, Xte)

        acc_tr = accuracy_from_logits(logits_tr, ytr)
        acc_va = accuracy_from_logits(logits_va, yva)
        acc_te = accuracy_from_logits(logits_te, yte)

        Ltr = value(loss_batch(m_eval, Xtr, ytr))
        Lva = value(loss_batch(m_eval, Xva, yva))
        Lte = value(loss_batch(m_eval, Xte, yte))

        t = toc!(lg)

        log_epoch!(lg;
            epoch=epoch,
            loss_train=Ltr, acc_train=acc_tr,
            loss_val=Lva,   acc_val=acc_va,
            loss_test=Lte,  acc_test=acc_te,
            time_epoch_s=t
        )

        if acc_va > best_val_acc
            best_val_acc = acc_va
            best_epoch   = epoch
        end
    end

    write_summary!(lg, "best_val_acc=$(best_val_acc)\nbest_epoch=$(best_epoch)")
    return lg, best_val_acc, best_epoch
end

end # module InterTrainerTwoLayerFullABPhiC