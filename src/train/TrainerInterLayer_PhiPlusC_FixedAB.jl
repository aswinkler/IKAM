# src/train/TrainerInterLayer_PhiPlusC_FixedAB.jl

module InterTrainerPhiPlusC

using Random
using Statistics

using ..AD: Dual, value, deriv, seed!
using ..Inner: PWLinearNodal
using ..InterLayers
using ..InterModel: OneLayerInterModel, loss_batch
using ..Train: SGD, reset!, sgd_step!
using ..Utils: RunLogger, new_run!, log_epoch!, write_summary!, tic!, toc!
using ..Metrics: accuracy_from_logits, cross_entropy_from_logits

export train_interlayer_phi_plus_C_fixedAB!

# -----------------------------
# helpers: clipping
# -----------------------------
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

# -----------------------------
# helper: evaluate PWLinearNodal with override values (Float or Dual)
# knots stay Float64; values can be Dual
# -----------------------------
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

# -----------------------------
# indexing: pack/unpack phi values and active C
# -----------------------------
"""
Builds:
- phi_idxs: Vector of (i,q,p,k) for k=1..K+1
- C_idxs:   Vector of (i,q,p,r) over I2
"""
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

# -----------------------------
# forward for one sample using:
# - A,B from layer (Float)
# - phi values from phi_vec (Float or Dual)
# - C from C_vec (Float or Dual)
# - λ provided externally (scalar Float)
# Normalization by number of pairs included
# -----------------------------
function forward_with_phiCvec(layer::InterLayers.KAInterLayer{T},
                              x::AbstractVector{T},
                              phi_vec,
                              C_vec,
                              phi_idxs,
                              C_idxs,
                              λ::T) where {T<:Real}

    # zero with correct tangent dimension (Dual-safe)
    z0 = phi_vec[1] - phi_vec[1]
    Tout = typeof(z0)

    out   = Vector{Tout}(undef, layer.m_out)
    φvals = Vector{Tout}(undef, layer.n_in)

    # fast access: for each (i,q,p) we need its K+1 values
    # We'll compute base offset because phi_idxs is ordered by i,q,p,k
    Kp1 = layer.K + 1

    # C_idxs is ordered by i,q, then I2 order, so we can compute base similarly
    Lpairs = length(layer.I2)

    @inbounds for i in 1:layer.m_out
        acc_i = z0
        for q in 1:layer.Q

            # s = sum_p φ_{i,q,p}(x_p)
            s = z0
            for p in 1:layer.n_in
                # base index for (i,q,p,1)
                base_phi = (((i-1)*layer.Q + (q-1))*layer.n_in + (p-1))*Kp1
                vals_override = view(phi_vec, base_phi+1:base_phi+Kp1)
                vp = evaluate_with_values(layer.phi[i,q,p], x[p], vals_override)
                φvals[p] = vp
                s += vp
            end

            # affine term (A,B fixed)
            acc_i += layer.A[i,q] * s + layer.B[i,q]

            # interaction
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

# -----------------------------
# regularizers
# -----------------------------
function smoothness_penalty(phi_vec, layer)::Any
    # sum over each local PW: sum_k (v_{k+1}-v_k)^2
    Kp1 = layer.K + 1
    z0 = phi_vec[1] - phi_vec[1]
    acc = z0
    # phi_vec grouped by blocks of size Kp1
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

# -----------------------------
# main: Phase 3B trainer
# -----------------------------
"""
    train_interlayer_phi_plus_C_fixedAB!(model, split; ...)

Phase 3B:
- A,B fixed
- train phi values and C (active entries from I2)
- λ is scheduled via ramp: λ_e = λ_max * min(1, epoch / warmup_epochs)
- penalties:
    alpha_smooth * smoothness(phi)
    beta_C       * ||C||^2
"""
function train_interlayer_phi_plus_C_fixedAB!(m::OneLayerInterModel{T}, split;
    epochs::Int=200,
    lr::Real=0.05,
    seed::Int=123,
    run_tag::String="iris_interlayer_phi_plus_C_fixedAB",
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
        "phase"=>"phi_plus_C_fixedAB",
        "lambda_max"=>λ_max, "warmup_epochs"=>warmup_epochs,
        "alpha_smooth"=>alpha_smooth, "beta_C"=>beta_C,
        "clip_value"=>clip_value,
        "n_in"=>m.layer.n_in, "m_out"=>m.layer.m_out, "Q"=>m.layer.Q, "K"=>m.layer.K,
        "I2_size"=>length(m.layer.I2)
    ))

    Xtr, ytr = split.X_train, split.y_train
    Xva, yva = split.X_val,   split.y_val
    Xte, yte = split.X_test,  split.y_test

    layer = m.layer

    # indices for trainable params
    phi_idxs, C_idxs = build_phi_C_indices(layer)

    @show length(phi_idxs)
    @show length(C_idxs)
    @show (length(phi_idxs) + length(C_idxs))


    # parameter vector θ = [phi_vec; C_vec]
    function pack_theta(layer)
        vphi = pack_phi(layer, phi_idxs)
        vC   = pack_C(layer, C_idxs)
        return vcat(vphi, vC)
    end

    function unpack_theta!(layer, θ::Vector{T})
        Pphi = length(phi_idxs)
        vphi = view(θ, 1:Pphi)
        vC   = view(θ, Pphi+1:length(θ))
        unpack_phi!(layer, phi_idxs, collect(vphi))
        unpack_C!(layer, C_idxs, collect(vC))
        return nothing
    end

    # optimizer over θ
    opt = SGD(T(lr); momentum=0.0)
    reset!(opt)

    best_val_acc = -1.0
    best_epoch   = 0

    Pphi = length(phi_idxs)
    PC   = length(C_idxs)
    Ptot = Pphi + PC

    # antes
    phi_norm0 = 0.0
    for ph in m.layer.phi
        phi_norm0 += sum(abs2, ph.values)
    end
    C_norm0 = sum(abs2, m.layer.C)    

    for epoch in 1:epochs
        tic!(lg)

        # λ schedule (ramp)
        λe = T(λ_max) * T(min(1.0, epoch / max(warmup_epochs, 1)))

        # --- build θ0 ---
        θ0 = pack_theta(layer)
        @assert length(θ0) == Ptot

        # --- seed all parameters (vector-mode forward AD) ---
        θd = Vector{Dual{T}}(undef, Ptot)
        @inbounds for j in 1:Ptot
            θd[j] = seed!(T(θ0[j]), j, Ptot)
        end

        phi_vec_d = view(θd, 1:Pphi)
        C_vec_d   = view(θd, Pphi+1:Ptot)

        # --- dual loss over batch (uses fixed A,B, trainable phi/C) ---
        N = size(Xtr, 2)
        Lacc = zero(phi_vec_d[1])
        @inbounds for n in 1:N
            xn = @view Xtr[:, n]
            logits = forward_with_phiCvec(layer, xn, phi_vec_d, C_vec_d, phi_idxs, C_idxs, λe)
            Lacc += cross_entropy_from_logits(logits, ytr[n])
        end
        Ldata = Lacc / N

        # regularization
        Lsmooth = smoothness_penalty(phi_vec_d, layer)
        LC = l2_penalty(C_vec_d)

        L = Ldata + T(alpha_smooth)*Lsmooth + T(beta_C)*LC

        # gradient is deriv(L)::Vector{T} of length Ptot
        g = deriv(L)
        @assert length(g) == Ptot

        # --- DEBUG AD DIM ---
        @show λe
        @show value(Ldata) value(Lsmooth) value(LC) value(L)
        @show length(deriv(L)) maximum(abs.(deriv(L)))

        # --------------------

        # optional clipping
        clip_by_value!(g, T(clip_value))

        # update θ in Float space
        θ = θ0
        params = [θ]
        grads  = [g]
        sgd_step!(opt, params, grads)
        unpack_theta!(layer, params[1])

        # metrics (Float)
        # note: we DO NOT use layer.λ here; we evaluate model as defined.
        # For consistent logging with λe, you can reconstruct a model via with_lambda if you want,
        # but training itself used λe inside forward_with_phiCvec.
        layer_eval = InterLayers.with_lambda(layer, λe)

        logits_tr = InterLayers.forward_batch(layer_eval, Xtr)
        logits_va = InterLayers.forward_batch(layer_eval, Xva)
        logits_te = InterLayers.forward_batch(layer_eval, Xte)

        acc_tr = accuracy_from_logits(logits_tr, ytr)
        acc_va = accuracy_from_logits(logits_va, yva)
        acc_te = accuracy_from_logits(logits_te, yte)

        m_eval = OneLayerInterModel(layer_eval)

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
        @show λe mean(abs.(view(g, Pphi+1:Ptot)))  # grad promedio de C
        @show λe mean(abs.(view(g, 1:Pphi)))      # grad promedio de phi
    end

    # después
    phi_norm1 = 0.0
    for ph in m.layer.phi
        phi_norm1 += sum(abs2, ph.values)
    end
    C_norm1 = sum(abs2, m.layer.C)

    @show phi_norm0 phi_norm1 (phi_norm1 - phi_norm0)
    @show C_norm0 C_norm1 (C_norm1 - C_norm0)  

    write_summary!(lg, "best_val_acc=$(best_val_acc)\nbest_epoch=$(best_epoch)")
    return lg, best_val_acc, best_epoch
end

end # module InterTrainerPhiPlusC