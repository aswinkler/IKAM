# src/train/TrainInterlayerFullABPhiC
module InterTrainerFullABPhiC

using Random

using ..AD: Dual, value, deriv, seed!
using ..Inner: PWLinearNodal
using ..InterLayers
using ..InterModel: OneLayerInterModel, loss_batch
using ..Train: SGD, reset!, sgd_step!
using ..Utils: RunLogger, new_run!, log_epoch!, write_summary!, tic!, toc!
using ..Metrics: accuracy_from_logits, cross_entropy_from_logits

export train_interlayer_full_AB_phi_C!

# ------------------------------------------------------------
# helpers: clipping (idéntico estilo a InterTrainerPhiPlusC)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# helper: evaluate PWLinearNodal with override values (Float or Dual)
# (idéntico a InterTrainerPhiPlusC)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# indexing: phi & C (idéntico orden que tu InterTrainerPhiPlusC)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# regularizers (idéntico a InterTrainerPhiPlusC)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# forward for one sample using:
# - A,B from external vectors (Float or Dual)
# - phi values from phi_vec (Float or Dual)
# - C from C_vec (Float or Dual)
# - λ provided externally (scalar Float)
#
# IMPORTANT: Mantiene la normalización por |I2|
# y mantiene EXACTAMENTE el orden de bloques de phi y C.
# ------------------------------------------------------------
function forward_with_ABphiCvec(layer::InterLayers.KAInterLayer{T},
                                x::AbstractVector{T},
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

    # A_vec, B_vec son vec(layer.A), vec(layer.B) con orden columna (Julia)
    # índice lineal para (i,q) en matriz (m_out × Q): i + (q-1)*m_out
    @inline linAQ(i,q) = i + (q-1)*layer.m_out

    @inbounds for i in 1:layer.m_out
        acc_i = z0
        for q in 1:layer.Q

            # s = sum_p φ_{i,q,p}(x_p)
            s = z0
            for p in 1:layer.n_in
                base_phi = (((i-1)*layer.Q + (q-1))*layer.n_in + (p-1))*Kp1
                vals_override = view(phi_vec, base_phi+1:base_phi+Kp1)
                vp = evaluate_with_values(layer.phi[i,q,p], x[p], vals_override)
                φvals[p] = vp
                s += vp
            end

            # affine term (A,B from external vectors)
            a = A_vec[linAQ(i,q)]
            b = B_vec[linAQ(i,q)]
            acc_i += a * s + b

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

# ------------------------------------------------------------
# MAIN TRAINER: full AB + phi + C
# - θ = [vec(A); vec(B); phi_vec; C_vec]
# - λ schedule externo (igual que Phase 3B)
# - regularización sobre phi y C (igual que Phase 3B)
# ------------------------------------------------------------
function train_interlayer_full_AB_phi_C!(m::OneLayerInterModel{T}, split;
    epochs::Int=200,
    lr::Real=0.03,
    seed::Int=123,
    run_tag::String="iris_interlayer_full_AB_phi_C",
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
        "phase"=>"full_AB_phi_C",
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

    # indices
    phi_idxs, C_idxs = build_phi_C_indices(layer)

    # pack/unpack θ = [A; B; phi; C]
    function pack_theta(layer)::Vector{T}
        vA   = vec(copy(layer.A))
        vB   = vec(copy(layer.B))
        vphi = pack_phi(layer, phi_idxs)
        vC   = pack_C(layer, C_idxs)
        return vcat(vA, vB, vphi, vC)
    end

    function unpack_theta!(layer, θ::Vector{T})
        PA = length(layer.A)
        PB = length(layer.B)
        Pphi = length(phi_idxs)
        PC = length(C_idxs)

        @assert length(θ) == PA + PB + Pphi + PC

        # A,B
        layer.A .= reshape(view(θ, 1:PA), size(layer.A))
        layer.B .= reshape(view(θ, PA+1:PA+PB), size(layer.B))

        # phi,C
        off = PA + PB
        unpack_phi!(layer, phi_idxs, collect(view(θ, off+1:off+Pphi)))
        unpack_C!(layer, C_idxs,   collect(view(θ, off+Pphi+1:off+Pphi+PC)))
        return nothing
    end

    # optimizer over θ
    opt = SGD(T(lr); momentum=0.0)
    reset!(opt)

    best_val_acc = -1.0
    best_epoch   = 0

    # epoch 0 evaluation (con λ=0 porque aún no hay schedule aplicado)
    # Para consistencia con tus otros trainers, empezamos a loguear desde epoch=0 con λe=0.
    layer0 = InterLayers.with_lambda(layer, 0.0)
    logits_tr0 = InterLayers.forward_batch(layer0, Xtr)
    logits_va0 = InterLayers.forward_batch(layer0, Xva)
    logits_te0 = InterLayers.forward_batch(layer0, Xte)

    acc_tr0 = accuracy_from_logits(logits_tr0, ytr)
    acc_va0 = accuracy_from_logits(logits_va0, yva)
    acc_te0 = accuracy_from_logits(logits_te0, yte)

    m0 = OneLayerInterModel(layer0)
    Ltr0 = value(loss_batch(m0, Xtr, ytr))
    Lva0 = value(loss_batch(m0, Xva, yva))
    Lte0 = value(loss_batch(m0, Xte, yte))

    log_epoch!(lg; epoch=0,
        loss_train=Ltr0, acc_train=acc_tr0,
        loss_val=Lva0,   acc_val=acc_va0,
        loss_test=Lte0,  acc_test=acc_te0,
        time_epoch_s=0.0
    )
    best_val_acc = acc_va0
    best_epoch   = 0

    # sizes for θ split
    PA   = length(layer.A)
    PB   = length(layer.B)
    Pphi = length(phi_idxs)
    PC   = length(C_idxs)
    Ptot = PA + PB + Pphi + PC

    for epoch in 1:epochs
        tic!(lg)

        # λ schedule (igual que Phase 3B)
        λe = T(λ_max) * T(min(1.0, epoch / max(warmup_epochs, 1)))

        # build θ0
        θ0 = pack_theta(layer)
        @assert length(θ0) == Ptot

        # seed all params (vector-mode)
        θd = Vector{Dual{T}}(undef, Ptot)
        @inbounds for j in 1:Ptot
            θd[j] = seed!(T(θ0[j]), j, Ptot)
        end

        # split θd
        A_vec_d   = view(θd, 1:PA)
        B_vec_d   = view(θd, PA+1:PA+PB)
        phi_vec_d = view(θd, PA+PB+1:PA+PB+Pphi)
        C_vec_d   = view(θd, PA+PB+Pphi+1:Ptot)

        # dual loss over batch
        N = size(Xtr, 2)
        Lacc = zero(phi_vec_d[1])
        @inbounds for n in 1:N
            xn = @view Xtr[:, n]
            logits = forward_with_ABphiCvec(layer, xn, A_vec_d, B_vec_d, phi_vec_d, C_vec_d, λe)
            Lacc += cross_entropy_from_logits(logits, ytr[n])
        end
        Ldata = Lacc / N

        # regularization (mismo que Phase 3B)
        Lsmooth = smoothness_penalty(phi_vec_d, layer)
        LC      = l2_penalty(C_vec_d)

        L = Ldata + T(alpha_smooth)*Lsmooth + T(beta_C)*LC

        g = deriv(L)
        @assert length(g) == Ptot

        clip_by_value!(g, T(clip_value))

        # SGD update in Float space
        θ = θ0
        params = [θ]
        grads  = [g]
        sgd_step!(opt, params, grads)
        unpack_theta!(layer, params[1])

        # metrics evaluated with layer.λ = λe (igual que Phase 3B)
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
    end

    write_summary!(lg, "best_val_acc=$(best_val_acc)\nbest_epoch=$(best_epoch)")
    return lg, best_val_acc, best_epoch
end

end # module InterTrainerFullABPhiC