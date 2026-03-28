# src/train/TrainerInterLayer_PhiOnly_FixedAB.jl

module InterTrainerPhiOnly

using Random

using ..AD: Dual, value, deriv
using ..InterLayers
using ..InterModel: OneLayerInterModel, forward_batch, loss_batch
using ..Train: SGD, reset!, sgd_step!
using ..Utils: RunLogger, new_run!, log_epoch!, write_summary!, tic!, toc!
using ..Metrics: accuracy_from_logits, cross_entropy_from_logits

export train_interlayer_phi_only_fixedAB!

# -----------------------------
# Helpers: pack/unpack phi values
# -----------------------------
# Order: i, q, p, k  (k = 1..K+1)
function pack_phi(layer::InterLayers.KAInterLayer{T}) where {T<:Real}
    K1 = layer.K + 1
    P  = layer.m_out * layer.Q * layer.n_in * K1
    θ  = Vector{T}(undef, P)
    idx = 0
    @inbounds for i in 1:layer.m_out, q in 1:layer.Q, p in 1:layer.n_in, k in 1:K1
        idx += 1
        θ[idx] = layer.phi[i,q,p].values[k]
    end
    return θ
end

function unpack_phi!(layer::InterLayers.KAInterLayer{T}, θ::AbstractVector{T};
                     clip_value::Real = 0.0) where {T<:Real}
    K1 = layer.K + 1
    P  = layer.m_out * layer.Q * layer.n_in * K1
    @assert length(θ) == P
    idx = 0
    @inbounds for i in 1:layer.m_out, q in 1:layer.Q, p in 1:layer.n_in, k in 1:K1
        idx += 1
        v = θ[idx]
        if clip_value > 0
            v = clamp(v, -T(clip_value), T(clip_value))
        end
        layer.phi[i,q,p].values[k] = v
    end
    return nothing
end

# -----------------------------
# PWLinear evaluation with external "values"
# -----------------------------
# knots are Float64, x may be Float64, and values_ext may be Float64 or Dual.
# We decide segment using value(x) (compatible with your design).
function _segment_index(knots::AbstractVector{T}, x) where {T<:Real}
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

function eval_pw_external(knots, values_ext, x)
    k = _segment_index(knots, x)
    x0 = knots[k]
    x1 = knots[k+1]
    y0 = values_ext[k]
    y1 = values_ext[k+1]
    t  = (x - x0) / (x1 - x0)
    return y0 + (y1 - y0) * t
end

# -----------------------------
# Regularization: smoothness on nodal values
# sum (v[k+1]-v[k])^2 averaged
# -----------------------------
function smoothness_penalty(phi_vec, K1::Int)
    # phi_vec is packed over all (i,q,p) blocks, each block length K1
    # returns same scalar type (Float64 or Dual) as elements
    z0 = phi_vec[1] - phi_vec[1]
    acc = z0
    P = length(phi_vec)
    nb = div(P, K1)
    @inbounds for b in 0:(nb-1)
        base = b*K1
        for k in 1:(K1-1)
            d = phi_vec[base+k+1] - phi_vec[base+k]
            acc += d*d
        end
    end
    # normalize (optional): per block, per edge
    return acc / (nb*(K1-1))
end

# -----------------------------
# Forward with external phi vector
# λ is OFF in this phase (we force it to 0 at the top-level)
# A,B are taken from layer (Float64), C ignored because λ=0
# -----------------------------
function forward_with_phivec(layer::InterLayers.KAInterLayer{T},
                             x::AbstractVector,
                             phi_vec) where {T<:Real}
    @assert length(x) == layer.n_in

    # zero of correct tangent type
    z0 = phi_vec[1] - phi_vec[1]
    proto = z0 + layer.A[1,1] + layer.B[1,1]
    Tout = typeof(proto)

    out = Vector{Tout}(undef, layer.m_out)

    K1 = layer.K + 1
    # offset for block (i,q,p): ((i-1)*Q*n_in + (q-1)*n_in + (p-1))*K1
    @inbounds for i in 1:layer.m_out
        acc_i = z0
        for q in 1:layer.Q
            s = z0
            for p in 1:layer.n_in
                blk = ((i-1)*layer.Q*layer.n_in + (q-1)*layer.n_in + (p-1)) * K1
                knots = layer.phi[i,q,p].knots           # Float64
                vals  = view(phi_vec, blk+1:blk+K1)      # Float64 or Dual
                s += eval_pw_external(knots, vals, x[p])
            end
            acc_i += layer.A[i,q]*s + layer.B[i,q]
        end
        out[i] = acc_i
    end
    return out
end

# loss over batch using logits columns
function loss_from_logits_cols(logits_cols, y)
    N = length(y)
    z0 = logits_cols[1][1] - logits_cols[1][1]
    acc = z0
    @inbounds for j in 1:N
        acc += cross_entropy_from_logits(logits_cols[j], y[j])
    end
    return acc / N
end

# -----------------------------
# Gradient wrt phi only (forward-mode, one seed per parameter)
# -----------------------------
function grad_phi_only(m::OneLayerInterModel{T},
                       X::AbstractMatrix{T},
                       y::AbstractVector{<:Integer};
                       alpha_smooth::Real = 0.0) where {T<:Real}

    layer = m.layer
    @assert size(X,1) == layer.n_in
    N = size(X,2)

    θ0 = pack_phi(layer)  # Float64 vector
    P  = length(θ0)
    g  = zeros(T, P)

    # Float loss for logging
    L0 = value(loss_batch(m, X, y))

    K1 = layer.K + 1

    @inbounds for k in 1:P
        θd = Vector{Dual{T}}(undef, P)
        for j in 1:P
            θd[j] = Dual(θ0[j], [j==k ? one(T) : zero(T)])
        end

        logits_cols = Vector{Vector{typeof(θd[1]-θd[1])}}(undef, N)
        for j in 1:N
            xj = @view X[:, j]
            logits_cols[j] = forward_with_phivec(layer, xj, θd)
        end

        Ld = loss_from_logits_cols(logits_cols, y)

        if alpha_smooth > 0
            Ld += T(alpha_smooth) * smoothness_penalty(θd, K1)
        end

        g[k] = deriv(Ld)[1]
    end

    return L0, g
end

# -----------------------------
# Trainer: phi-only, A,B fixed, λ forced to 0
# -----------------------------
function train_interlayer_phi_only_fixedAB!(m::OneLayerInterModel{T}, split;
    epochs::Int=200,
    lr::Real=0.05,
    seed::Int=123,
    run_tag::String="iris_interlayer_phi_only_fixedAB",
    runs_root::String="runs",
    alpha_smooth::Real=1e-3,
    clip_value::Real=3.0
) where {T<:Real}

    # FORCE λ = 0 (immutable layer)
    if m.layer.λ != zero(T)
        m = OneLayerInterModel(InterLayers.with_lambda(m.layer, 0.0))
    end

    # logger
    lg = RunLogger(runs_root=runs_root)
    new_run!(lg; tag=run_tag, meta=Dict(
        "seed" => seed,
        "epochs" => epochs,
        "lr" => lr,
        "phase" => "phi_only_fixedAB",
        "lambda" => m.layer.λ,
        "alpha_smooth" => alpha_smooth,
        "clip_value" => clip_value
    ))

    # data
    Xtr, ytr = split.X_train, split.y_train
    Xva, yva = split.X_val,   split.y_val
    Xte, yte = split.X_test,  split.y_test

    # optimizer (on phi-vector only)
    opt = SGD(T(lr); momentum=0.0)
    reset!(opt)

    best_val_acc = -1.0
    best_epoch   = 0

    for epoch in 1:epochs
        tic!(lg)

        Ltr, g = grad_phi_only(m, Xtr, ytr; alpha_smooth=alpha_smooth)

        # SGD step on phi vector
        θ = pack_phi(m.layer)
        params = [θ]
        grads  = [g]
        sgd_step!(opt, params, grads)

        # write back (with value clipping)
        unpack_phi!(m.layer, params[1]; clip_value=clip_value)

        # metrics
        logits_tr = forward_batch(m, Xtr)
        logits_va = forward_batch(m, Xva)
        logits_te = forward_batch(m, Xte)

        acc_tr = accuracy_from_logits(logits_tr, ytr)
        acc_va = accuracy_from_logits(logits_va, yva)
        acc_te = accuracy_from_logits(logits_te, yte)

        Lva = value(loss_batch(m, Xva, yva))
        Lte = value(loss_batch(m, Xte, yte))

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
    return lg, best_val_acc, best_epoch, m
end

end # module