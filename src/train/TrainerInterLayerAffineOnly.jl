module InterTrainer

using Random

using ..AD: Dual, value, deriv
using ..InterModel: OneLayerInterModel, loss_batch, forward_batch
using ..InterLayers
using ..Inner: evaluate
using ..Train: SGD, reset!, sgd_step!
using ..Utils: RunLogger, new_run!, log_epoch!, tic!, toc!
using ..Metrics: cross_entropy_from_logits, accuracy_from_logits

export train_interlayer_affine_only!

"""
    train_interlayer_affine_only!(model, split; epochs, lr, seed, run_tag, runs_root)

Phase 1 (affine only):
- Train ONLY A and B of KAInterLayer.
- Keep φ and C frozen.
- Assumes interaction is disabled (e.g., λ=0.0) in the layer initialization.
"""
function train_interlayer_affine_only!(m::OneLayerInterModel{T}, split;
    epochs::Int=200,
    lr::Real=0.05,
    seed::Int=123,
    run_tag::String="iris_interlayer_affine_only",
    runs_root::String="runs"
) where {T<:Real}

    rng = MersenneTwister(seed)  # (not used yet, but kept for consistency)

    # Optimizer
    opt = SGD(T(lr); momentum=0.0)
    reset!(opt)

    # Logger
    lg = RunLogger(runs_root=runs_root)
    new_run!(lg; tag=run_tag, meta=Dict(
        "seed"  => seed,
        "epochs"=> epochs,
        "lr"    => lr,
        "n_in"  => m.layer.n_in,
        "m_out" => m.layer.m_out,
        "Q"     => m.layer.Q,
        "K"     => m.layer.K,
        "phase" => "inter_affine_only",
        # We do NOT mutate λ (layer is immutable). This is informational.
        "lambda" => m.layer.λ
    ))

    # Data
    Xtr, ytr = split.X_train, split.y_train
    Xva, yva = split.X_val,   split.y_val
    Xte, yte = split.X_test,  split.y_test

    best_val_acc = -1.0
    best_epoch   = 0

    # -----------------------------
    # Pack / unpack ONLY A and B
    # -----------------------------
    function pack_AB(layer::InterLayers.KAInterLayer{T})
        return vcat(vec(copy(layer.A)), vec(copy(layer.B)))
    end

    function unpack_AB!(layer::InterLayers.KAInterLayer{T}, θ::Vector{T})
        P_A = length(layer.A)
        layer.A .= reshape(θ[1:P_A], size(layer.A))
        layer.B .= reshape(θ[P_A+1:end], size(layer.B))
        return nothing
    end

    # ---------------------------------------------------------
    # Forward logits with overridden A,B (possibly Dual matrices)
    # while keeping phi, C, I2, etc. as stored in the layer.
    # ---------------------------------------------------------
    function forward_logits_AB(layer::InterLayers.KAInterLayer{T},
                               x::AbstractVector,
                               Ad, Bd)

        @assert length(x) == layer.n_in

        # Dual "zero" that preserves derivative dimension (seed length=1)
        z0 = Ad[1,1] - Ad[1,1]
        Tout = typeof(z0)

        out = Vector{Tout}(undef, layer.m_out)

        @inbounds for i in 1:layer.m_out
            acc_i = z0
            for q in 1:layer.Q
                s = z0
                for p in 1:layer.n_in
                    # evaluate(phi, Float64) -> Float64, promoted into Dual via + s
                    s += evaluate(layer.phi[i,q,p], x[p])
                end
                acc_i += Ad[i,q] * s + Bd[i,q]
            end
            out[i] = acc_i
        end

        return out
    end

    # -----------------------------------------
    # Compute (loss, grad) w.r.t. packed (A,B)
    # using forward-mode AD with Dual (seed dim 1)
    # -----------------------------------------
    function grad_AB(m::OneLayerInterModel{T}, X::AbstractMatrix, y::AbstractVector{<:Integer})
        layer = m.layer

        θ = pack_AB(layer)
        P = length(θ)
        g = zeros(T, P)

        seed0 = [zero(T)]
        seed1 = [one(T)]

        # We compute g[k] by seeding θ[k]
        for k in 1:P
            θd = Vector{Dual{T}}(undef, P)
            @inbounds for j in 1:P
                θd[j] = Dual(θ[j], (j==k) ? seed1 : seed0)
            end

            P_A = length(layer.A)
            Ad = reshape(θd[1:P_A], size(layer.A))
            Bd = reshape(θd[P_A+1:end], size(layer.B))

            # batch loss accumulation in Dual
            N = size(X, 2)
            acc = Ad[1,1] - Ad[1,1]  # Dual zero

            @inbounds for j in 1:N
                xj = @view X[:, j]  # Float64
                logits = forward_logits_AB(layer, xj, Ad, Bd)
                acc += cross_entropy_from_logits(logits, y[j])
            end

            Ld = acc / N
            g[k] = deriv(Ld)[1]   # seed dim = 1
        end

        L0 = value(loss_batch(m, X, y))
        return L0, g
    end

    # -----------------------------
    # Training loop
    # -----------------------------
    for epoch in 1:epochs
        tic!(lg)

        Ltr, g = grad_AB(m, Xtr, ytr)

        θ = pack_AB(m.layer)
        params = [θ]
        grads  = [g]
        sgd_step!(opt, params, grads)
        unpack_AB!(m.layer, params[1])

        # Metrics
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
            loss_train=Ltr,
            acc_train=acc_tr,
            loss_val=Lva,
            acc_val=acc_va,
            loss_test=Lte,
            acc_test=acc_te,
            time_epoch_s=t
        )

        if acc_va > best_val_acc
            best_val_acc = acc_va
            best_epoch   = epoch
        end
    end

    # Store best summary if your RunLogger supports it; otherwise just return them.
    # If RunLogger is mutable and has these fields, keep them; if not, ignore safely.
    try
        setproperty!(lg, :best_val_acc, best_val_acc)
        setproperty!(lg, :best_epoch, best_epoch)
    catch
        # ignore if logger is immutable or fields do not exist
    end

    return lg, best_val_acc, best_epoch
end

end # module InterTrainer