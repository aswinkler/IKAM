# TrainerInterLayer_AB_plus_C_FixedPhi.jl

module InterTrainerABplusC

using Random

using ..AD: Dual, value, deriv
using ..Inner: evaluate
using ..InterLayers
using ..InterModel: OneLayerInterModel, forward_batch, loss_batch
using ..Train: SGD, reset!, sgd_step!
using ..Utils: RunLogger, new_run!, log_epoch!, write_summary!, tic!, toc!
using ..Metrics: accuracy_from_logits, cross_entropy_from_logits

export train_interlayer_AB_plus_C_phi_fixed!

# --- util: gradient clipping by norm ---
function clip_by_norm!(g::AbstractVector{T}, maxnorm::T) where {T<:Real}
    nrm2 = zero(T)
    @inbounds for i in eachindex(g)
        nrm2 += g[i]*g[i]
    end
    nrm = sqrt(nrm2)
    if nrm > maxnorm && nrm > zero(T)
        s = maxnorm / nrm
        @inbounds for i in eachindex(g)
            g[i] *= s
        end
    end
    return g
end

"""
    train_interlayer_AB_plus_C_phi_fixed!(model, split; ...)

Fase R1:
- λ fijo (no se toca aquí; lo fijas en la capa antes de llamar).
- Entrena SOLO C en los pares de I2 (y en todos i,q).
- A,B congelados.
- φ congelada.
"""
function train_interlayer_AB_plus_C_phi_fixed!(m::OneLayerInterModel{T}, split;
    epochs::Int=200,
    lr::Real=0.05,
    seed::Int=123,
    run_tag::String="iris_interlayer_AB_plus_C_FixedPhi",
    runs_root::String="runs",
    clip_norm::Real=5.0,
    ab_scale::Real=0.01,
    c_scale::Real=1.0
) where {T<:Real}

    # rng = MersenneTwister(seed)

    # --- logger/run ---
    lg = RunLogger(runs_root=runs_root)

    run_dir = new_run!(lg; tag=run_tag, meta=Dict(
        "seed" => seed,
        "epochs" => epochs,
        "lr" => lr,
        "phase" => "AB_plus_C_PhixedPhi",
        "n_in" => m.layer.n_in,
        "m_out" => m.layer.m_out,
        "Q" => m.layer.Q,
        "K" => m.layer.K,
        "lambda" => m.layer.λ,
        "I2_size" => length(m.layer.I2),
        "clip_norm" => clip_norm
    ))

    # --- data ---
    Xtr, ytr = split.X_train, split.y_train
    Xva, yva = split.X_val,   split.y_val
    Xte, yte = split.X_test,  split.y_test

    # --- optimizer (only for C-params vector) ---
    opt = SGD(T(lr); momentum=0.0)
    reset!(opt)

    # --- build index list for trainable C entries ---
    # We train only C[i,q,p,r] for (p,r) in I2.
    idxs = Vector{NTuple{4,Int}}()
    sizehint!(idxs, m.layer.m_out * m.layer.Q * length(m.layer.I2))
    @inbounds for i in 1:m.layer.m_out, q in 1:m.layer.Q
        for (p,r) in m.layer.I2
            push!(idxs, (i,q,p,r))
        end
    end
    P = length(idxs)

    # ----- pack/unpack C over idxs ----

    function pack_C(layer)::Vector{T}
        θ = Vector{T}(undef, P)
        @inbounds for k in 1:P
            (i,q,p,r) = idxs[k]
            θ[k] = layer.C[i,q,p,r]
        end
        return θ
    end

    function unpack_C!(layer, θ::Vector{T})
        @assert length(θ) == P
        @inbounds for k in 1:P
            (i,q,p,r) = idxs[k]
            layer.C[i,q,p,r] = θ[k]
        end
        return nothing
    end

    # -----------------------------
    # Pack / unpack A,B,C(active)
    # -----------------------------
    function pack_ABC(layer::InterLayers.KAInterLayer{T}) where {T<:Real}
        θAB = vcat(vec(copy(layer.A)), vec(copy(layer.B)))
        θC  = pack_C(layer)              # usa idxs + P del closure
        return vcat(θAB, θC)
    end

    function unpack_ABC!(layer::InterLayers.KAInterLayer{T}, θ::Vector{T}) where {T<:Real}
        P_A = length(layer.A)            # m_out*Q
        P_B = length(layer.B)            # m_out*Q
        P_AB = P_A + P_B

        @assert length(θ) == P_AB + P

        # A
        layer.A .= reshape(view(θ, 1:P_A), size(layer.A))
        # B
        layer.B .= reshape(view(θ, P_A+1:P_AB), size(layer.B))

        # C(active)
        θC = view(θ, P_AB+1:length(θ))
        @inbounds for k in 1:P
            (i,q,p,r) = idxs[k]
            layer.C[i,q,p,r] = θC[k]
        end
        return nothing
    end

    # --- forward with "external" C vector (Dual or Float), without converting φ ---
    # returns logits for one sample as Vector{Tout}
    function forward_with_Cvec(layer, x::AbstractVector, Cvec)
        @assert length(x) == layer.n_in

        z0 = Cvec[1] - Cvec[1]
        # proto = z0 + layer.A[1,1] - layer.A[1,1] + (layer.B[1,1] - layer.B[1,1])
        proto = z0
        Tout = typeof(proto)

        out = Vector{Tout}(undef, layer.m_out)
        φvals = Vector{Tout}(undef, layer.n_in)

        # helper: locate C index in Cvec by (i,q,p,r) index order
        # We use idxs list and a running pointer per (i,q) block to avoid search.
        # Since idxs is ordered by i,q then pairs, we can compute offset.
        Lpairs = length(layer.I2)

        @inbounds for i in 1:layer.m_out
            acc_i = z0
            for q in 1:layer.Q
                s = z0
                for p in 1:layer.n_in
                    vp = evaluate(layer.phi[i,q,p], x[p])  # Float64
                    vpD = z0 + vp
                    φvals[p] = vpD                           # promoted if needed
                    s += vpD
                end

                # affine term (A,B frozen)
                acc_i += layer.A[i,q] * s + layer.B[i,q]

                # el término de interacción utiliza Cvec
                if layer.λ != zero(T)
                    inter = z0
                    base = ((i-1)*layer.Q + (q-1))*Lpairs
                    for t in 1:Lpairs
                        (p,r) = layer.I2[t]
                        inter += Cvec[base + t] * φvals[p] * φvals[r]
                    end

                    # --- NORMALIZACIÓN ---
                    if Lpairs > 0
                        inter /= T(Lpairs)
                    end

                    acc_i += layer.λ * inter
                end
            end
            out[i] = acc_i
        end
        return out
    end

    # --- loss from logits over batch (column-wise) ---
    function loss_from_logits_cols(logits_cols, y)
        N = length(y)
        z0 = logits_cols[1][1] - logits_cols[1][1]
        acc = z0
        @inbounds for j in 1:N
            acc += cross_entropy_from_logits(logits_cols[j], y[j])
        end
        return acc / N
    end

    # ---------------------------------------------------------
    # grad_ABC: gradient w.r.t θ = [vec(A); vec(B); C(active)]
    # φ fija (Float64), y se inyectan A,B,C desde θ (Dual)
    # ---------------------------------------------------------
    function grad_ABC(m::OneLayerInterModel{T},
                    X::AbstractMatrix{T},
                    y::AbstractVector{<:Integer};
                    clip_norm::Real = 0.0) where {T<:Real}

        layer = m.layer
        @assert size(X,1) == layer.n_in
        N = size(X,2)

        # ----- sizes -----
        P_A  = length(layer.A)
        P_B  = length(layer.B)
        P_AB = P_A + P_B
        P_C  = P                  # P = length(idxs)
        Ptot = P_AB + P_C

        # ----- mapping: (i,q,p,r) -> k in 1:P_C -----
        # Usamos Dict para no crear un arreglo 4D grande.
        posC = Dict{NTuple{4,Int},Int}()
        @inbounds for k in 1:P_C
            posC[idxs[k]] = k
        end

        # ----- helper: forward logits usando θ (Dual) -----
        function forward_with_theta(x::AbstractVector{T}, θd::Vector{Dual{T}})
            # A,B como matrices Dual desde θd
            Ad = reshape(view(θd, 1:P_A), size(layer.A))
            Bd = reshape(view(θd, P_A+1:P_AB), size(layer.B))
            θCd = view(θd, P_AB+1:Ptot)  # C(active) en Dual

            z0 = θd[1] - θd[1]
            proto = z0 + Ad[1,1] + Bd[1,1]
            Tout = typeof(proto)

            out   = Vector{Tout}(undef, layer.m_out)
            φvals = Vector{Tout}(undef, layer.n_in)

            @inbounds for i in 1:layer.m_out
                acc_i = z0
                for q in 1:layer.Q
                    s = z0
                    for p in 1:layer.n_in
                        vp = evaluate(layer.phi[i,q,p], x[p])     # Float64
                        vpd = z0 + vp                              # promote -> Dual
                        φvals[p] = vpd
                        s += vpd
                    end

                    # término afín
                    acc_i += Ad[i,q] * s + Bd[i,q]

                    # interacción
                    if layer.λ != zero(T)
                        inter = z0
                        for (p,r) in layer.I2
                            k = get(posC, (i,q,p,r), 0)
                            if k != 0
                                inter += θCd[k] * φvals[p] * φvals[r]
                            else
                                # Si idxs está bien construido, esto NO debería ocurrir.
                                # Se deja como 0 para no romper ejecución.
                                inter += z0
                            end
                        end
                        # --- NORMALIZACIÓN ---
                        Lpairs = length(layer.I2)
                        if Lpairs > 0
                            inter /= T(Lpairs)
                        end

                        acc_i += layer.λ * inter
                    end
                end
                out[i] = acc_i
            end
            return out
        end

        # ----- helper: loss batch desde logits por columna -----
        function loss_from_logits_cols(logits_cols, y)
            acc = logits_cols[1][1] - logits_cols[1][1] # Dual zero
            @inbounds for j in 1:length(y)
                acc += cross_entropy_from_logits(logits_cols[j], y[j])
            end
            return acc / length(y)
        end

        # ----- base θ (Float64) -----
        θ0 = pack_ABC(layer)   # tu pack_ABC (A,B,C(active))
        @assert length(θ0) == Ptot

        # ----- gradient (forward-mode, 1 seed por parámetro) -----
        g = zeros(T, Ptot)

        # (Opcional) computar L0 una sola vez en Float64
        L0 = value(loss_batch(m, X, y))

        for k in 1:Ptot
            θd = Vector{Dual{T}}(undef, Ptot)
            @inbounds for j in 1:Ptot
                θd[j] = Dual(θ0[j], [j==k ? one(T) : zero(T)])
            end

            # logits por columna
            logits_cols = Vector{Vector{typeof(θd[1]-θd[1])}}(undef, N)
            @inbounds for j in 1:N
                xj = @view X[:, j]
                logits_cols[j] = forward_with_theta(xj, θd)
            end

            Ld = loss_from_logits_cols(logits_cols, y)
            g[k] = deriv(Ld)[1]
        end

        # ----- clip opcional -----
        #if clip_norm > 0
        #    gn = sqrt(sum(abs2, g))
        #    if gn > clip_norm
        #        g .*= (clip_norm / gn)
        #    end
        #end

        return L0, g
    end


    # --- gradient w.r.t. C only via forward AD seeding 1 param at a time ---
    function grad_C(m::OneLayerInterModel{T}, X::AbstractMatrix, y::AbstractVector{<:Integer})
        θ = pack_C(m.layer)                   # Float64 vector
        P = length(θ)
        g = zeros(T, P)

        # cache dims
        N = size(X, 2)

        # seeds are length-1 tangent vectors in your AD
        @inbounds for k in 1:P
            θd = Vector{Dual{T}}(undef, P)
            for j in 1:P
                θd[j] = Dual(θ[j], [j==k ? one(T) : zero(T)])
            end

            logits_cols = Vector{Vector{typeof(θd[1] - θd[1])}}(undef, N)
            for j in 1:N
                xj = @view X[:, j]
                logits_cols[j] = forward_with_Cvec(m.layer, xj, θd)
            end

            Ld = loss_from_logits_cols(logits_cols, y)
            # d = deriv(Ld)
            # @show length(d)

            g[k] = deriv(Ld)[1]
        end

        # also compute Float loss once (for logging)
        L0 = value(loss_batch(m, X, y))
        return L0, g
    end

    best_val_acc = -1.0
    best_epoch   = 0

    # ---- epoch 0 evaluation (before any update) ----
    logits_tr0 = forward_batch(m, Xtr)
    logits_va0 = forward_batch(m, Xva)
    logits_te0 = forward_batch(m, Xte)

    acc_tr0 = accuracy_from_logits(logits_tr0, ytr)
    acc_va0 = accuracy_from_logits(logits_va0, yva)
    acc_te0 = accuracy_from_logits(logits_te0, yte)

    Ltr0 = value(loss_batch(m, Xtr, ytr))
    Lva0 = value(loss_batch(m, Xva, yva))
    Lte0 = value(loss_batch(m, Xte, yte))

    log_epoch!(lg; epoch=0,
        loss_train=Ltr0, acc_train=acc_tr0,
        loss_val=Lva0,   acc_val=acc_va0,
        loss_test=Lte0,  acc_test=acc_te0,
        time_epoch_s=0.0
    )

    best_val_acc = acc_va0
    best_epoch   = 0


    for epoch in 1:epochs
        tic!(lg)

        # gradient + update (C only)
        Ltr, g = grad_ABC(m, Xtr, ytr; clip_norm=clip_norm)
        # --- split sizes (must match pack_ABC) ---
        P_A  = length(m.layer.A)
        P_B  = length(m.layer.B)
        P_AB = P_A + P_B
        P_C  = length(g) - P_AB

        # --- scale gradient blocks ---
        @inbounds for k in 1:P_AB
            g[k] *= T(ab_scale)
        end
        @inbounds for k in (P_AB+1):length(g)
            g[k] *= T(c_scale)
        end
        clip_by_norm!(g, T(clip_norm))

        θ = pack_ABC(m.layer)
        params = [θ]
        grads  = [g]
        sgd_step!(opt, params, grads)
        unpack_ABC!(m.layer, params[1])

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
    return lg, best_val_acc, best_epoch
end

end # module