using Random
using IKAM
using IKAM.Data
using IKAM.InterLayers
using IKAM.InterModel
using IKAM.InterTrainer          # tu trainer affine-only (A,B)
using IKAM.InterTrainerCOnly     # el trainer C-only que acabas de correr

function inter_vs_affine_stats(layer, X)
    N = size(X,2)
    aff = 0.0
    inter = 0.0
    for j in 1:N
        x = view(X,:,j)
        # una salida i=1, un canal q=1 basta para estimar magnitudes
        i, q = 1, 1
        s = 0.0
        φ = zeros(layer.n_in)
        for p in 1:layer.n_in
            φ[p] = IKAM.Inner.evaluate(layer.phi[i,q,p], x[p])
            s += φ[p]
        end
        aff += abs(layer.A[i,q]*s + layer.B[i,q])
        tmp = 0.0
        for (p,r) in layer.I2
            tmp += layer.C[i,q,p,r]*φ[p]*φ[r]
        end
        inter += abs(layer.λ*tmp)
    end
    return aff/N, inter/N
end

function inter_nonzero_fraction(layer, X; eps=1e-15)
    N = size(X,2)
    nz = 0
    tot = 0
    for j in 1:N
        x = view(X,:,j)
        for i in 1:layer.m_out, q in 1:layer.Q
            φ = zeros(layer.n_in)
            for p in 1:layer.n_in
                φ[p] = IKAM.Inner.evaluate(layer.phi[i,q,p], x[p])
            end
            tmp = 0.0
            for (p,r) in layer.I2
                tmp += layer.C[i,q,p,r]*φ[p]*φ[r]
            end
            tot += 1
            nz += (abs(tmp) > eps) ? 1 : 0
        end
    end
    return nz / tot
end

# dataset
split = IKAM.Data.load_iris_split(; ratios=(0.7,0.15,0.15), seed=123)

# model
n_in = size(split.X_train, 1)
m_out = 3
Q = 2n_in + 1
K = 8
I2 = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

layer = IKAM.InterLayers.init_kainterlayer(n_in, m_out;
    Q=Q, K=K, λ=0.0, I2=I2,
    scale=0.05,
    rng=Random.MersenneTwister(123)
)
model = IKAM.InterModel.OneLayerInterModel(layer)

@show inter_nonzero_fraction(model.layer, split.X_train)

# -------------------------
# Phase 1: train A,B only (λ forced off inside trainer)
# -------------------------
@show model.layer.λ
lg1, best1, ep1 = IKAM.InterTrainer.train_interlayer_affine_only!(model, split;
    epochs=200, lr=0.05, seed=123,
    run_tag="iris_interlayer_AB_only",
    runs_root="runs"
)
println("AB-only: best_val_acc=$(best1) best_epoch=$(ep1) run_dir=$(lg1.run_dir)")
@show inter_vs_affine_stats(model.layer, split.X_train)


# -------------------------
# Phase 2: train C only (λ on)
# -------------------------
# model.layer.λ = 1.0
model = IKAM.InterModel.OneLayerInterModel(IKAM.InterLayers.with_lambda(model.layer, 50.0))
@show model.layer.λ
@show maximum(abs.(model.layer.C))

lg2, best2, ep2 = IKAM.InterTrainerCOnly.train_interlayer_C_only!(model, split;
    epochs=200, lr=0.05, seed=123,
    run_tag="iris_interlayer_AB_then_C_lambda10",
    runs_root="runs",
    clip_norm=5.0
)
println("AB->C: best_val_acc=$(best2) best_epoch=$(ep2) run_dir=$(lg2.run_dir)")