# experiments/iris/run_interlayer_lambda_sweep_AB_plus_C_fixedPhi.jl

using Random
using IKAM
using IKAM.Data
using IKAM.InterLayers
using IKAM.InterModel
using IKAM.InterTrainer             # AB-only
using IKAM.InterTrainerABplusC      # AB+C (phi fixed)

# --- diagnostics (optional) ---
function inter_vs_affine_stats(layer, X)
    N = size(X, 2)
    aff = 0.0
    inter = 0.0
    for j in 1:N
        x = view(X, :, j)
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

        # si ya normalizaste en forward, normaliza también aquí para comparar magnitudes
        Lpairs = length(layer.I2)
        if Lpairs > 0
            tmp /= Lpairs
        end

        inter += abs(layer.λ * tmp)
    end
    return aff/N, inter/N
end

# 1) dataset
split = IKAM.Data.load_iris_split(; ratios=(0.7,0.15,0.15), seed=123)
Xtr = split.X_train

# 2) base model
n_in  = size(split.X_train, 1)
m_out = 3
Q     = 2n_in + 1
K     = 8
I2    = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

layer0 = IKAM.InterLayers.init_kainterlayer(n_in, m_out;
    Q=Q, K=K, λ=0.0, I2=I2,
    scale=0.05,
    rng=Random.MersenneTwister(123)
)
model0 = IKAM.InterModel.OneLayerInterModel(layer0)

# 3) phase 1: AB-only once (reutilizable para todos los lambdas)
lgAB, bestAB, epAB = IKAM.InterTrainer.train_interlayer_affine_only!(model0, split;
    epochs=200, lr=0.05, seed=123,
    run_tag="iris_interlayer_AB_only_for_sweep",
    runs_root="runs"
)
println("AB-only: best_val_acc=$(bestAB) best_epoch=$(epAB) run_dir=$(lgAB.run_dir)")
@show inter_vs_affine_stats(model0.layer, Xtr)

# 4) phase 3: sweep lambdas (AB+C phi fixed)
lambdas = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]

for λ in lambdas
    # KAInterLayer immutable => rebuild via with_lambda
    model = IKAM.InterModel.OneLayerInterModel(IKAM.InterLayers.with_lambda(model0.layer, λ))

    println("\n==============================")
    println("λ = ", λ)
    @show inter_vs_affine_stats(model.layer, Xtr)

    lg, best, ep = IKAM.InterTrainerABplusC.train_interlayer_AB_plus_C_phi_fixed!(model, split;
        epochs=200, lr=0.05, seed=123,
        run_tag="iris_interlayer_AB_plus_C_fixedPhi_lambda$(λ)",
        runs_root="runs",
        clip_norm=5.0,
        ab_scale=0.01,
        c_scale=1.0
    )

    println("AB+C (phi fixed): best_val_acc=$(best) best_epoch=$(ep) run_dir=$(lg.run_dir)")
    @show inter_vs_affine_stats(model.layer, Xtr)
end