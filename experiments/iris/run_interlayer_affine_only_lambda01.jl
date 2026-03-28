using Random
using IKAM
using IKAM.Data
using IKAM.InterLayers
using IKAM.InterModel
using IKAM.InterTrainer

# 1) dataset
split = IKAM.Data.load_iris_split(; ratios=(0.7,0.15,0.15), seed=123)

# 2) interlayer (interaction enabled but C frozen; train only A,B)
n_in = size(split.X_train, 1)
m_out = 3
Q = 2n_in + 1
K = 8
I2 = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

λ = 0.1

layer = IKAM.InterLayers.init_kainterlayer(n_in, m_out;
    Q=Q, K=K, λ=λ, I2=I2,
    scale=0.05,
    rng=Random.MersenneTwister(123)
)

model = IKAM.InterModel.OneLayerInterModel(layer)

# 3) training (affine only)
@show model.layer.λ
@show maximum(abs.(model.layer.C))

lg, best_val_acc, best_epoch =
    IKAM.InterTrainer.train_interlayer_affine_only!(model, split;
        epochs=200, lr=0.05, seed=123,
        run_tag="iris_interlayer_affine_only_lambda01",
        runs_root="runs"
    )

println("best_val_acc=$(best_val_acc)  best_epoch=$(best_epoch)")
println("run_dir=$(lg.run_dir)")