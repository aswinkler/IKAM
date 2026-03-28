# src/experiments/run_two_interlayer_full_ABPhiC.jl
using Random
using IKAM
using IKAM.Data
using IKAM.InterLayers
using IKAM.TwoInterModel
using IKAM.InterTrainerTwoLayerFullABPhiC

split = IKAM.Data.load_iris_split(; ratios=(0.7,0.15,0.15), seed=123)

n_in      = size(split.X_train, 1)
n_hidden  = 5
m_out     = 3

Q1 = 2n_in + 1
Q2 = 2n_hidden + 1
K  = 8

I2_1 = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
I2_2 = [(1,2),(1,3),(1,4),(1,5),
        (2,3),(2,4),(2,5),
        (3,4),(3,5),
        (4,5)]

rng1 = Random.MersenneTwister(123)
rng2 = Random.MersenneTwister(124)

layer1 = IKAM.InterLayers.init_kainterlayer(n_in, n_hidden;
    Q=Q1, K=K,
    λ=0.0,
    I2=I2_1,
    scale=0.05,
    rng=rng1
)

layer2 = IKAM.InterLayers.init_kainterlayer(n_hidden, m_out;
    Q=Q2, K=K,
    λ=0.0,
    I2=I2_2,
    scale=0.05,
    rng=rng2
)

hidden_scaler = IKAM.TwoInterModel.fit_hidden_scaler(layer1, split.X_train)

model = IKAM.TwoInterModel.TwoLayerInterModel(layer1, layer2, hidden_scaler)

lg, best, ep = IKAM.InterTrainerTwoLayerFullABPhiC.train_two_interlayer_full_AB_phi_C!(model, split;
    epochs=300,
    lr=0.03,
    seed=123,
    run_tag="iris_two_interlayer_full_AB_phi_C",
    runs_root="runs",
    λ_max=15.0,
    warmup_epochs=40,
    alpha_smooth=1e-3,
    beta_C=0.001,
    clip_value=2.5
)

println("best_val_acc=$(best)  best_epoch=$(ep)  run_dir=$(lg.run_dir)")
