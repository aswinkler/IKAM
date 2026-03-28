using Random
using IKAM
using IKAM.Data
using IKAM.InterLayers
using IKAM.InterModel
using IKAM.InterTrainerPhiPlusC   # <-- este trainer nuevo
using IKAM.Metrics

# 1) dataset
split = IKAM.Data.load_iris_split(; ratios=(0.7,0.15,0.15), seed=123)

# 2) model (A,B fixed from init; you can also load from a previous AB-only checkpoint later)
n_in  = size(split.X_train, 1)
m_out = 3
Q     = 2n_in + 1
K     = 8
I2    = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

layer = IKAM.InterLayers.init_kainterlayer(n_in, m_out;
    Q=Q, K=K,
    λ=0.0,            # layer λ is not used by trainer's internal λ schedule
    I2=I2,
    scale=0.05,
    rng=Random.MersenneTwister(123)
)
model = IKAM.InterModel.OneLayerInterModel(layer)

# 3) train phi + C with AB fixed
lg, best, ep = IKAM.InterTrainerPhiPlusC.train_interlayer_phi_plus_C_fixedAB!(model, split;
    epochs=200,
    lr=0.05,
    seed=123,
    run_tag="iris_interlayer_phi_plus_C_fixedAB",
    runs_root="runs",
    λ_max=20.0,
    warmup_epochs=20,
    alpha_smooth=1e-3,
    beta_C=1e-3,
    clip_value=3.0
)

println("best_val_acc=$(best)  best_epoch=$(ep)  run_dir=$(lg.run_dir)")