using Random
using IKAM
using IKAM.Data
using IKAM.InterLayers
using IKAM.InterModel
using IKAM.InterTrainerPhiOnly   # <-- el módulo nuevo

# dataset
split = IKAM.Data.load_iris_split(; ratios=(0.7,0.15,0.15), seed=123)

# model
n_in  = size(split.X_train, 1)
m_out = 3
Q     = 2n_in + 1
K     = 8
I2    = [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]

layer = IKAM.InterLayers.init_kainterlayer(n_in, m_out;
    Q=Q, K=K, λ=0.0, I2=I2,
    scale=0.05,
    rng=Random.MersenneTwister(123)
)
model = IKAM.InterModel.OneLayerInterModel(layer)

# train phi only (AB fixed, λ forced 0)
lg, best, ep, model_trained = IKAM.InterTrainerPhiOnly.train_interlayer_phi_only_fixedAB!(model, split;
    epochs=200, lr=0.05, seed=123,
    run_tag="iris_interlayer_phi_only_fixedAB",
    runs_root="runs",
    alpha_smooth=1e-3,   # suavidad nodal
    clip_value=3.0       # clip de values nodales
)

println("phi-only: best_val_acc=$(best) best_epoch=$(ep) run_dir=$(lg.run_dir)")