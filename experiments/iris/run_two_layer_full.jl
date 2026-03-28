using Random
using IKAM

using IKAM.Data
using IKAM.Layers
using IKAM.InterLayer
using IKAM.Model2
using IKAM.Trainer2Full

seed = 123
epochs = 400
lr1 = 0.001
lr2 = 0.01
K = 8
d = 5
margin_factor = 2.0

split = IKAM.Data.load_iris_split(seed=seed, ratios=(0.7,0.15,0.15))
n_in = size(split.X_train, 1)
m_out = 3

layer1 = IKAM.Layers.init_kalayer(n_in, d; K=K, rng=MersenneTwister(2026), scale=0.01, Q=2n_in+1)
layer2 = IKAM.Layers.init_kalayer(d, m_out; K=K, rng=MersenneTwister(2027), scale=0.01, Q=2d+1)

Ztr = IKAM.Layers.forward_batch(layer1, split.X_train)
scaler = IKAM.InterLayer.fit_interlayer_minmax(Ztr, margin_factor=margin_factor)
model = IKAM.Model2.TwoLayerModelScaled(layer1, layer2, scaler)

run_dir = IKAM.Trainer2Full.train_two_layer_full!(model, split;
    epochs=epochs, lr1=lr1, lr2=lr2, clip_norm=1.0, seed=seed, run_tag="iris_two_layer_full", runs_root="runs"
)

println("Done. Results in: ", run_dir)