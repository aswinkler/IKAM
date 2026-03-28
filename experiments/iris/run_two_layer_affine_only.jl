using Random
using IKAM

using IKAM.Data
using IKAM.Layers
using IKAM.Model2
using IKAM.Trainer2Affine

seed = 123
epochs = 300
lr = 0.05
K = 8
d = 5

split = IKAM.Data.load_iris_split(seed=seed, ratios=(0.7,0.15,0.15))
n_in = size(split.X_train, 1)
m_out = 3

layer1 = IKAM.Layers.init_kalayer(n_in, d; K=K, rng=MersenneTwister(2026), scale=0.01, Q=2n_in+1)
layer2 = IKAM.Layers.init_kalayer(d, m_out; K=K, rng=MersenneTwister(2027), scale=0.01, Q=2d+1)

model = IKAM.Model2.TwoLayerModel(layer1, layer2)

run_dir = IKAM.Trainer2Affine.train_two_layer_affine_only!(model, split;
    epochs=epochs, lr=lr, seed=seed, run_tag="iris_two_layer_affine_only", runs_root="runs"
)

println("Done. Results in: ", run_dir)