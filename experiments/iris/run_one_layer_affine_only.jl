using Random
using IKAM

using IKAM.Data
using IKAM.Layers
using IKAM.Model
using IKAM.Trainer

# Config
seed = 123
epochs = 200
lr = 0.05
K = 8

# Data
split = IKAM.Data.load_iris_split(seed=seed, ratios=(0.7,0.15,0.15))

n_in = size(split.X_train, 1)
m_out = 3

# Model: One layer
rng = MersenneTwister(2026)
layer = IKAM.Layers.init_kalayer(n_in, m_out; K=K, rng=rng, scale=0.01)
model = IKAM.Model.OneLayerModel(layer)

run_dir = IKAM.Trainer.train_one_layer_affine_only!(model, split;
    epochs=epochs, lr=lr, seed=seed, run_tag="iris_one_layer_affine_only", runs_root="runs"
)

println("Done. Results in: ", run_dir)