using Random
using IKAM

using IKAM.Data
using IKAM.Layers
using IKAM.InterModel
using IKAM.TrainerFull

seed = 123
epochs = 300
lr = 0.02
K = 8

split = IKAM.Data.load_iris_split(seed=seed, ratios=(0.7,0.15,0.15))
n_in = size(split.X_train, 1)
m_out = 3

layer = IKAM.Layers.init_kalayer(n_in, m_out; K=K, rng=MersenneTwister(2026), scale=0.01)
model = IKAM.Model.OneLayerModel(layer)

run_dir = IKAM.TrainerFull.train_one_layer_full!(model, split;
    epochs=epochs, lr=lr, seed=seed,
    run_tag="iris_one_layer_full", runs_root="runs"
)

println("Done. Results in: ", run_dir)