using Test
using Random
using IKAM

using IKAM.Data
using IKAM.Layers
using IKAM.Model2
using IKAM.Trainer2Affine
using IKAM.Trainer2Full

@testset "Two-layer trainers run and log" begin
    split = IKAM.Data.load_iris_split(seed=123, ratios=(0.7,0.15,0.15))
    n_in = size(split.X_train, 1)
    m_out = 3
    d = 5

    layer1 = IKAM.Layers.init_kalayer(n_in, d; K=8, rng=MersenneTwister(1), scale=0.01, Q=2n_in+1)
    layer2 = IKAM.Layers.init_kalayer(d, m_out; K=8, rng=MersenneTwister(2), scale=0.01, Q=2d+1)
    model = IKAM.Model2.TwoLayerModel(layer1, layer2)

    mktempdir() do tmp
        dir1 = IKAM.Trainer2Affine.train_two_layer_affine_only!(model, split;
            epochs=2, lr=0.05, seed=123, run_tag="unit_2_affine", runs_root=joinpath(tmp,"runs"))
        @test isfile(joinpath(dir1, "history.csv"))

        dir2 = IKAM.Trainer2Full.train_two_layer_full!(model, split;
            epochs=2, lr=0.01, seed=123, run_tag="unit_2_full", runs_root=joinpath(tmp,"runs"))
        @test isfile(joinpath(dir2, "history.csv"))
    end
end