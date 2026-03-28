using Test
using Random
using IKAM

using IKAM.Data
using IKAM.Layers
using IKAM.Model
using IKAM.TrainerFull

@testset "Trainer full (A,B,phi-values) runs and logs" begin
    split = IKAM.Data.load_iris_split(seed=123, ratios=(0.7,0.15,0.15))
    n_in = size(split.X_train, 1)
    m_out = 3

    layer = IKAM.Layers.init_kalayer(n_in, m_out; K=8, rng=MersenneTwister(1), scale=0.01)
    model = IKAM.Model.OneLayerModel(layer)

    mktempdir() do tmp
        run_dir = IKAM.TrainerFull.train_one_layer_full!(model, split;
            epochs=2, lr=0.02, seed=123,
            run_tag="unit_full", runs_root=joinpath(tmp, "runs")
        )
        @test isdir(run_dir)
        @test isfile(joinpath(run_dir, "history.csv"))
        lines = readlines(joinpath(run_dir, "history.csv"))
        @test length(lines) == 1 + 2
    end
end