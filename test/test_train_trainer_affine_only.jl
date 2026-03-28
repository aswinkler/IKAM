using Test
using Random
using IKAM

using IKAM.Data
using IKAM.Layers
using IKAM.Model
using IKAM.Trainer

@testset "Trainer affine-only runs and logs" begin
    split = IKAM.Data.load_iris_split(seed=123, ratios=(0.7,0.15,0.15))
    n_in = size(split.X_train, 1)
    m_out = 3

    layer = IKAM.Layers.init_kalayer(n_in, m_out; K=8, rng=MersenneTwister(1), scale=0.01)
    model = IKAM.Model.OneLayerModel(layer)

    mktempdir() do tmp
        run_dir = IKAM.Trainer.train_one_layer_affine_only!(model, split;
            epochs=3, lr=0.05, seed=123, run_tag="unit_affine_only", runs_root=joinpath(tmp, "runs")
        )
        @test isdir(run_dir)
        @test isfile(joinpath(run_dir, "history.csv"))
        @test isfile(joinpath(run_dir, "summary.txt"))

        lines = readlines(joinpath(run_dir, "history.csv"))
        @test length(lines) == 1 + 3  # header + 3 epochs
    end
end