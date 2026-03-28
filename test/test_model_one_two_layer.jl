using Test
using Random

using IKAM.Data
using IKAM.Layers
using IKAM.Model
using IKAM.Model2

@testset "Model: OneLayer and TwoLayer loss/accuracy basic" begin
    rng = MersenneTwister(2026)

    # Datos Iris ya normalizados a [0,1]
    split = load_iris_split(seed=123, ratios=(0.7, 0.15, 0.15))
    Xtr, ytr = split.X_train, split.y_train
    Xva, yva = split.X_val, split.y_val

    n_in = size(Xtr, 1)
    m_out = 3
    K = 8

    # One layer
    layer = init_kalayer(n_in, m_out; K=K, rng=rng, scale=0.01)
    m1 = OneLayerModel(layer)

    Ltr = IKAM.Model.loss_batch(m1, Xtr, ytr)
    Lva = IKAM.Model.loss_batch(m1, Xva, yva)
    @test isfinite(Ltr)
    @test isfinite(Lva)
    @test Ltr > 0
    @test Lva > 0

    acc_tr = IKAM.Model.accuracy(m1, Xtr, ytr)
    acc_va = IKAM.Model.accuracy(m1, Xva, yva)
    @test 0.0 <= acc_tr <= 1.0
    @test 0.0 <= acc_va <= 1.0

    # Two layer
    d = 5
    layer1 = init_kalayer(n_in, d; K=K, rng=MersenneTwister(1), scale=0.01, Q=2n_in+1)
    layer2 = init_kalayer(d, m_out; K=K, rng=MersenneTwister(2), scale=0.01, Q=2d+1)
    m2 = TwoLayerModel(layer1, layer2)

    Ltr2 = IKAM.Model2.loss_batch(m2, Xtr, ytr)
    Lva2 = IKAM.Model2.loss_batch(m2, Xva, yva)
    @test isfinite(Ltr2)
    @test isfinite(Lva2)
    @test Ltr2 > 0
    @test Lva2 > 0

    acc_tr2 = IKAM.Model2.accuracy(m2, Xtr, ytr)
    acc_va2 = IKAM.Model2.accuracy(m2, Xva, yva)
    @test 0.0 <= acc_tr2 <= 1.0
    @test 0.0 <= acc_va2 <= 1.0
end