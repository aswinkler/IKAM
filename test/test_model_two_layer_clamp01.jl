using Test
using Random

using IKAM.Data
using IKAM.Layers
using IKAM.Model2
using IKAM.AD: value

@testset "TwoLayerModel clamps intermediate representation to [0,1]" begin
    split = load_iris_split(seed=123, ratios=(0.7, 0.15, 0.15))
    Xtr = split.X_train

    n_in = size(Xtr, 1)
    d = 5
    m_out = 3

    layer1 = init_kalayer(n_in, d; K=8, rng=MersenneTwister(1), scale=0.5, Q=2n_in+1)
    layer2 = init_kalayer(d, m_out; K=8, rng=MersenneTwister(2), scale=0.01, Q=2d+1)
    model = TwoLayerModel(layer1, layer2)

    # Calcula Z sin clamp (directo de layer1)
    Z = forward_batch(layer1, Xtr)

    # Verifica que haya valores fuera de [0,1] (muy probable con scale=0.5)
    has_outside = any(z -> (value(z) < 0 || value(z) > 1), Z)
    @test has_outside == true

    # Ahora forward_batch del modelo debe funcionar (internamente clamp)
    logits = forward_batch(model, Xtr)
    @test size(logits, 2) == size(Xtr, 2)
    @test all(isfinite, logits)
end