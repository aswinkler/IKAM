using Test
using Random
using IKAM.AD
using IKAM.Layers

@testset "KALayer: shapes, forward, Dual analytic check" begin
    rng = MersenneTwister(2026)
    n_in = 4
    m_out = 3
    K = 8
    layer = init_kalayer(n_in, m_out; K=K, rng=rng, scale=0.01)

    @test layer.n_in == n_in
    @test layer.m_out == m_out
    @test layer.Q == 2n_in + 1
    @test size(layer.A) == (m_out, layer.Q)
    @test size(layer.B) == (m_out, layer.Q)
    @test size(layer.phi) == (m_out, layer.Q, n_in)

    # Forward con Float64
    x = rand(rng, n_in)
    y = forward(layer, x)
    @test length(y) == m_out
    @test all(isfinite, y)

    # Forward batch
    X = rand(rng, n_in, 10)
    Y = forward_batch(layer, X)
    @test size(Y) == (m_out, 10)
    @test all(isfinite, Y)

    # --- Prueba analítica con φ(x)=x, A=1, B=0 ---
    # Entonces:
    # s_{i,q}(x) = sum_p x_p
    # f_i(x) = sum_q (1 * s + 0) = Q * sum_p x_p
    # d f_i / d x1 = Q
    layer2 = init_kalayer(n_in, m_out; K=K, rng=MersenneTwister(1), scale=0.0)
    set_identity_phi!(layer2)
    set_affine!(layer2; a=1.0, b=0.0)

    P = 1
    xd = Vector{Dual{Float64}}(undef, n_in)
    xd[1] = seed!(0.37, 1, P)  # deriv wrt x1
    for p in 2:n_in
        xd[p] = Dual{Float64}(0.1 * p, zeros(Float64, P)) # constantes
    end

    yd = forward(layer2, xd)

    @show typeof(yd[1])
    @show deriv(yd[1]) typeof(deriv(yd[1])) length(deriv(yd[1]))

    @test length(yd) == m_out
    for i in 1:m_out
        #@test nderiv(yd[i]) == 1
        @test deriv(yd[i])[1] ≈ layer2.Q atol=1e-12
    end
end