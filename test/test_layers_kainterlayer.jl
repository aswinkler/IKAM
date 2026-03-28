# test/test_layers_kainterlayer.jl
using Test
using IKAM
using IKAM.Layers
using IKAM.InterLayers
using IKAM.AD

@testset "KAInterLayer: forward and Dual derivative sanity" begin
    rng = Random.MersenneTwister(123)

    n_in = 4
    m_out = 1
    Q = 3
    K = 8

    # Solo un par de interacción para test: (1,2)
    I2 = [(1,2)]

    layer = init_kainterlayer(n_in, m_out; Q=Q, K=K, rng=rng, scale=0.0, λ=1.0, I2=I2, T=Float64)

    # φ identidad, A=B=0, y C[1,1,1,2]=c
    InterLayers.set_identity_phi!(layer)
    InterLayers.set_affine!(layer; a=0.0, b=0.0)
    layer.C .= 0.0
    c = 2.5
    for q in 1:Q 
        layer.C[1,q,1,2] = c
    end
    # layer.C[1,1,1,2] = c

    # Entonces, para q=1..Q:
    # y = sum_q [ λ * c * x1 * x2 ] = Q * c * x1 * x2
    x = [0.2, 0.7, 0.1, 0.9]
    y = InterLayers.forward(layer, x)[1]
    @test isfinite(y)
    @test y ≈ (Q * c * x[1] * x[2])

    # Dual check: deriv wrt x1 should be Q*c*x2
    # Construye xdual con semilla en componente 1
    # Ajusta esto si tu constructor Dual difiere:
    xdual = Vector{Dual{Float64}}(undef, n_in)
    xdual[1] = Dual(x[1], [1.0])   # d/dx1 = 1
    xdual[2] = Dual(x[2], [0.0])
    xdual[3] = Dual(x[3], [0.0])
    xdual[4] = Dual(x[4], [0.0])

    ydual = InterLayers.forward(layer, xdual)[1]
    dy_dx1 = deriv(ydual)[1]
    @test dy_dx1 ≈ (Q * c * x[2]) atol=1e-10
end