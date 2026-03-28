using Test
using IKAM.AD

@testset "Dual basic arithmetic" begin
    P = 3
    x = seed!(2.0, 1, P)         # dx/dθ1 = 1
    y = seed!(3.0, 2, P)         # dy/dθ2 = 1

    z = x*y + 5.0
    @test value(z) ≈ 11.0
    @test deriv(z)[1] ≈ 3.0      # ∂(xy)/∂θ1 = y
    @test deriv(z)[2] ≈ 2.0      # ∂(xy)/∂θ2 = x
    @test deriv(z)[3] ≈ 0.0

    w = exp(x)
    @test value(w) ≈ exp(2.0)
    @test deriv(w)[1] ≈ exp(2.0)
end