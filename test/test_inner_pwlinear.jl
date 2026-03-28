using Test
using IKAM.AD
using IKAM.Inner

@testset "PWLinearNodal evaluates and supports Dual" begin
    K = 8
    knots = uniform_knots(Float64, K)
    values = collect(0.0:1.0:8.0) ./ 8.0  # y_k = k/8

    pw = PWLinearNodal{Float64}(knots, values)

    # Primal checks
    @test evaluate(pw, 0.0) ≈ 0.0
    @test evaluate(pw, 1.0) ≈ 1.0
    @test evaluate(pw, 0.5) ≈ 0.5

    # Dual check: derivada debe ser la pendiente del tramo
    # En y=x con nudos uniformes, la pendiente es 1 en todos los tramos
    P = 1
    xd = seed!(0.37, 1, P)
    yd = evaluate(pw, xd)

    @test value(yd) ≈ 0.37 atol=1e-12
    @test deriv(yd)[1] ≈ 1.0 atol=1e-12
end