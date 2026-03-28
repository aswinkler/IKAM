using Test
using IKAM.Train

@testset "SGD step: simple quadratic minimization" begin
    # Minimizamos f(w) = (w - 3)^2 con grad g = 2(w-3)
    opt = SGD(0.1)  # lr

    # params como bloques vectorizados
    p = [ [0.0] ]   # w=0
    for _ in 1:60
        w = p[1][1]
        g = [ [2*(w - 3.0)] ]
        sgd_step!(opt, p, g)
    end
    @test abs(p[1][1] - 3.0) < 1e-3
end