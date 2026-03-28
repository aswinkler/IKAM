using Test
using IKAM.AD
using IKAM.Metrics

@testset "Softmax and Cross-Entropy from logits" begin
    # Caso simple
    logits = [1.0, 2.0, 3.0]
    p = softmax(logits)

    @test length(p) == 3
    @test sum(p) ≈ 1.0 atol=1e-12
    @test all(pi -> pi > 0.0, p)

    # NLL estable: logsumexp - logit[y]
    y = 3
    ce = cross_entropy_from_logits(logits, y)
    @test ce ≈ (logsumexp(logits) - logits[y]) atol=1e-12

    # Si sube el logit correcto, la pérdida baja
    logits2 = [1.0, 2.0, 4.0]
    ce2 = cross_entropy_from_logits(logits2, y)
    @test ce2 < ce

    # Estabilidad numérica con logits grandes
    big = [1000.0, 1001.0, 1002.0]
    pbig = softmax(big)
    @test sum(pbig) ≈ 1.0 atol=1e-12
    @test all(isfinite, pbig)

    cebig = cross_entropy_from_logits(big, 3)
    @test isfinite(cebig)

    # Compatibilidad con Dual: derivada en la dirección del logit correcto es negativa
    P = 3
    ld = [seed!(1.0, 1, P), seed!(2.0, 2, P), seed!(3.0, 3, P)]
    ced = cross_entropy_from_logits(ld, 3)
    # d/d logit[y] = softmax[y] - 1 < 0
    pd = softmax(ld)
    @test deriv(ced)[3] ≈ (value(pd[3]) - 1.0) atol=1e-10
end