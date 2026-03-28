# src/metrics/SoftmaxCrossEntropy.jl

module Metrics

using ..AD: Dual, value

export softmax, logsumexp, cross_entropy_from_logits, accuracy_from_logits,
       nll_from_logits

"""
    logsumexp(logits)

Computa log(sum(exp(logits))) de forma estable numéricamente, compatible con Dual.
Para Dual, el desplazamiento se decide con el valor primal.
"""
function logsumexp(logits::AbstractVector)
    @assert length(logits) > 0
    # shift = max(logits) usando valor primal para estabilidad y branching
    maxv = maximum(value.(logits))
    s = zero(logits[1])
    for z in logits
        s += exp(z - maxv)
    end
    return log(s) + maxv
end

"""
    softmax(logits) -> Vector

Softmax estable numéricamente (devuelve probabilidades).
Compatible con Dual (la selección del max usa valor primal).
"""
function softmax(logits::AbstractVector)
    @assert length(logits) > 0
    maxv = maximum(value.(logits))
    exps = similar(logits, length(logits))
    s = zero(logits[1])
    @inbounds for i in eachindex(logits)
        exps[i] = exp(logits[i] - maxv)
        s += exps[i]
    end
    @inbounds for i in eachindex(exps)
        exps[i] = exps[i] / s
    end
    return exps
end

"""
    nll_from_logits(logits, y)

Negative log-likelihood (NLL) para una etiqueta y (1..C), desde logits.
Implementación estable: NLL = logsumexp(logits) - logits[y].
Compatible con Dual.
"""
function nll_from_logits(logits::AbstractVector, y::Integer)
    @assert 1 <= y <= length(logits)
    return logsumexp(logits) - logits[y]
end

"""
    cross_entropy_from_logits(logits, y)

Alias de NLL para una sola observación.
"""
cross_entropy_from_logits(logits::AbstractVector, y::Integer) = nll_from_logits(logits, y)

"""
    accuracy_from_logits(logits_mat, y)

Calcula accuracy dada una matriz de logits y un vector de etiquetas.
- logits_mat: (C, N) o (N, C) se detecta automáticamente si `size` cuadra con length(y).
- y: etiquetas enteras en 1..C
"""
function accuracy_from_logits(logits_mat::AbstractArray, y::AbstractVector{<:Integer})
    N = length(y)
    sz = size(logits_mat)
    @assert length(sz) == 2 "logits_mat debe ser 2D"

    if sz[2] == N
        # (C, N)
        C = sz[1]
        correct = 0
        @inbounds for j in 1:N
            # argmax en columna j (por valor primal)
            best_i = 1
            best_v = value(logits_mat[1, j])
            for i in 2:C
                v = value(logits_mat[i, j])
                if v > best_v
                    best_v = v
                    best_i = i
                end
            end
            correct += (best_i == y[j]) ? 1 : 0
        end
        return correct / N
    elseif sz[1] == N
        # (N, C)
        C = sz[2]
        correct = 0
        @inbounds for j in 1:N
            best_i = 1
            best_v = value(logits_mat[j, 1])
            for i in 2:C
                v = value(logits_mat[j, i])
                if v > best_v
                    best_v = v
                    best_i = i
                end
            end
            correct += (best_i == y[j]) ? 1 : 0
        end
        return correct / N
    else
        error("No puedo inferir la orientación de logits_mat: size=$(sz), length(y)=$N")
    end
end

end # module Metrics