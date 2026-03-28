# src/model/TwoLayerInterModel.jl
module TwoInterModel

using ..AD: value
using ..Data
using ..InterLayers
using ..Metrics: cross_entropy_from_logits

export TwoLayerInterModel, fit_hidden_scaler, forward, forward_batch, loss, loss_batch

mutable struct TwoLayerInterModel{T<:Real}
    layer1::InterLayers.KAInterLayer{T}
    layer2::InterLayers.KAInterLayer{T}
    hidden_scaler::Data.MinMaxScaler{T}
end

"""
    fit_hidden_scaler(layer1, X)

Ajusta un MinMaxScaler sobre la salida batch de la primera capa,
usando únicamente el conjunto de entrenamiento.
"""
function fit_hidden_scaler(layer1::InterLayers.KAInterLayer{T}, X::AbstractMatrix{T}) where {T<:Real}
    Z = InterLayers.forward_batch(layer1, X)   # (m_hidden, N)
    return Data.fit_minmax(Z)
end

@inline function _clamp01_same(u)
    uv = value(u)
    if uv < 0
        return zero(u)
    elseif uv > 1
        return one(u)
    else
        return u
    end
end

"""
    transform_hidden(scaler, z; clamp01=true, eps=1e-12)

Escala un vector oculto z mediante min-max componente a componente.
Compatible con Real y Dual.
"""
function transform_hidden(s::Data.MinMaxScaler{T}, z::AbstractVector; clamp01::Bool=true, eps::Real=1e-12) where {T<:Real}
    @assert length(z) == length(s.mins) == length(s.maxs)
    Tout = typeof(z[1] - z[1] + s.mins[1])
    out = Vector{Tout}(undef, length(z))

    @inbounds for i in eachindex(z)
        mn = s.mins[i]
        mx = s.maxs[i]
        den = mx - mn
        if den == zero(T)
            out[i] = zero(z[i])
        else
            v = (z[i] - mn) / (den + T(eps))
            out[i] = clamp01 ? _clamp01_same(v) : v
        end
    end
    return out
end

function forward(m::TwoLayerInterModel{T}, x::AbstractVector) where {T<:Real}
    z1  = InterLayers.forward(m.layer1, x)
    z1n = transform_hidden(m.hidden_scaler, z1; clamp01=true)
    return InterLayers.forward(m.layer2, z1n)
end

function forward_batch(m::TwoLayerInterModel{T}, X::AbstractMatrix) where {T<:Real}
    @assert size(X,1) == m.layer1.n_in
    N = size(X,2)

    z0 = X[1,1] - X[1,1]
    proto = z0 + m.layer2.A[1,1] + m.layer2.B[1,1]
    Tout = typeof(proto)

    logits = Matrix{Tout}(undef, m.layer2.m_out, N)
    @inbounds for j in 1:N
        xj = @view X[:,j]
        yj = forward(m, xj)
        for i in 1:m.layer2.m_out
            logits[i,j] = yj[i]
        end
    end
    return logits
end

loss(m::TwoLayerInterModel{T}, x::AbstractVector, y::Integer) where {T<:Real} =
    cross_entropy_from_logits(forward(m, x), y)

function loss_batch(m::TwoLayerInterModel{T}, X::AbstractMatrix, y::AbstractVector{<:Integer}) where {T<:Real}
    @assert size(X,2) == length(y)
    logits = forward_batch(m, X)
    N = length(y)
    acc = zero(logits[1,1])
    @inbounds for j in 1:N
        acc += cross_entropy_from_logits(view(logits, :, j), y[j])
    end
    return acc / N
end

end # module TwoInterModel