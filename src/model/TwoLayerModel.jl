module Model2

using ..Layers: KALayer
import ..Layers: forward, forward_batch
using ..InterLayer: InterLayerMinMax, transform_interlayer
using ..Metrics: cross_entropy_from_logits, accuracy_from_logits
using ..AD: value
using ..MathUtils: clamp01

export TwoLayerModel, TwoLayerModelScaled, clamp01, forward, forward_batch, loss, loss_batch,
       predict, accuracy

"""
    TwoLayerModel{T}

Modelo de dos capas KA:
- layer1: n_in -> d
- layer2: d -> m_out
Produce logits finales (m_out = #clases).
"""
struct TwoLayerModel{T<:Real}
    layer1::KALayer{T}
    layer2::KALayer{T}
end

struct TwoLayerModelScaled{T<:Real, S<:Real}
    layer1::KALayer{T}
    layer2::KALayer{T}
    scaler::InterLayerMinMax{S}
end

# Forward para una muestra
function forward(m::TwoLayerModel{T}, x::AbstractVector) where {T<:Real}
    z = forward(m.layer1, x)
    zproj = similar(z)
    #@inbounds for i in eachindex(z)
    #    zproj[i] = clamp01(z[i])
    #end

    return forward(m.layer2, z)
end

function forward(m::TwoLayerModelScaled{T}, x::AbstractVector) where {T<:Real}
    z = forward(m.layer1, x)
    z01 = transform_interlayer(m.scaler, z)
    return forward(m.layer2, z01)
end


# Forward batch: (n_in, N) -> (m_out, N)
function forward_batch(m::TwoLayerModel{T}, X::AbstractMatrix) where {T<:Real}
    @assert size(X, 1) == m.layer1.n_in
    N = size(X, 2)

    # logits intermedios: (d, N)
    Z = forward_batch(m.layer1, X)

    # Ahora aplicar layer2 columna por columna (d, N) -> (m_out, N)
    # (podemos optimizar después; aquí priorizamos claridad y corrección)
    logits = Matrix{typeof(Z[1,1] + m.layer2.A[1,1] + m.layer2.B[1,1])}(undef, m.layer2.m_out, N)
    # Reusar buffer para evitar el alloc por columna
    zproj = Vector{eltype(Z)}(undef, size(Z,1))
    @inbounds for j in 1:N
        zj = view(Z, :, j)
        #for r in eachindex(zproj)
        #    zproj[r] = clamp01(zj[r])
        #end
        yj = forward(m.layer2, zj)
        for i in 1:m.layer2.m_out
            logits[i, j] = yj[i]
        end
    end
    return logits
end


function forward_batch(m::TwoLayerModelScaled{T}, X::AbstractMatrix) where {T<:Real}
    @assert size(X, 1) == m.layer1.n_in
    N = size(X, 2)
    Z = forward_batch(m.layer1, X)  # (d, N)

    logits = Matrix{typeof(Z[1,1] + m.layer2.A[1,1] + m.layer2.B[1,1])}(undef, m.layer2.m_out, N)

    @inbounds for j in 1:N
        zj = @view Z[:, j]
        z01 = transform_interlayer(m.scaler, zj)
        yj = forward(m.layer2, z01)
        for i in 1:m.layer2.m_out
            logits[i, j] = yj[i]
        end
    end
    return logits
end

# Loss single
loss(m::TwoLayerModel{T}, x::AbstractVector, y::Integer) where {T<:Real} =
    cross_entropy_from_logits(forward(m, x), y)

loss(m::TwoLayerModelScaled{T}, x::AbstractVector, y::Integer) where {T<:Real} =
    cross_entropy_from_logits(forward(m, x), y)

    # Loss batch
function loss_batch(m::TwoLayerModel{T}, X::AbstractMatrix, y::AbstractVector{<:Integer}) where {T<:Real}
    @assert size(X, 2) == length(y)
    logits = forward_batch(m, X)  # (C, N)
    N = length(y)
    acc = zero(logits[1,1])
    @inbounds for j in 1:N
        acc += cross_entropy_from_logits(view(logits, :, j), y[j])
    end
    return acc / N
end

function loss_batch(m::TwoLayerModelScaled{T}, X::AbstractMatrix, y::AbstractVector{<:Integer}) where {T<:Real}
    @assert size(X, 2) == length(y)
    logits = forward_batch(m, X)  # (C, N)
    N = length(y)
    acc = zero(logits[1,1])
    @inbounds for j in 1:N
        acc += cross_entropy_from_logits(view(logits, :, j), y[j])
    end
    return acc / N
end

# Predict (single)
function predict(m::TwoLayerModel{T}, x::AbstractVector) where {T<:Real}
    z = forward(m, x)
    best_i = 1
    best_v = value(z[1])
    @inbounds for i in 2:length(z)
        v = value(z[i])
        if v > best_v
            best_v = v
            best_i = i
        end
    end
    return best_i
end

function predict(m::TwoLayerModelScaled{T}, x::AbstractVector) where {T<:Real}
    z = forward(m, x)
    best_i = 1
    best_v = value(z[1])
    @inbounds for i in 2:length(z)
        v = value(z[i])
        if v > best_v
            best_v = v
            best_i = i
        end
    end
    return best_i
end

# Accuracy
accuracy(m::TwoLayerModel{T}, X::AbstractMatrix, y::AbstractVector{<:Integer}) where {T<:Real} =
    accuracy_from_logits(forward_batch(m, X), y)
accuracy(m::TwoLayerModelScaled{T}, X::AbstractMatrix, y::AbstractVector{<:Integer}) where {T<:Real} =
    accuracy_from_logits(forward_batch(m, X), y)



end # module Model2