module Model

using ..Layers: KALayer
import ..Layers: forward, forward_batch
using ..Metrics: cross_entropy_from_logits, accuracy_from_logits
using ..AD: value

export OneLayerModel, forward, forward_batch, loss, loss_batch,
       predict, predict_batch, accuracy

"""
    OneLayerModel{T}

Modelo de una sola capa KA que produce logits (m_out = #clases).
La pérdida es cross-entropy desde logits (softmax implícita en la pérdida).
"""
struct OneLayerModel{T<:Real}
    layer::KALayer{T}
end

# --- Forward ---
forward(m::OneLayerModel{T}, x::AbstractVector) where {T<:Real} = forward(m.layer, x)
forward_batch(m::OneLayerModel{T}, X::AbstractMatrix) where {T<:Real} = forward_batch(m.layer, X)

# --- Loss (single) ---
loss(m::OneLayerModel{T}, x::AbstractVector, y::Integer) where {T<:Real} =
    cross_entropy_from_logits(forward(m, x), y)

# --- Loss (batch) ---
function loss_batch(m::OneLayerModel{T}, X::AbstractMatrix, y::AbstractVector{<:Integer}) where {T<:Real}
    @assert size(X, 2) == length(y)
    logits = forward_batch(m, X)  # (C, N)
    N = length(y)
    acc = zero(logits[1,1])
    @inbounds for j in 1:N
        acc += cross_entropy_from_logits(view(logits, :, j), y[j])
    end
    return acc / N
end

# --- Predict ---
function predict(m::OneLayerModel{T}, x::AbstractVector) where {T<:Real}
    z = forward(m, x)
    # argmax por valor primal (soporta Dual)
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

function predict_batch(m::OneLayerModel{T}, X::AbstractMatrix) where {T<:Real}
    logits = forward_batch(m, X) # (C, N)
    N = size(X, 2)
    yhat = Vector{Int}(undef, N)
    C = size(logits, 1)
    @inbounds for j in 1:N
        best_i = 1
        best_v = value(logits[1, j])
        for i in 2:C
            v = value(logits[i, j])
            if v > best_v
                best_v = v
                best_i = i
            end
        end
        yhat[j] = best_i
    end
    return yhat
end

# --- Accuracy ---
accuracy(m::OneLayerModel{T}, X::AbstractMatrix, y::AbstractVector{<:Integer}) where {T<:Real} =
    accuracy_from_logits(forward_batch(m, X), y)

end # module Model