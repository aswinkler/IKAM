# src/model/OneLayerInterModel.jl
module InterModel

using ..InterLayers
using ..Metrics: cross_entropy_from_logits

export OneLayerInterModel, forward, forward_batch, loss, loss_batch

struct OneLayerInterModel{T<:Real}
    layer::InterLayers.KAInterLayer{T}
end

forward(m::OneLayerInterModel{T}, x::AbstractVector) where {T<:Real} =
    InterLayers.forward(m.layer, x)

forward_batch(m::OneLayerInterModel{T}, X::AbstractMatrix) where {T<:Real} =
    InterLayers.forward_batch(m.layer, X)

loss(m::OneLayerInterModel{T}, x::AbstractVector, y::Integer) where {T<:Real} =
    cross_entropy_from_logits(forward(m, x), y)

function loss_batch(m::OneLayerInterModel{T}, X::AbstractMatrix, y::AbstractVector{<:Integer}) where {T<:Real}
    @assert size(X,2) == length(y)
    logits = forward_batch(m, X)     # (C, N)
    N = length(y)
    acc = zero(logits[1,1])
    @inbounds for j in 1:N
        acc += cross_entropy_from_logits(view(logits, :, j), y[j])
    end
    return acc / N
end

end # module InterModel