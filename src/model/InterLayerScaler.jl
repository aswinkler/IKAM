module InterLayer

using ..AD: value
using ..MathUtils: clamp01
export InterLayerMinMax, fit_interlayer_minmax, transform_interlayer

struct InterLayerMinMax{T<:Real}
    mins::Vector{T}
    maxs::Vector{T}
end

function fit_interlayer_minmax(Z::AbstractMatrix{T}; margin_factor::T = T(5)) where {T<:Real}
    d = size(Z, 1)
    mins = Vector{T}(undef, d)
    maxs = Vector{T}(undef, d)
    @inbounds for r in 1:d
        mn = minimum(@view Z[r, :])
        mx = maximum(@view Z[r, :])
        rng = mx - mn
        mins[r] = mn - margin_factor * rng
        maxs[r] = mx + margin_factor * rng
    end
    return InterLayerMinMax{T}(mins, maxs)
end

# Transformación suave (derivada 1/(max-min) si max!=min), sin clamp
function transform_interlayer(s::InterLayerMinMax{T}, z::AbstractVector) where {T<:Real}
    d = length(z)
    out = similar(z)
    @inbounds for r in 1:d
        mn = s.mins[r]
        mx = s.maxs[r]
        den = mx - mn
        if abs(den) < eps(T)
            out[r] = z[r] - z[r]   # 0 con dimensión correcta si Dual
        else
            u = (z[r] - mn) / den
            out[r] = clamp01(u)
        end
    end
    return out
end

end