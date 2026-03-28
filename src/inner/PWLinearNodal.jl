# src/inner/PWLinearNodal.jl
module Inner

using ..AD: Dual, value

export PWLinearNodal, evaluate, uniform_knots

"""
    PWLinearNodal{T}

Interpolación lineal por tramos definida por:
- knots: vector creciente en [0,1], longitud K+1
- values: valores en nudos, longitud K+1 (entrenables cuando toque)
"""
struct PWLinearNodal{T<:Real}
    knots::Vector{T}
    values::Vector{T}
    function PWLinearNodal{T}(knots::Vector{T}, values::Vector{T}) where {T<:Real}
        @assert length(knots) == length(values) "knots y values deben tener la misma longitud"
        @assert length(knots) >= 2 "se requieren al menos 2 nudos"
        @assert all(diff(knots) .> 0) "knots debe ser estrictamente creciente"
        return new{T}(knots, values)
    end
end

"""
    uniform_knots(T, K)

Genera K intervalos uniformes en [0,1] => K+1 nudos.
"""
function uniform_knots(::Type{T}, K::Int) where {T<:Real}
    @assert K >= 1
    knots = collect(range(zero(T), one(T), length=K+1))
    return knots
end

# Obtiene índice de tramo k tal que knots[k] <= x <= knots[k+1], con k en 1..K
# Para Dual, usamos value(x) para decidir el tramo.
function _segment_index(knots::Vector{T}, x) where {T<:Real}
    xv = value(x)
    if xv <= knots[1]
        return 1
    elseif xv >= knots[end]
        return length(knots) - 1
    else
        # searchsortedlast devuelve índice i tal que knots[i] <= xv < knots[i+1]
        i = searchsortedlast(knots, T(xv))
        return clamp(i, 1, length(knots)-1)
    end
end

"""
    evaluate(pw, x)

Evalúa la interpolación lineal por tramos (compatible con Dual).
"""
function evaluate(pw::PWLinearNodal{T}, x) where {T<:Real}
    k = _segment_index(pw.knots, x)

    x0 = pw.knots[k]
    x1 = pw.knots[k+1]
    y0 = pw.values[k]
    y1 = pw.values[k+1]

    # Interpolación lineal: y = y0 + (y1-y0) * (x-x0)/(x1-x0)
    t = (x - x0) / (x1 - x0)
    return y0 + (y1 - y0) * t
end

end # module Inner