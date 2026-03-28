module Layers

using Random
using ..AD: Dual, value
using ..Inner: PWLinearNodal, evaluate, uniform_knots

export KALayer, init_kalayer, forward, forward_batch,
       set_affine!, set_identity_phi!

"""
    KALayer{T}

Capa Kolmogorov-consistente (una capa) con:
- Q = 2n + 1 canales externos
- Φ_{i,q}(t) = A[i,q] * t + B[i,q] (afín, sin activación)
- φ_{i,q,p} piecewise Newton grado 1 por nodos (PWLinearNodal)

Estructura:
- n_in: dimensión de entrada
- m_out: dimensión de salida (logits)
- Q: canales externos (por default 2n+1)
- K: #intervalos de φ (K+1 nudos)
- A, B: matrices (m_out × Q)
- phi: arreglo 3D (m_out × Q × n_in) de PWLinearNodal
"""
struct KALayer{T<:Real}
    n_in::Int
    m_out::Int
    Q::Int
    K::Int
    A::Matrix{T}  # (m_out, Q)
    B::Matrix{T}  # (m_out, Q)
    phi::Array{PWLinearNodal{T}, 3}  # (m_out, Q, n_in)
end

"""
    init_kalayer(n_in, m_out; K=8, rng=MersenneTwister(0), scale=0.05, Q=2n_in+1, T=Float64)

Inicializa una capa KA:
- A ~ Normal(0, scale), B = 0
- φ por nodos con nudos uniformes; valores ~ Normal(0, scale)
"""
function init_kalayer(n_in::Int, m_out::Int;
                      K::Int=8,
                      rng::AbstractRNG=MersenneTwister(0),
                      scale::Real=0.05,
                      Q::Int=2n_in+1,
                      T::Type{<:Real}=Float64)

    @assert n_in > 0 && m_out > 0
    @assert K >= 1
    @assert Q >= 1

    A = Matrix{T}(undef, m_out, Q)
    B = zeros(T, m_out, Q)

    # init A con randn escalado
    @inbounds for i in 1:m_out, q in 1:Q
        A[i, q] = T(scale) * randn(rng)
    end

    knots = uniform_knots(T, K)  # longitud K+1

    phi = Array{PWLinearNodal{T}, 3}(undef, m_out, Q, n_in)
    @inbounds for i in 1:m_out, q in 1:Q, p in 1:n_in
        vals = Vector{T}(undef, K+1)
        for k in 1:(K+1)
            vals[k] = T(scale) * randn(rng)
        end
        phi[i, q, p] = PWLinearNodal{T}(knots, vals)
    end

    return KALayer{T}(n_in, m_out, Q, K, A, B, phi)
end

"""
    set_affine!(layer; A=nothing, B=nothing, a=nothing, b=nothing)

Conveniencia para fijar parámetros afines:
- Si se da `a::Real`, llena A con ese valor.
- Si se da `b::Real`, llena B con ese valor.
- O puedes pasar matrices completas `A`, `B`.
"""
function set_affine!(layer::KALayer{T};
                     A=nothing, B=nothing,
                     a=nothing, b=nothing) where {T<:Real}
    if A !== nothing
        @assert size(A) == size(layer.A)
        layer.A .= T.(A)
    elseif a !== nothing
        layer.A .= T(a)
    end

    if B !== nothing
        @assert size(B) == size(layer.B)
        layer.B .= T.(B)
    elseif b !== nothing
        layer.B .= T(b)
    end
    return layer
end

"""
    set_identity_phi!(layer)

Fija todas las φ_{i,q,p}(x) = x en [0,1] usando valores nodales y_k = knot_k.
Esto es útil para pruebas analíticas.
"""
function set_identity_phi!(layer::KALayer{T}) where {T<:Real}
    knots = uniform_knots(T, layer.K)
    vals = copy(knots)  # y_k = x_k
    @inbounds for i in 1:layer.m_out, q in 1:layer.Q, p in 1:layer.n_in
        layer.phi[i, q, p] = PWLinearNodal{T}(knots, copy(vals))
    end
    return layer
end

"""
    forward(layer, x) -> Vector

Evalúa la capa para una sola muestra:
- x: vector de longitud n_in (Float64 o Dual o mezcla compatible)
- salida: vector longitud m_out (logits)
"""
function forward(layer::KALayer{T}, x::AbstractVector) where {T<:Real}
    @assert length(x) == layer.n_in

    # Cero que preserva la dimensión de derivadas si z[1]es Dual
    z0 = x[1] - x[1]
    
    # Tipo de salida: usar tipo del acumulador (puede ser Dual si x lo es)
    proto = z0 + layer.A[1,1] + layer.B[1,1]
    Tout = typeof(proto)
    out  = Vector{Tout}(undef, layer.m_out)
    # out = Vector{typeof(x[1] + layer.A[1,1] + layer.B[1,1])}(undef, layer.m_out)

    @inbounds for i in 1:layer.m_out
        # acc_i = zero(out[1])
        acc_i = z0
        for q in 1:layer.Q
            # s = zero(out[1])
            s = z0
            for p in 1:layer.n_in
                s += evaluate(layer.phi[i, q, p], x[p])
            end
            acc_i += layer.A[i, q] * s + layer.B[i, q]
        end
        out[i] = acc_i
    end
    return out
end

"""
    forward_batch(layer, X) -> Matrix

Evalúa sobre un batch full-batch:
- X: (n_in, N)
- retorna logits: (m_out, N)
"""
function forward_batch(layer::KALayer{T}, X::AbstractMatrix) where {T<:Real}
    @assert size(X, 1) == layer.n_in
    N = size(X, 2)
    # Cero que preserva la dimensión de derivadas si z[1]es Dual
    z0 = X[1,1]-X[1,1]

    proto = z0 + layer.A[1,1] + layer.B[1,1]
    Tout = typeof(proto)

    logits = Matrix{Tout}(undef, layer.m_out, N)

    @inbounds for j in 1:N
        xj = @view X[:, j]
        yj = forward(layer, xj)
        for i in 1:layer.m_out
            logits[i, j] = yj[i]
        end
    end
    return logits
end

end # module Layers