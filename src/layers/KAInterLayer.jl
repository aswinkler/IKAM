# src/layers/KAInterLayer.jl

module InterLayers

using Random
using ..AD: Dual, value
using ..Inner: PWLinearNodal, evaluate, uniform_knots

export KAInterLayer, init_kainterlayer, forward, forward_batch, with_lambda,
       set_affine!, set_identity_phi!


"""
    KAInterLayer{T}

Capa Kolmogorov-consistente con interacción controlada de orden 2:

Para cada salida i:
y_i(x) = sum_{q=1}^Q [ A[i,q]*s_{i,q}(x) + B[i,q] + λ * I_{i,q}(x) ]

donde:
s_{i,q}(x) = sum_{p=1}^{n_in} φ_{i,q,p}(x_p)

I_{i,q}(x) = sum_{(p,r) in I2} C[i,q,p,r] * φ_{i,q,p}(x_p) * φ_{i,q,r}(x_r)

- φ: PWLinearNodal grado 1 por nodos en [0,1]
- Φ externa: afín (A,B)
- I2: conjunto (lista) de pares elegibles (p<r)
"""
struct KAInterLayer{T<:Real}
    n_in::Int
    m_out::Int
    Q::Int
    K::Int
    λ::T
    I2::Vector{Tuple{Int,Int}}          # pares (p,r), 1 <= p < r <= n_in
    A::Matrix{T}                        # (m_out, Q)
    B::Matrix{T}                        # (m_out, Q)
    C::Array{T,4}                       # (m_out, Q, n_in, n_in) (solo usamos p<r)
    phi::Array{PWLinearNodal{T}, 3}     # (m_out, Q, n_in)
end

"""
    init_kainterlayer(n_in, m_out; K=8, rng=MersenneTwister(0), scale=0.05,
                      Q=2n_in+1, λ=1.0, I2=nothing, T=Float64)

Inicializa KAInterLayer:
- A ~ Normal(0, scale), B = 0
- C ~ Normal(0, scale) (solo relevante para pares en I2; el resto puede quedar en 0)
- φ por nodos uniformes; valores ~ Normal(0, scale)

I2 por defecto: todos los pares (p,r), p<r.
"""
function init_kainterlayer(n_in::Int, m_out::Int;
                           K::Int=8,
                           rng::AbstractRNG=MersenneTwister(0),
                           scale::Real=0.05,
                           Q::Int=2n_in+1,
                           λ::Real=1.0,
                           I2=nothing,
                           T::Type{<:Real}=Float64)

    @assert n_in > 0 && m_out > 0
    @assert K >= 1
    @assert Q >= 1

    # pares por defecto
    I2pairs = I2 === nothing ? [(p,r) for p in 1:n_in for r in (p+1):n_in] : I2
    @assert all(p < r for (p,r) in I2pairs)
    @assert all(1 <= p <= n_in && 1 <= r <= n_in for (p,r) in I2pairs)

    A = Matrix{T}(undef, m_out, Q)
    B = zeros(T, m_out, Q)

    @inbounds for i in 1:m_out, q in 1:Q
        A[i,q] = T(scale) * randn(rng)
    end

    # C completo, pero solo usaremos p<r en I2
    C = zeros(T, m_out, Q, n_in, n_in)
    @inbounds for i in 1:m_out, q in 1:Q
        for (p,r) in I2pairs
            C[i,q,p,r] = T(scale) * randn(rng)
        end
    end

    knots = uniform_knots(T, K)

    phi = Array{PWLinearNodal{T}, 3}(undef, m_out, Q, n_in)
    @inbounds for i in 1:m_out, q in 1:Q, p in 1:n_in
        vals = Vector{T}(undef, K+1)
        for k in 1:(K+1)
            vals[k] = T(scale) * randn(rng)
        end
        phi[i,q,p] = PWLinearNodal{T}(knots, vals)
    end

    return KAInterLayer{T}(n_in, m_out, Q, K, T(λ), I2pairs, A, B, C, phi)
end

# --- utilidades opcionales (muy útiles en tests / sanity checks) ---

"""
    set_identity_phi!(layer::KAInterLayer)

Fija todas φ_{i,q,p}(x)=x en [0,1] usando y_k = knot_k.
"""
function set_identity_phi!(layer::KAInterLayer{T}) where {T<:Real}
    knots = uniform_knots(T, layer.K)
    vals = copy(knots)
    @inbounds for i in 1:layer.m_out, q in 1:layer.Q, p in 1:layer.n_in
        layer.phi[i,q,p] = PWLinearNodal{T}(knots, copy(vals))
    end
    return layer
end

"""
    set_affine!(layer::KAInterLayer; A=nothing, B=nothing, a=nothing, b=nothing)
Conveniencia para fijar parámetros afines.
"""
function set_affine!(layer::KAInterLayer{T};
                     A=nothing, B=nothing, a=nothing, b=nothing) where {T<:Real}
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
    forward(layer::KAInterLayer, x) -> Vector

Evalúa la capa para una muestra.
Compatible con Dual: sólo usa suma y productos.
"""
function forward(layer::KAInterLayer{T}, x::AbstractVector) where {T<:Real}
    @assert length(x) == layer.n_in

    z0 = x[1] - x[1]
    proto = z0 + layer.A[1,1] + layer.B[1,1]
    Tout = typeof(proto)
    out = Vector{Tout}(undef, layer.m_out)

    # buffer para φ_p (evita recomputar para interacción)
    φvals = Vector{Tout}(undef, layer.n_in)
    Lpairs = length(layer.I2)  # <-- AÑADE esto (se usa para normalizar)

    @inbounds for i in 1:layer.m_out
        acc_i = z0
        for q in 1:layer.Q
            s = z0
            for p in 1:layer.n_in
                vp = evaluate(layer.phi[i,q,p], x[p])
                φvals[p] = vp
                s += vp
            end

            # término afín
            acc_i += layer.A[i,q] * s + layer.B[i,q]

            # interacción (controlada por I2)
            if layer.λ != zero(T)
                inter = z0
                for (p,r) in layer.I2
                    inter += layer.C[i,q,p,r] * φvals[p] * φvals[r]
                end

                # --- NORMALIZACIÓN (cambio clave) ---
                if Lpairs > 0
                    inter /= T(Lpairs)
                end

                acc_i += layer.λ * inter
            end
        end
        out[i] = acc_i
    end
    return out
end

"""
    forward_batch(layer::KAInterLayer, X) -> Matrix

Evalúa batch: X (n_in, N) -> logits (m_out, N)
"""
function forward_batch(layer::KAInterLayer{T}, X::AbstractMatrix) where {T<:Real}
    @assert size(X,1) == layer.n_in
    N = size(X,2)

    z0 = X[1,1] - X[1,1]
    proto = z0 + layer.A[1,1] + layer.B[1,1]
    Tout = typeof(proto)

    logits = Matrix{Tout}(undef, layer.m_out, N)
    @inbounds for j in 1:N
        xj = @view X[:,j]
        yj = forward(layer, xj)
        for i in 1:layer.m_out
            logits[i,j] = yj[i]
        end
    end
    return logits
end

"""
    with_lambda(layer, λnew)

Devuelve una copia de `layer` con el mismo contenido pero con λ = λnew.
No muta el layer (útil porque KAInterLayer es inmutable).
"""

function with_lambda(layer::KAInterLayer{T}, λ::Real) where {T<:Real}
    # Copias profundas para evitar aliasing entre corridas del sweep
    A = copy(layer.A)
    B = copy(layer.B)
    C = copy(layer.C)

    # Copiar phi: mismo tipo, pero con copia de values/knots por elemento
    phi = similar(layer.phi)
    @inbounds for i in eachindex(layer.phi)
        ph = layer.phi[i]
        # Asumo PWLinearNodal(knots, values)
        phi[i] = PWLinearNodal{T}(copy(ph.knots), copy(ph.values))
    end

    return KAInterLayer{T}(layer.n_in, layer.m_out, layer.Q, layer.K,
                          T(λ), copy(layer.I2),
                          A, B, C, phi)
end

end # module Layers