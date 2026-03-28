# src/ad/Dual.jl

module AD

export Dual, dual, value, deriv, nderiv, seed!, unseed

import Base: +, -, *, /, ^, convert, promote_rule, show,
             zero, one, isfinite, abs, exp, log, sqrt, inv,
             isless, <, <=, >, >=

"""
    Dual{T}

Número dual para forward-AD con derivada vectorial.
- `val`: valor primal
- `d`: vector de derivadas ∂/∂θ_j
"""
#struct Dual{T<:Real} <: Number
struct Dual{T<:Real} <: Real
    val::T
    d::Vector{T}
end

Dual{T}(v::T) where {T<:Real} = Dual{T}(v, T[])
Dual(v::T) where {T<:Real} = Dual{T}(v, T[])

# --- Constructores y utilidades ---

value(x::Dual) = x.val
value(x::Real) = x

deriv(x::Dual) = x.d
deriv(::Real) = nothing

nderiv(x::Dual) = length(x.d)

dual(val::T, d::Vector{T}) where {T<:Real} = Dual{T}(val, d)

"""
    seed!(x, j, P)

Crea un Dual con `P` derivadas, sembrando 1.0 en la posición `j`.
"""
function seed!(x::T, j::Int, P::Int) where {T<:Real}
    d = zeros(T, P)
    d[j] = one(T)
    return Dual{T}(x, d)
end

"""
    unseed(x)

Extrae el valor primal de un Dual (o deja el real intacto).
"""
unseed(x::Dual) = x.val
unseed(x::Real) = x

# --- Promoción y conversión ---
promote_rule(::Type{Dual{T}}, ::Type{S}) where {T<:Real,S<:Real} = Dual{promote_type(T,S)}
promote_rule(::Type{Dual{T}}, ::Type{Dual{S}}) where {T<:Real,S<:Real} = Dual{promote_type(T,S)}

convert(::Type{Dual{T}}, x::Real) where {T<:Real} = Dual{T}(convert(T, x), T[])
convert(::Type{T}, x::Dual{T}) where {T<:Real} = x.val

# Nota: Para convertir Real -> Dual con derivadas de dimensión P, se requiere contexto;
# por eso, aquí convert crea d = [] (vacío). En el entrenamiento, siempre siembra con seed!.

# --- show ---
function show(io::IO, x::Dual)
    print(io, "Dual(", x.val, ", d=", x.d, ")")
end

# --- zero/one ---
# zero(x::Dual{T}) where {T<:Real} = Dual{T}(zero(T), zeros(T, length(x.d)))
zero(x::Dual{T}) where {T<:Real} = Dual(zero(x.val), zeros(T, length(x.d)))
zero(::Type{Dual{T}}) where {T<:Real} = Dual(zero(T), T[])

one(x::Dual{T})  where {T<:Real} = Dual{T}(one(T),  zeros(T, length(x.d)))

# --- Comparación (para branches) ---
# Ordenamos por valor primal; esto permite searchsorted/max, etc.
# isless(a::Dual, b::Dual) = isless(a.val, b.val)
# isless(a::Dual, b::Real) = isless(a.val, b)
# isless(a::Real, b::Dual) = isless(a, b.val)

isless(a::Dual{T}, b::Dual{T}) where {T<:Real} = isless(a.val, b.val)
isless(a::Dual{T}, b::Real)    where {T<:Real} = isless(a.val, b)
isless(a::Real,    b::Dual{T}) where {T<:Real} = isless(a, b.val)

<(a::Dual{T}, b::Dual{T}) where {T<:Real} = a.val < b.val
<(a::Dual{T}, b::Real)    where {T<:Real} = a.val < b
<(a::Real,    b::Dual{T}) where {T<:Real} = a < b.val

<=(a::Dual{T}, b::Dual{T}) where {T<:Real} = a.val <= b.val
<=(a::Dual{T}, b::Real)    where {T<:Real} = a.val <= b
<=(a::Real,    b::Dual{T}) where {T<:Real} = a <= b.val

>(a::Dual{T}, b::Dual{T}) where {T<:Real} = a.val > b.val
>(a::Dual{T}, b::Real)    where {T<:Real} = a.val > b
>(a::Real,    b::Dual{T}) where {T<:Real} = a > b.val

>=(a::Dual{T}, b::Dual{T}) where {T<:Real} = a.val >= b.val
>=(a::Dual{T}, b::Real)    where {T<:Real} = a.val >= b
>=(a::Real,    b::Dual{T}) where {T<:Real} = a >= b.val





# --- Operaciones básicas ---
+(a::Dual{T}, b::Dual{T}) where {T<:Real} = Dual{T}(a.val + b.val, a.d .+ b.d)
+(a::Dual{T}, b::Real)    where {T<:Real} = Dual{T}(a.val + T(b), copy(a.d))
+(a::Real, b::Dual{T})    where {T<:Real} = b + a

-(a::Dual{T}) where {T<:Real} = Dual{T}(-a.val, .-(a.d))
-(a::Dual{T}, b::Dual{T}) where {T<:Real} = Dual{T}(a.val - b.val, a.d .- b.d)
-(a::Dual{T}, b::Real)    where {T<:Real} = Dual{T}(a.val - T(b), copy(a.d))
-(a::Real, b::Dual{T})    where {T<:Real} = Dual{T}(T(a) - b.val, .-(b.d))

*(a::Dual{T}, b::Dual{T}) where {T<:Real} = Dual{T}(a.val * b.val, a.d .* b.val .+ b.d .* a.val)
*(a::Dual{T}, b::Real)    where {T<:Real} = Dual{T}(a.val * T(b), a.d .* T(b))
*(a::Real, b::Dual{T})    where {T<:Real} = b * a

inv(a::Dual{T}) where {T<:Real} = Dual{T}(inv(a.val), .-(a.d) ./ (a.val*a.val))

/(a::Dual{T}, b::Dual{T}) where {T<:Real} = a * inv(b)
/(a::Dual{T}, b::Real)    where {T<:Real} = Dual{T}(a.val / T(b), a.d ./ T(b))
/(a::Real, b::Dual{T})    where {T<:Real} = Dual{T}(T(a), zeros(T, length(b.d))) / b

# Potencias: usamos caso Real exponente (suficiente para lo actual)
^(a::Dual{T}, p::Real) where {T<:Real} = begin
    pp = T(p)
    v = a.val^pp
    Dual{T}(v, a.d .* (pp * (a.val^(pp - one(T)))))
end

# --- Funciones elementales ---
abs(a::Dual{T}) where {T<:Real} = (a.val >= 0) ? a : Dual{T}(-a.val, .-(a.d))

isfinite(a::Dual) = isfinite(a.val)

exp(a::Dual{T}) where {T<:Real} = begin
    v = exp(a.val)
    Dual{T}(v, a.d .* v)
end

log(a::Dual{T}) where {T<:Real} = Dual{T}(log(a.val), a.d ./ a.val)

sqrt(a::Dual{T}) where {T<:Real} = begin
    v = sqrt(a.val)
    Dual{T}(v, a.d ./ (2v))
end

end # module AD