# src/train/SGD.jl
module Train

export SGD, sgd_step!, reset!

"""
    SGD(lr; momentum=0.0)

Optimizador SGD con momentum opcional.
- lr: learning rate
- momentum: 0.0 desactiva momentum
Internamente guarda buffers de velocidad por parámetro.
"""
mutable struct SGD{T<:Real}
    lr::T
    momentum::T
    v::Vector{Vector{T}}   # buffers por tensor/param (mismo shape que cada param vectorizado)
end

function SGD(lr::T; momentum::T=zero(T)) where {T<:Real}
    @assert lr > 0
    @assert momentum >= 0 && momentum < 1
    return SGD{T}(lr, momentum, Vector{Vector{T}}())
end

"""
    reset!(opt)

Limpia buffers (útil cuando cambias el set de parámetros entrenables entre fase 1 y fase 2).
"""
function reset!(opt::SGD)
    empty!(opt.v)
    return opt
end

"""
    sgd_step!(opt, params, grads)

Actualiza params in-place:
- params: Vector{Vector{T}} (cada elemento es un bloque de parámetros vectorizado)
- grads:  Vector{Vector{T}} del mismo tamaño
"""
function sgd_step!(opt::SGD{T}, params::Vector{Vector{T}}, grads::Vector{Vector{T}}) where {T<:Real}
    @assert length(params) == length(grads)

    # Inicializa buffers si hace falta
    if isempty(opt.v)
        opt.v = [zeros(T, length(p)) for p in params]
    else
        @assert length(opt.v) == length(params)
        @inbounds for i in eachindex(params)
            @assert length(opt.v[i]) == length(params[i])
        end
    end

    η = opt.lr
    μ = opt.momentum

    if μ == zero(T)
        @inbounds for i in eachindex(params)
            p = params[i]; g = grads[i]
            @assert length(p) == length(g)
            for k in eachindex(p)
                p[k] -= η * g[k]
            end
        end
    else
        @inbounds for i in eachindex(params)
            p = params[i]; g = grads[i]; v = opt.v[i]
            @assert length(p) == length(g) == length(v)
            for k in eachindex(p)
                v[k] = μ * v[k] + g[k]
                p[k] -= η * v[k]
            end
        end
    end

    return nothing
end

end # module Train