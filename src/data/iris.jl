module Data

using Random
using MLDatasets
import DataFrames
using ..AD: value

export IrisSplit, load_iris_split,
       MinMaxScaler, fit_minmax, transform_minmax!, transform_minmax,
       stratified_split_indices

"""
    MinMaxScaler{T}

Escalador min-max por feature (por fila), con opción de clamping a [0,1].
- mins, maxs: vectores de longitud n_features
"""
struct MinMaxScaler{T<:Real}
    mins::Vector{T}
    maxs::Vector{T}
end

"""
    fit_minmax(X)

Ajusta min-max por feature (X es n_features × N).
Devuelve MinMaxScaler.
"""
function fit_minmax(X::AbstractMatrix{T}) where {T<:Real}
    nfeat = size(X, 1)
    mins = Vector{T}(undef, nfeat)
    maxs = Vector{T}(undef, nfeat)
    @inbounds for i in 1:nfeat
        mins[i] = minimum(@view X[i, :])
        maxs[i] = maximum(@view X[i, :])
    end
    return MinMaxScaler{T}(mins, maxs)
end

"""
    transform_minmax(scaler, X; clamp01=true)

Transforma X con (x-min)/(max-min) por feature.
Si clamp01=true, fuerza el resultado a [0,1] para respetar el dominio del modelo.
Devuelve una nueva matriz.
"""
function transform_minmax(s::MinMaxScaler{T}, X::AbstractMatrix{T}; clamp01::Bool=true) where {T<:Real}
    Y = similar(X)
    transform_minmax!(Y, s, X; clamp01=clamp01)
    return Y
end

"""
    transform_minmax!(Y, scaler, X; clamp01=true)

Versión in-place: Y := transform(X).
"""
function transform_minmax!(Y::AbstractMatrix{T}, s::MinMaxScaler{T}, X::AbstractMatrix{T}; clamp01::Bool=true) where {T<:Real}
    @assert size(Y) == size(X)
    nfeat, N = size(X)
    @assert length(s.mins) == nfeat
    @assert length(s.maxs) == nfeat

    @inbounds for i in 1:nfeat
        mn = s.mins[i]
        mx = s.maxs[i]
        den = mx - mn
        if den == zero(T)
            # feature constante -> 0
            for j in 1:N
                Y[i, j] = zero(T)
            end
        else
            invden = one(T) / den
            for j in 1:N
                v = (X[i, j] - mn) * invden
                if clamp01
                    v = ifelse(v < zero(T), zero(T), ifelse(v > one(T), one(T), v))
                end
                Y[i, j] = v
            end
        end
    end
    return Y
end

"""
    IrisSplit

Estructura de datos del split Iris (full-batch):
- X*: matrices n_features × N*
- y*: etiquetas Int en 1..C
- scaler: minmax ajustado en train
- idx*: índices del dataset original
"""
struct IrisSplit{T<:Real}
    X_train::Matrix{T}
    y_train::Vector{Int}
    X_val::Matrix{T}
    y_val::Vector{Int}
    X_test::Matrix{T}
    y_test::Vector{Int}
    scaler::MinMaxScaler{T}
    idx_train::Vector{Int}
    idx_val::Vector{Int}
    idx_test::Vector{Int}
    classes::Vector{String}  # nombres originales
end

"""
    stratified_split_indices(y; ratios=(0.7,0.15,0.15), rng=MersenneTwister(0))

Devuelve (idx_train, idx_val, idx_test) preservando proporciones por clase.
y debe ser Vector{Int} con clases 1..C.
"""
function stratified_split_indices(y::AbstractVector{<:Integer};
                                  ratios::NTuple{3,Float64}=(0.7,0.15,0.15),
                                  rng::AbstractRNG=MersenneTwister(0))
    @assert isapprox(sum(ratios), 1.0; atol=1e-12)
    N = length(y)
    C = maximum(y)

    idx_train = Int[]
    idx_val   = Int[]
    idx_test  = Int[]

    for c in 1:C
        idx_c = findall(==(c), y)
        Random.shuffle!(rng, idx_c)

        n_c = length(idx_c)
        n_tr = round(Int, ratios[1] * n_c)
        n_va = round(Int, ratios[2] * n_c)
        # resto a test para cerrar exacto
        n_te = n_c - n_tr - n_va

        append!(idx_train, idx_c[1:n_tr])
        append!(idx_val,   idx_c[n_tr+1 : n_tr+n_va])
        append!(idx_test,  idx_c[n_tr+n_va+1 : end])
        @assert length(idx_c) == n_tr + n_va + n_te
    end

    # Mezclamos globalmente para evitar bloques por clase
    Random.shuffle!(rng, idx_train)
    Random.shuffle!(rng, idx_val)
    Random.shuffle!(rng, idx_test)

    # Sanidad: disjuntos y cubren todo
    @assert isempty(intersect(idx_train, idx_val))
    @assert isempty(intersect(idx_train, idx_test))
    @assert isempty(intersect(idx_val, idx_test))
    @assert length(idx_train) + length(idx_val) + length(idx_test) == N

    return (idx_train, idx_val, idx_test)
end

# --- Carga Iris desde MLDatasets y mapeo de etiquetas a 1..3 ---
function _load_iris_raw(; T::Type{<:Real}=Float64)
    # Según docs: Iris(as_df=false)[:] -> x: 4×150 Matrix{Float64}, y: 1×150 Matrix{InlineStrings...}
    x, y = MLDatasets.Iris(as_df=false)[:]  # :contentReference[oaicite:2]{index=2}
 
    X = Matrix{T}(x)                 # 4×150
    yraw = vec(y)                    # longitud 150, elementos tipo string/inline
    labels = String.(yraw)           # aseguramos String

    # Clases en Iris típicamente: Iris-setosa, Iris-versicolor, Iris-virginica
    classes = sort(unique(labels))
    @assert length(classes) == 3

    # map label -> Int
    mapdict = Dict{String,Int}(cls => i for (i, cls) in enumerate(classes))
    yint = [mapdict[lbl] for lbl in labels]
    @assert size(X, 1) == 4
    @assert size(X, 2) == 150
    @assert length(yint) == 150
    @assert all(1 .<= yint .<= 3)
    return X, yint, classes
end

"""
    load_iris_split(; ratios=(0.7,0.15,0.15), seed=0, clamp01=true, T=Float64)

Carga Iris, hace split estratificado, ajusta min-max en train y transforma train/val/test.
Devuelve IrisSplit.
"""
function load_iris_split(; ratios::NTuple{3,Float64}=(0.7,0.15,0.15),
                           seed::Integer=0,
                           clamp01::Bool=true,
                           T::Type{<:Real}=Float64)
    rng = MersenneTwister(seed)

    X, y, classes = _load_iris_raw(; T=T)
    # @show typeof(y) size(y)
    idx_tr, idx_va, idx_te = stratified_split_indices(y; ratios=ratios, rng=rng)

    Xtr = X[:, idx_tr]
    Xva = X[:, idx_va]
    Xte = X[:, idx_te]
    ytr = y[idx_tr]
    yva = y[idx_va]
    yte = y[idx_te]

    scaler = fit_minmax(Xtr)
    Xtrn = transform_minmax(scaler, Xtr; clamp01=clamp01)
    Xvan = transform_minmax(scaler, Xva; clamp01=clamp01)
    Xten = transform_minmax(scaler, Xte; clamp01=clamp01)

    return IrisSplit{T}(Xtrn, ytr, Xvan, yva, Xten, yte,
                        scaler, idx_tr, idx_va, idx_te, classes)
end

end # module Data