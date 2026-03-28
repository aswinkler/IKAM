using Flux
using Flux: onehotbatch, onecold, logitcrossentropy
using MLDatasets
using Statistics
using Random
using DataFrames   # <- necesario en tu entorno para Iris(as_df=false)

function stratified_split_indices(
    y_int::Vector{Int};
    ratios::NTuple{3,Float64} = (0.70, 0.15, 0.15),
    rng::AbstractRNG = MersenneTwister(123)
)
    @assert isapprox(sum(ratios), 1.0; atol=1e-12)

    idx_train = Int[]
    idx_val   = Int[]
    idx_test  = Int[]

    classes = sort(unique(y_int))

    for c in classes
        idx_c = findall(==(c), y_int)
        Random.shuffle!(rng, idx_c)

        n_c  = length(idx_c)
        n_tr = round(Int, ratios[1] * n_c)
        n_va = round(Int, ratios[2] * n_c)
        n_te = n_c - n_tr - n_va

        append!(idx_train, idx_c[1:n_tr])
        append!(idx_val,   idx_c[n_tr+1:n_tr+n_va])
        append!(idx_test,  idx_c[n_tr+n_va+1:end])

        @assert n_tr + n_va + n_te == n_c
    end

    Random.shuffle!(rng, idx_train)
    Random.shuffle!(rng, idx_val)
    Random.shuffle!(rng, idx_test)

    return idx_train, idx_val, idx_test
end

function accuracy(model, x, y_int)
    preds = onecold(model(x), 1:3)
    return mean(preds .== y_int)
end

function main()
    # ============================================================
    # 1. Load Iris
    # ============================================================
    x, y = Iris(as_df=false)[:]   # x: 4×150, y: 1×150
    X = Float32.(x)
    labels = vec(String.(y))

    classes = sort(unique(labels))
    label_to_int = Dict(cls => i for (i, cls) in enumerate(classes))
    y_int = [label_to_int[lbl] for lbl in labels]

    # ============================================================
    # 2. Stratified split 70/15/15
    # ============================================================
    rng = MersenneTwister(123)
    idx_tr, idx_va, idx_te = stratified_split_indices(y_int; rng=rng)

    Xtr = X[:, idx_tr]
    Xva = X[:, idx_va]
    Xte = X[:, idx_te]

    ytr = y_int[idx_tr]
    yva = y_int[idx_va]
    yte = y_int[idx_te]

    # ============================================================
    # 3. Normalize using training statistics only
    # ============================================================
    μ = mean(Xtr, dims=2)
    σ = std(Xtr, dims=2)
    σ = max.(σ, 1f-6)

    Xtr = (Xtr .- μ) ./ σ
    Xva = (Xva .- μ) ./ σ
    Xte = (Xte .- μ) ./ σ

    # ============================================================
    # 4. One-hot labels
    # ============================================================
    ytr_oh = onehotbatch(ytr, 1:3)
    yva_oh = onehotbatch(yva, 1:3)
    yte_oh = onehotbatch(yte, 1:3)

    # ============================================================
    # 5. Define model
    #    IMPORTANT: no softmax here; logitcrossentropy expects logits
    # ============================================================
    model = Chain(
        Dense(4 => 32, relu),
        Dense(32 => 3)
    )

    loss(m, x, y) = logitcrossentropy(m(x), y)

    # ============================================================
    # 6. Optimizer state (Flux current API)
    # ============================================================
    opt_rule = Flux.Descent(0.01)
    opt_state = Flux.setup(opt_rule, model)

    # ============================================================
    # 7. Training loop
    # ============================================================
    epochs = 200
    best_val_acc = -Inf
    best_epoch = 0

    for epoch in 1:epochs
        grads = Flux.gradient(model) do m
            loss(m, Xtr, ytr_oh)
        end

        Flux.update!(opt_state, model, grads[1])

        train_loss = loss(model, Xtr, ytr_oh)
        val_loss   = loss(model, Xva, yva_oh)
        test_loss  = loss(model, Xte, yte_oh)

        train_acc = accuracy(model, Xtr, ytr)
        val_acc   = accuracy(model, Xva, yva)
        test_acc  = accuracy(model, Xte, yte)

        if val_acc > best_val_acc
            best_val_acc = val_acc
            best_epoch = epoch
        end

        println(
            "Epoch $epoch | " *
            "train_loss=$(round(train_loss, digits=4)) " *
            "train_acc=$(round(train_acc, digits=4)) | " *
            "val_loss=$(round(val_loss, digits=4)) " *
            "val_acc=$(round(val_acc, digits=4)) | " *
            "test_loss=$(round(test_loss, digits=4)) " *
            "test_acc=$(round(test_acc, digits=4))"
        )
    end

    println()
    println("Best validation accuracy = $(round(best_val_acc, digits=4)) at epoch $best_epoch")
    println("Final train accuracy = $(round(accuracy(model, Xtr, ytr), digits=4))")
    println("Final validation accuracy = $(round(accuracy(model, Xva, yva), digits=4))")
    println("Final test accuracy = $(round(accuracy(model, Xte, yte), digits=4))")
end

main()