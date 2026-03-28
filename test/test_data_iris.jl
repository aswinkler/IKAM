using Test
using IKAM.Data

@testset "Iris data: split, minmax, reproducibility" begin
    split1 = load_iris_split(seed=123, ratios=(0.7,0.15,0.15))
    split2 = load_iris_split(seed=123, ratios=(0.7,0.15,0.15))
    split3 = load_iris_split(seed=124, ratios=(0.7,0.15,0.15))

    # Tamaños esperados (Iris: 150 instancias, 3 clases balanceadas)
    @test size(split1.X_train, 1) == 4
    @test size(split1.X_val, 1) == 4
    @test size(split1.X_test, 1) == 4

    @test length(split1.y_train) + length(split1.y_val) + length(split1.y_test) == 150

    # Reproducibilidad con misma semilla
    @test split1.idx_train == split2.idx_train
    @test split1.idx_val == split2.idx_val
    @test split1.idx_test == split2.idx_test

    # Cambia con semilla distinta (altamente probable)
    @test split1.idx_train != split3.idx_train ||
          split1.idx_val   != split3.idx_val   ||
          split1.idx_test  != split3.idx_test

    # Etiquetas en 1..3
    @test all(1 .<= split1.y_train .<= 3)
    @test all(1 .<= split1.y_val   .<= 3)
    @test all(1 .<= split1.y_test  .<= 3)

    # Minmax/clamp: train en [0,1]
    @test minimum(split1.X_train) >= 0.0 - 1e-12
    @test maximum(split1.X_train) <= 1.0 + 1e-12

    # Si clamp01=true, val/test también deben quedar en [0,1]
    @test minimum(split1.X_val) >= 0.0 - 1e-12
    @test maximum(split1.X_val) <= 1.0 + 1e-12
    @test minimum(split1.X_test) >= 0.0 - 1e-12
    @test maximum(split1.X_test) <= 1.0 + 1e-12

    # Estratificación: cada clase debe aparecer en cada split (para Iris, con estas proporciones sí)
    for c in 1:3
        @test any(==(c), split1.y_train)
        @test any(==(c), split1.y_val)
        @test any(==(c), split1.y_test)
    end
end