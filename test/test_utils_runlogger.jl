using Test
using IKAM.Utils

@testset "RunLogger creates run dir and writes history" begin
    mktempdir() do tmp
        lg = RunLogger(runs_root=joinpath(tmp, "runs"))
        dir = new_run!(lg; tag="unit_test", meta=Dict("seed"=>"123"))

        @test isdir(dir)
        @test isfile(joinpath(dir, "summary.txt"))

        tic!(lg)
        sleep(0.05)
        t = toc!(lg)
        @test t > 0

        log_epoch!(lg;
            epoch=1,
            loss_train=1.23, acc_train=0.4,
            loss_val=1.10, acc_val=0.5,
            loss_test=1.05, acc_test=0.55,
            time_epoch_s=t
        )

        @test isfile(joinpath(dir, "history.csv"))

        # Debe tener header + 1 fila
        lines = readlines(joinpath(dir, "history.csv"))
        @test length(lines) == 2
        @test startswith(lines[1], "epoch,loss_train")
        @test startswith(lines[2], "1,")
    end
end