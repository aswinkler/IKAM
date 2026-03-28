# src/utils/RunLogger.jl

module Utils

using Dates
using Printf

export RunLogger, new_run!, log_epoch!, write_summary!, now_timestamp,
       tic!, toc!, ensure_dir

"""
    now_timestamp()

Timestamp legible y ordenable: YYYYmmdd_HHMMSS
"""
now_timestamp() = Dates.format(Dates.now(), dateformat"yyyymmdd_HHMMSS")

"""
    ensure_dir(path)

Crea el directorio si no existe.
"""
function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

mutable struct RunLogger
    runs_root::String
    run_dir::String
    history_path::String
    summary_path::String
    header_written::Bool
    t0_ns::UInt64
end

"""
    RunLogger(; runs_root="runs")

No crea nada aún. Usa `new_run!` para inicializar una corrida.
"""
function RunLogger(; runs_root::AbstractString="runs")
    return RunLogger(String(runs_root), "", "", "", false, 0)
end

"""
    new_run!(logger; tag="iris_one_layer", meta=Dict())

Crea runs/<timestamp>_<tag>/ y prepara history.csv y summary.txt.
Devuelve la ruta del directorio de corrida.
"""
function new_run!(lg::RunLogger; tag::AbstractString="run", meta=Dict{String,Any}())
    ensure_dir(lg.runs_root)

    run_name = string(now_timestamp(), "_", tag)
    lg.run_dir = joinpath(lg.runs_root, run_name)
    ensure_dir(lg.run_dir)

    lg.history_path = joinpath(lg.run_dir, "history.csv")
    lg.summary_path = joinpath(lg.run_dir, "summary.txt")
    lg.header_written = false

    # Guardar meta mínima en summary.txt
    open(lg.summary_path, "w") do io
        println(io, "run_dir: ", lg.run_dir)
        println(io, "created_at: ", Dates.now())
        println(io, "tag: ", tag)
        if !isempty(meta)
            println(io, "\nmeta:")
            for (k,v) in meta
                println(io, "  ", k, ": ", v)
            end
        end
    end

    return lg.run_dir
end

"""
    log_epoch!(logger; epoch, loss_train, acc_train, loss_val, acc_val, loss_test, acc_test, time_epoch_s)

Agrega una fila a history.csv. Escribe header una sola vez.
"""
function log_epoch!(lg::RunLogger; epoch::Int,
                    loss_train, acc_train,
                    loss_val, acc_val,
                    loss_test, acc_test,
                    time_epoch_s)

    @assert lg.history_path != "" "Primero llama new_run!(logger, ...)."

    if !lg.header_written
        open(lg.history_path, "w") do io
            println(io, "epoch,loss_train,acc_train,loss_val,acc_val,loss_test,acc_test,time_epoch_s")
        end
        lg.header_written = true
    end

    open(lg.history_path, "a") do io
        @printf(io, "%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.6f\n",
                epoch,
                float(loss_train), float(acc_train),
                float(loss_val),   float(acc_val),
                float(loss_test),  float(acc_test),
                float(time_epoch_s))
    end

    return nothing
end

"""
    write_summary!(logger, text)

Agrega texto al summary.txt (al final).
"""
function write_summary!(lg::RunLogger, text::AbstractString)
    @assert lg.summary_path != "" "Primero llama new_run!(logger, ...)."
    open(lg.summary_path, "a") do io
        println(io, "\n", text)
    end
    return nothing
end

# --- Timing helpers ---
"""
    tic!(logger)

Marca inicio (ns).
"""
function tic!(lg::RunLogger)
    lg.t0_ns = time_ns()
    return nothing
end

"""
    toc!(logger) -> seconds

Tiempo transcurrido desde tic!
"""
function toc!(lg::RunLogger)
    @assert lg.t0_ns != 0
    dt = (time_ns() - lg.t0_ns) / 1e9
    return dt
end

end # module Utils