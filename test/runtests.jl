using Test
using IKAM

# include("test_dual_basic.jl")
# include("test_inner_pwlinear.jl")
# include("test_metrics_softmax_ce.jl")
# include("test_data_iris.jl")
include("test_layers_kalayer.jl")
include("test_model_one_two_layer.jl")
include("test_train_sgd.jl")
include("test_utils_runlogger.jl")
include("test_train_trainer_affine_only.jl")
include("test_train_trainer_full.jl")
# include("test_train_two_layer.jl")
# include("test_model_two_layer_clamp01.jl")
include("test_layers_kainterlayer.jl")
