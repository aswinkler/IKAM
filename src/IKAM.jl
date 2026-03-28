module IKAM

# AD
include("ad/Dual.jl")

# Inner functions (piecewise Newton nodal, degree 1)
include("inner/PWLinearNodal.jl")

# Metrics
include("metrics/SoftmaxCrossEntropy.jl")

# Data
include("data/iris.jl")

# Layers
include("layers/KALayer.jl")
include("layers/KAInterLayer.jl")

# Utils
include("utils/MathUtils.jl")

# Model
include("model/OneLayerModel.jl")
include("model/InterLayerScaler.jl")
include("model/TwoLayerModel.jl")
include("model/OneLayerInterModel.jl")
include("model/TwoLayerInterModel.jl")

# Train and Utils
include("train/SGD.jl")
include("utils/RunLogger.jl") 
include("train/TrainerOneLayerAffineOnly.jl")
include("train/TrainerOneLayerFull.jl")
include("train/TrainerTwoLayerAffineOnly.jl")
include("train/TrainerTwoLayerFull.jl")
include("train/TrainerInterLayerAffineOnly.jl")
include("train/TrainerInterLayerCOnly.jl")
include("train/TrainerInterLayer_AB_plus_C_FixedPhi.jl")
include("train/TrainerInterLayer_PhiOnly_FixedAB.jl")
include("train/TrainerInterLayer_PhiPlusC_FixedAB.jl")
include("train/TrainerInterLayerFullABPhiC.jl")
include("train/TrainTwoInterLayerFullABPhiC.jl")
end # module