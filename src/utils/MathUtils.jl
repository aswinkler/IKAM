module MathUtils

using ..AD:value

export clamp01

# clamp01(x) = clamp(x, 0.0, 1.0)
# Clamp a 0-1
@inline function clamp01(u)
    uv = value(u)
    if uv < 0
        return 0(u)
    elseif uv > 1
        return one(u)
    else
        return u
    end
end

end