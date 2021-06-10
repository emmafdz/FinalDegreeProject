# Global variables
const epsilon = 10.0^(-10);
const epsilon2 = 1e-8;
const dx=0.1
const dt = 0.01;

const total_rate = 40;
const maximum_stim=3.5
using Polynomials

convert(::Type{Float64}, x::ForwardDiff.Dual) = Float64(x.value)
function convert(::Array{Float64}, x::Array{ForwardDiff.Dual})
    y = zeros(size(x));
    for i in 1:prod(size(x))
        y[i] = convert(Float64, x[i])
    end
    return y
end

convert(Float64, [4.0])
