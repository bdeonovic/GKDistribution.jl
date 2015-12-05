#################### G and K Distribution ####################

immutable GK <: ContinuousUnivariateDistribution
  A::Float64
  B::Float64
  g::Float64
  k::Float64
  c::Float64

  function GK(A::Real, B::Real, g::Real, k::Real, c::Real)
    ## check args
    0.0 <= c < 1.0 || throw(Argumenterror("c must be 0 <= c < 1"))
    

    ## distribution
    new(A, B, g, k, c)
  end
  GK(A::Real, B::Real, g::Real, k::Real) = GK(A, B, g, k, 0.8)
end

## Parameters

location(d::GK) = d.A
scale(d::GK) = d.B
skewness(d::GK) = d.g
kurtosis(d::GK) = d.k
asymmetry(d::GK) = d.c

params(d::GK) = (d.A, d.B, d.g, d.k, d.c)

minimum(d::GK) = -Inf
maximum(d::GK) = Inf

insupport{T <: Real}(d::GK, x::AbstractVector{T}) = true

function quantile(d::GK, p::Float64)
  z = quantile(Normal(), p)
  z2gk(d, z)
end


function Qinv(d::GK, x::Real)
  z = gk2z(d, x)
  cdf(Normal(), z)
end

function validate(d::GK)
  f = z -> -Fgkz(d, z)
  optres = optimize(f, 0, 20 / abs(skewness(d)))
  max(optres.f_minimum, 1.0) <= 1 / asymmetry(d)  && 
    scale(d) > 0 && kurtosis(d) > -0.5
end



function logpdf(d::GK, x::Real)
  if !validate(d)
    return -Inf
  end

  z = gk2z(d, x)

  term0 = exp(-skewness(d) * z)
  term1 = log(scale(d)) + log(asymmetry(d)) + 0.5 * log(2 * pi) + 
    z^2 / 2 + kurtosis(d) * log(1 + z^2)
  term2 = (1 / asymmetry(d) + (1 - term0) / (1 + term0))
  term3 = (1 + (2 * kurtosis(d) + 1) * z^2) / (1 + z^2)
  term4 = 2 * skewness(d) * z * term0 / (1 + term0)^2

  logQp = term1 + log(term2 * term3 + term4)
  
  return -logQp
end

function logpdf{T <: Real}(d::GK, x::AbstractVector{T})
  if !validate(d)
    return -Inf
  end
  
  value = 0.0
  for i in 1:length(x)
    z = gk2z(d, x[i])

    term0 = exp(-skewness(d) * z)
    term1 = log(scale(d)) + log(asymmetry(d)) + 0.5 * log(2 * pi) + 
      z^2 / 2 + kurtosis(d) * log(1 + z^2)
    term2 = (1 / asymmetry(d) + (1 - term0) / (1 + term0))
    term3 = (1 + (2 * kurtosis(d) + 1) * z^2) / (1 + z^2)
    term4 = 2 * skewness(d) * z * term0 / (1 + term0)^2

    value -= term1 + log(term2 * term3 + term4)
  end
  return value
end

function fit_mle{T <: Real}(d::GK, x::AbstractVector{T})
  loglikelihood = theta ->
    - logpdf(GK(theta[1], theta[2], theta[3], theta[4]), x)

  A0 = median(x)
  B0 = diff(quantile(x, [0.25,0.75]))[1]
  g0 = mean((x - A0) .^ 3) / (var(x) ^ (3/2))
  k0 = mean((x - A0) .^ 4) / (sumabs2(x - A0) / length(x))^2 - 3

  optres = optimize(loglikelihood, [A0, B0, g0, k0])

  return GK(optres.minimum...)
end


function gk2z(d::GK, x::Real; max_expands::Integer = 100, expand::Integer = 10)
  y = (x - location(d)) / scale(d)
  if y == 0 
    return 0
  elseif isfinite(y)
    d0 = GK(0, 1, skewness(d), kurtosis(d), asymmetry(d))
    inter = [-abs(y) / (1 - asymmetry(d)), abs(y)]
    iter = 0
    a, b = z2gk(d0, inter[1]) - y, z2gk(d0, inter[2]) - y
    while a * b > 0
      idx = findmin(abs([a,b]))[2]
      inter[idx] -= sign(a)*expand
      iter += 1
      iter < max_expands || break
    end
    return fzero(z -> z2gk(d0, z) - y, inter)
  else
    return y
  end
end

function Fgkz(d::GK, z::Float64)
  term1 = exp(-skewness(d) * z)
  term2 = 2 * skewness(d) * z * term1
  term3 = (1 + (2 * kurtosis(d) + 1) * z ^ 2) / (1 + z ^ 2)
  
  (1 - term1) / (1 + term1) + term2 / ((1 + term1) * term3)
end

function z2gk(d::GK, z::Float64)
  term1 = exp(-skewness(d) * z)
  term2 = (1.0 + asymmetry(d) * (1.0 - term1) / (1.0 + term1))
  term3 = (1.0 + z ^ 2) ^ kurtosis(d)
  location(d) + scale(d) * z * term2 * term3 
end


