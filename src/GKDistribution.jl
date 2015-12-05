using Distributions

module GKDistribution

  import Distributions:
         ContinuousUnivariateDistribution,
         location,
         scale,
         skewness,
         kurtosis,
         params,
         minimum,
         maximum,
         insupport,
         quantile,
         logpdf,
         fit_mle,
         Normal,
         cdf,
         pdf

  import Optim: 
         optimize
  import Roots:
         fzero

  include("gk.jl")
  export GK, 
         logpdf, 
         quantile, 
         location, 
         scale, 
         skewness, 
         kurtosis, 
         asymmetry, 
         logpdf, 
         quantile, 
         params, 
         minimum, 
         maximum, 
         insupport,
         pdf

end # module
