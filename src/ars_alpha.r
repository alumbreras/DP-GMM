#############
# ARS for the sampling of alpha in GMM and IGMM
############
library(ars)


# function
f_y <- function(y, K=3, U=10){
  y*(K-3/2) - 1/(2*exp(y)) + lgamma(exp(y)) - lgamma(exp(y)+U)
}

# first derivative function
f_y_prima <- function(y, K=3, U=10){
  (K-3/2) + 1/(2*exp(y)) + exp(y)*digamma(exp(y)) - exp(y)*digamma(exp(y)+U)
}

# Wrapper for ARS function
sample_alpha <- function(K=3, U=10){
  y <- ars2(1, f_y, f_y_prima, x=c(-2,0,30), K=K, U=U)
  exp(y)
}

###############################################################################
# Hack ARS function so that it returns the ifault codes and not just print them
# so that we can handle it like an exception
###############################################################################

ars2 <- function (n = 1, f, fprima, x = c(-4, 1, 4), ns = 100, m = 3, 
                  emax = 64, lb = FALSE, ub = FALSE, xlb = 0, xub = 0, ...) 
{
  mysample <- rep(0, n)
  iwv <- rep(0, ns + 7)
  rwv <- rep(0, 6 * (ns + 1) + 9)
  hx <- f(x, ...)
  hpx <- fprima(x, ...)
  initial <- .C("initial_", as.integer(ns), as.integer(m), 
                as.double(emax), as.double(x), as.double(hx), as.double(hpx), 
                as.integer(lb), as.double(xlb), as.integer(ub), as.double(xub), 
                ifault = as.integer(0), iwv = as.integer(iwv), rwv = as.double(rwv))
  if (initial$ifault == 0) {
    h <- function(x) f(x, ...)
    hprima <- function(x) fprima(x, ...)
    for (i in 1:n) {
      sample <- .C("sample_", as.integer(initial$iwv), 
                   as.double(initial$rwv), h, hprima, new.env(), 
                   beta = as.double(0), ifault = as.integer(0))
      if (sample$ifault == 0) {
        if (i < ns) {
          x <- c(x, sample$beta)
          h <- function(x) f(x, ...)
          hprima <- function(x) fprima(x, ...)
        }
        mysample[i] <- sample$beta
      }
      else {
        cat("\nError in sobroutine sample_...")
        cat("\nifault=", sample$ifault, "\n")
        stop(paste("\nError in sobroutine sample_...", sample$ifault))
      }
    }
  }
  else {
    cat("\nError in sobroutine initial_...")
    cat("\nifault=", initial$ifault, "\n")
    stop(paste("\nError in sobroutine initial_...", initial$ifault))
  }
  return(mysample)
}

environment(ars2) <- environment(ars)