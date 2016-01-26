library(ars)

# function
f_y <- function(y, s=1, w=1){
  R <-length(s)
  
  sum_sw <- 0
  for (j in 1:R){
    sum_sw <- sum_sw + log(s[j]*w) - s[j]*w
  }
  r <- y # derivative for the variable transformation
  r <- r - R*lgamma(exp(y)/2)
  r <- r - 1/(2*exp(y))
  r <- r + ((R*exp(y)-3)/2)*(y - log(2))
  r <- r + (exp(y)/2)*sum_sw
  return(r)
}

# first derivative function
f_y_prima <- function(y, s=1, w=1){
  R <-length(s)
  sum_sw <- 0
  for (j in 1:R){
    sum_sw <- sum_sw + log(s[j]*w) - s[j]*w
  }
  
  r <- 1
  r <- r - R*digamma(exp(y)/2)*exp(y)/2
  r <- r + 0.5*exp(-y)
  r <- r + R*exp(y)*(y - log(2))/2 + (R*exp(y) - 3)/2
  r <- r + (exp(y)/2)*sum_sw
  
  return(r)
}

# Wrapper for ARS function
sample_beta_b0 <- function(s=3, w=10){
  #y <- ars2(1, f_y, f_y_prima, x=c(-4,4,10), s=s, w=w)
  y <- ars2(1, f_y, f_y_prima, x=c(-4,4,25), s=s, w=w)
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
