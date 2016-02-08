#############
# ARS for the sampling of beta_a0 in GMM and IGMM
############
# TODO: What about this package?
# http://www.inside-r.org/packages/cran/Runuran/docs/ars.new
library(ars)

# function
beta.f_y <-function(y, S, W){
  
  F <- dim(W)[1]
  R <- dim(S)[3]
  
  sum_SW <- 0
  for (r in 1:R){
    Sr <- S[,,r]
    Sr <- as.matrix(Sr) # compatibility with one-dimensional case
    sum_SW <- sum_SW + log(det(Sr)*det(W)) - sum(diag(Sr%*%W))
  }
  
  sum_gamma <- 0
  for (d in 1:F){
    sum_gamma <- sum_gamma + lgamma((exp(y)+d-F)/2)
  }
  r <- y
  r <-  r - R*sum_gamma 
  r <-  r - F/(2*(exp(y)-F+1))
  r <-  r - (3/2)*log(exp(y)-F+1)
  r <-  r + ((R*exp(y)*F)/2)*(y - log(2)) 
  r <-  r + (exp(y)/2)*sum_SW 
  
  return(r)
}

# first derivative function
beta.f_y_prima <- function(y, S, W){
  
  F <- dim(W)[1]
  R <- dim(S)[3]
  
  sum_SW <- 0
  for (r in 1:R){
    Sr <- S[,,r]
    Sr <- as.matrix(Sr) # compatibility with one-dimensional case
    sum_SW <- sum_SW + log(det(Sr)*det(W)) - sum(diag(Sr%*%W))
  }
  
  sum_digamma <- 0
  for (d in 1:F){
    sum_digamma <- sum_digamma + digamma((exp(y)+d-F)/2)
  }
  r <- 1
  r <-  r - R*exp(y)*0.5*sum_digamma 
  r <-  r + exp(y)*F/(2*((exp(y)-F+1))^2) 
  r <-  r - (3/2)*exp(y)/(exp(y)-F+1) + (R*F*exp(y))/2 +  R*F*exp(y)*(y-log(2))/2 
  r <-  r + (exp(y)/2)*sum_SW 
  
  return(r)
}

# Wrapper for ARS function
ars.sample_beta_a0 <- function(S, W, init=4){
  
  # make it compatible with the one-dimensional case
  # in case we want to use this version for both 
  # the multi and the uni-dimensional
  init <- log(init)
  if (is.matrix(W)){
     lb <- TRUE
     xlb <- log(dim(W)[1]-0.9)
     xpoints <- c(max(xlb, init-10), init, init+10) 
   }
  # in case of computational issues (W with lots of decimals and too close to 0), truncate W
  y <- tryCatch(ars2.beta(1, beta.f_y, beta.f_y_prima, x=xpoints, lb=lb, xlb=xlb, S=S, W=W),
                error=function(e){
                  print(e);
                  ars2.beta(1, beta.f_y, beta.f_y_prima, x=xpoints, lb=lb, xlb=xlb, S=S, W=signif(W,1))})
  exp(y)
}

###############################################################################
# Hack ARS function so that it returns the ifault codes and not just print them
# so that we can handle it like an exception
###############################################################################

ars2.beta <- function (n = 1, f, fprima, x = c(-4, 1, 4), ns = 100, m = 3, 
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
    cat("\nx:", x)
    cat("\nfprima:", fprima(x,...))
    cat("\nf:", f(x,...))
    stop(paste("\nError in sobroutine initial_...", initial$ifault))
  }
  return(mysample)
}

environment(ars2.beta) <- environment(ars)

# test:
# multidimensional works fine

# W <- diag(2)*26
# S <- array(NA, dim=c(10,2,2))
#for (i in 1:10){
#  S[i,,] <- rWishart(1, 2, solve(2*W))
#}
# sample_beta_a0(S,W)
#
# W <- diag(2)*26
# S <- array(NA, dim=c(10,2,2))
#for (i in 1:10){
#  S[i,,] <- 2*diag(2)*1/26 # exactly the expectancy
#}
# sample_beta_a0(S,W)
#
#


############################################
# Debug with this

# Problematics
# s <- c(0.00589706, 0.00566612,  0.00505162,  0.00458974)
# w <- 202.159996343
# 
# s <-  c(0.00495562,  0.00501441,  0.00484268,  0.005048)
# w <-  193.110366286
# 
# W <- as.matrix(w)
# S <- array(NA, dim=c(length(s),1,1))
# S[,,] <- s
# x <- seq(-5,5, by=0.5)
# 
# sample_beta_a0(s,w)
# 
# plot(x, f_y(x, S, W))
# plot(x, f_y_prima(x, S, W))
# 
# f_y_prima(c(-10,-4,4,25,100,300), S, W)

######################################################"""
S <- array(NA, dim=c(1, 1, 1))