# Main file
# Control the calls to the conditional samplers and store the traces
# a matlab implementation: 
# https://github.com/mim/igmm/blob/master/igmm_mv.m
library(abind)
library(coda)
library(mixtools)
source('conditionals.r')
set.seed(3)
PLOTS <- TRUE
data(iris)
data(geyser)

#res <- sample(t(iris[,1:4]), 1000)

#sample <- function(A, iters){
#   
  iters <- 1000
  A <- t(iris[,1:4])
  #A <- t(geyser)
  
  D <- dim(A)[1]
  N <- dim(A)[2]
  
  # traces
  traces.z <- matrix(NA, iters, N)
  traces <- matrix(NA, iters, 4)
  colnames(traces) <- c("n_clusters", "mean_mu_a0", "det(R_a0)", "det(W_a0)")
  
  # data mean and covariance
  Sigma_a <- cov(t(A))
  Lambda_a <- solve(Sigma_a)
  mu_a <- rowMeans(A)
  
  # init state
  alpha = 2
  z <- rep(1, dim(A)[2])
  #z <- as.numeric(factor(iris$Species))
  
  mu_a0 <- mu_a
  R_a0 <- solve(Sigma_a)
  W_a0 <- Sigma_a
  beta_a0 <- dim(A)[1]
  S_ar <- array(NA, dim=c(D, D, 100))
  mu_ar <- matrix(NA, D, 100)
  for (k in 1:length(unique(z))){
    mu_ar[,k] <- rowMeans(A[,z==k])
  }
  
  for(i in 1:iters){
    cat('\n', i)
    # Active components
    K <- length(unique(z))
    
    
    # Sample component parameters
    for (k in 1:length(unique(z))){
      mask <- (z==k)
      S_ar[,,k] <- sample_S_ar(A[,mask, drop=FALSE], mu_ar[,k, drop=FALSE], beta_a0, W_a0)
      mu_ar[,k] <- sample_mu_ar(A[,mask, drop=FALSE], S_ar[,,k], mu_a0, R_a0)
    }
    if(PLOTS){
      par(mfrow=c(1,2))
      plot(t(A[c(1,2),]), col=palette()[z+1], main=paste("Updated components"))
      for(k in 1:K){
        ellipse(mu=mu_a0[c(1,2)], sigma=W_a0[c(1,2),c(1,2)], alpha = .5, lwd=5, npoints = 250, col="black")
        ellipse(mu=mu_ar[c(1,2),k], sigma=solve(S_ar[c(1,2),c(1,2),k]),  alpha = .5, lwd=3, npoints = 250, col=palette()[k+1])
      }
    }
    
    # Sample hyperpriors
    mu_a0 <- sample_mu_a0(Lambda_a, mu_a, mu_ar[,1:K, drop=FALSE], R_a0)
    R_a0 <- sample_R_a0(Sigma_a, mu_ar[,1:K, drop=FALSE], mu_a0) 
    W_a0 <- sample_W_a0(Lambda_a, S_ar[,,1:K, drop=FALSE], beta_a0)
    #beta_a0 <- ars_sample_beta_a0(S=S_ar, W=W_a0, init=beta_a0)
    
    # Sample assignments and concentration parameter
    for (n in 1:N){
      res <- sample_z(n, A, alpha, z, mu_ar, S_ar, mu_a0, R_a0,  beta_a0, W_a0)
      z <- res$z
      mu_ar <- res$mu_ar
      S_ar <- res$S_ar
    }
    K <- length(unique(z))
    cat('\n', table(z))
    #alpha <- ars_sample_alpha(K = self.R, U = self.U)
    traces[i,] <- c(length(unique(z)), mean(mu_a0), det(R_a0), det(W_a0))
    traces.z[i,] <- z

   }
#  return(list(traces=traces,
#              traces.z=traces.z))
#}

#traces <- res$traces
#traces.z <- res$traces.z
chains <- mcmc(traces)
plot(chains)