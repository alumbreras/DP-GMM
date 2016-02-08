# A gibbs sampler for the DP-GMM model presented in 
# https://www.seas.harvard.edu/courses/cs281/papers/rasmussen-1999a.pdf
# author: Alberto Lumbreras
#
# There is a matlab implementation: 
# https://github.com/mim/igmm/blob/master/igmm_mv.m

library(coda)
library(mixtools)
source('conditionals.r')


gibbs <- function(A, z_init, iters=100, plots=TRUE, plots.freq=10){
  # Arguments:
  #   A: matrix with user features (one column per user)
  #   z_init: initial assignments (one per user)
  ######################################################   
  par(mfrow=c(1,1))
  
  D <- dim(A)[1]
  N <- dim(A)[2]
  
  # traces
  traces.z <- matrix(NA, iters, N)
  traces <- matrix(NA, iters, 6)
  colnames(traces) <- c("n_clusters", "alpha", "mean_mu_a0", "det(R_a0)", "det(W_a0)", "beta_a0")
  
  # data mean and covariance
  Sigma_a <- cov(t(A))
  Lambda_a <- solve(Sigma_a)
  mu_a <- rowMeans(A)
  
  # init state
  alpha = 2
  z <- z_init 
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
    
    # Sample hyperpriors
    mu_a0 <- sample_mu_a0(Lambda_a, mu_a, mu_ar[,1:K, drop=FALSE], R_a0)
    R_a0 <- sample_R_a0(Sigma_a, mu_ar[,1:K, drop=FALSE], mu_a0) 
    W_a0 <- sample_W_a0(Lambda_a, S_ar[,,1:K, drop=FALSE], beta_a0)
    beta_a0 <- sample_beta_a0(S_ar[,,1:K, drop=FALSE], W_a0, init=beta_a0)
    
    # Sample assignments
    for (n in 1:N){
      res <- sample_z(n, A, alpha, z, mu_ar, S_ar, mu_a0, R_a0,  beta_a0, W_a0)
      z <- res$z
      mu_ar <- res$mu_ar
      S_ar <- res$S_ar
    }
    cat('\n', table(z))

    # Sample concentration parameter
    K <- length(unique(z))
    alpha <- sample_alpha(K = K, U = N)
    
    traces[i,] <- c(length(unique(z)), alpha, mean(mu_a0), det(R_a0), det(W_a0), beta_a0)
    traces.z[i,] <- z
    
    if(plots && (i%%plots.freq==0)){
      plot(t(A[c(1,2),]), col=palette()[z+1])
      title(paste0("Sample #", i))
      for(k in 1:K){
        ellipse(mu=mu_a0[c(1,2)], sigma=W_a0[c(1,2),c(1,2)], alpha = .25, lwd=5, npoints = 250, col="black")
        ellipse(mu=mu_ar[c(1,2),k], sigma=solve(S_ar[c(1,2),c(1,2),k]),  alpha = .25, lwd=3, npoints = 250, col=palette()[k+1])
      }
    }
    
  }
  res <- list(traces=traces,
              traces.z=traces.z)
  
  return(res)
}