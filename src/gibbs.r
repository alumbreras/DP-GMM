# Main file
# Control the calls to the conditional samplers and store the traces
library(abind)
source('conditionals.r')
abind(Sigma_a, along = 3)
sample <- function(A, iters){
  
  A <- t(iris[,1:4])
  D <- dim(A)[1]
  
  # init traces
  Sigma_a <- cov(t(A))
  Lambda_a <- solve(Sigma_a)
  mu_a <- rowMeans(A)
  
  # init state
  alpha = 2
  z <- rep(1, dim(A)[2])
  mu_a0 <- mu_a
  R_a0 <- solve(Sigma_a)
  W_a0 <- Sigma_a
  beta_a0 <- dim(A)[1]
  
  
  #S_ar <- replicate(100, Lambda_a)
  #mu_ar <- replicate(100, mu_a0)
  S_ar <- array(NA, dim=c(D, D, 100))
  mu_ar <- matrix(NA, D, 100)
  for(i in 1:iters){
    cat('\n', i)
    # Active components
    K <- length(unique(z))
    
    # Sample component parameters
    for (k in 1:length(unique(z))){
      mask <- (z==k)
      S_ar[,,k] <- sample_S_ar(A[,mask], beta_a0, W_a0)
      mu_ar[,k] <- sample_mu_ar(A[,mask], S_ar[,,k], mu_a0, R_a0)
    }

    # Sample hyperpriors
    mu_a0 <- sample_mu_a0(Lambda_a, mu_a, mu_ar[,1:K, drop=FALSE], R_a0)
    R_a0 <- sample_R_a0(Sigma_a, mu_ar[,1:K, drop=FALSE], mu_a0) 
    W_a0 <- sample_W_a0(Lambda_a, S_ar[,,1:K, drop=FALSE], beta_a0)
    #beta_a0 <- ars_sample_beta_a0(S=S_ar, W=W_a0, init=beta_a0)
    
    # Sample assignments and concentration parameter
    #z <- sample_z()
    #alpha <- ars_sample_alpha(K = self.R, U = self.U)
    
    
  }
}

# Main #
data(iris)
sample(t(iris[,1:4]), 1000)