# Sampling functions
#
library(MASS)
library(mvtnorm)
source('ars_alpha.r')
source('ars_beta.r')

DEBUG <- FALSE
sample_z <- function(u, A, alpha, z, mu_ar, S_ar, mu_a0, R_a0, beta_a0, W_a0){
  #
  # Chinese Restaurant Process with auxiliary tables (Neal's algorithm 8)
  # 
  m <- 3 # number of auxiliary tables to approximate the infinite number of empty tables
  
  n <- tabulate(z)
  K <- length(unique(z))
  D <- dim(A)[1]
  R_a0_inv <- solve(R_a0)
  W_a0_inv <- solve(W_a0)
  
  # Compute likelihood for every table
  logprobs <- rep(NA, K+m)
  for(k in 1:K){
    logp <- log(n[k]-as.numeric(z[u]==k))
    logp <- logp + mvtnorm::dmvnorm(A[,u], mean=mu_ar[,k,drop=FALSE], sigma=solve(S_ar[,,k]), log=TRUE)
    logprobs[k] <- logp 
  }
    
    # Create m auxiliary tables from the base distribution
    aux.tables.mean <- t(mvrnorm(m, mu_a0, R_a0_inv))
    aux.tables.covariance <-  rWishart(m, max(beta_a0,D), W_a0/beta_a0)
    
    # If last point in cluster, re-use its parameters for the auxiliary table so that
    # it has some probability of staying 
    if (n[z[u]]==1){
      aux.tables.mean[,1] <- mu_ar[,z[u], drop=FALSE]
      aux.tables.covariance[,,1] <- S_ar[,,z[u]]
    }
    
    # Compute likelihood for auxilary table
    for(k in 1:m){
      logp <- log(alpha/m)
      logp <- logp + mvtnorm::dmvnorm(A[,u], mean=aux.tables.mean[,k], sigma=as.matrix(aux.tables.covariance[,,k]), log=TRUE)
      logprobs[K+k] <- logp 
    }
  
  
  # normalize probabilities to avoid numerical underflow
  probs <- logprobs-max(logprobs)
  probs <- exp(probs)
  probs <- probs/sum(probs)
  
  # Choose table
  chosen <- base::sample(1:(K+m), 1, prob=probs)
  
  
  # Re-label tables
  ##########################################
  # if auxiliary table is chosen, assign label K+i-th to this new table
  if(chosen>K){
    mu_ar[,K+1] <- aux.tables.mean[, chosen - K]
    S_ar[,,K+1] <- solve(aux.tables.covariance[,, chosen - K])
    z[u] <- K+1
  }
  else{
    z[u] <- chosen
  }
  
  # if old cluster is empty, shift right tables to the left
  # do it only if DP is used (m>0)
  n <- tabulate(z)
  if(any(n==0)){
    empty <- which(n==0)
    right <- (empty+1):(K+1)
    mu_ar[,(empty:K)] <- mu_ar[,(empty+1):(K+1)]
    S_ar[,,(empty:K)] <- S_ar[,,(empty+1):(K+1)]
    z[z>empty] <- z[z>empty] - 1
  }
  
  if( max(z) != length(unique(z)) ){
    stop("Some cluster is not being used")
  }
  
  return(list(z=z,
              mu_ar=mu_ar,
              S_ar=S_ar))
}


########################################################
# Component parameters
########################################################

sample_mu_ar <- function(A_r, S_ar, mu_a0, R_a0){
  # 
  # Samples a mean for cluster r 
  # from its conditional probability
  # 
  n <- dim(A_r)[2]
  
  # If nobody in the cluster sample from prior
  if (n == 0){
    return(mvrnorm(1, mu_a0, solve(R_a0)))
  }
  
  ar_mean <- rowMeans(A_r)
  Lambda_post <- R_a0 + n*S_ar
  Sigma_post <- solve(Lambda_post)
  mu_post <- Sigma_post %*% ((R_a0 %*% mu_a0) + n*(S_ar%*%ar_mean))  
  return(mvrnorm(1, mu_post, Sigma_post))
}

sample_S_ar <- function(A_r, mu_ar_k, beta_a0, W_a0){
  #
  # Sample attributes covariance matrix of cluster r
  # from its conditional probability
  #
  n <- dim(A_r)[2]
  D <- dim(A_r)[1]
  
  # If nobody in the cluster sample from prior
  if (n == 0){
    # unfortunately wishart implementation does not accept dof > F-1
    # but dof >= F
    # This will only affect when the cluster is empty.
    df <- max(beta_a0, D) 
    return(rWishart(1, df, solve(W_a0)/beta_a0))
  }
  # If only one point, scatter is just this product
  if(n==1){
    scatter_matrix <- (A_r-mu_ar_k)%*%t(A_r-mu_ar_k)
  }
  if(n > 1){
    scatter_matrix <- cov(t(A_r-c(mu_ar_k)))*(dim(A_r)[2]-1)
  }
  
  wishart_dof_post <- beta_a0 + n
  wishart_S_post <- solve(beta_a0*W_a0 + scatter_matrix)
  return(rWishart(1, wishart_dof_post, wishart_S_post)[,,1])
}       


########################################################
# Hyperparameters
########################################################

# Cluster means 
#################
sample_mu_a0 <- function(Lambda_a, mu_a, mu_ar, R_a0){
  #
  # Samples mean of gaussian hyperprior placed over clusters centroids
  #
  K <- dim(mu_ar)[2]
  mean_ar = rowMeans(mu_ar)
  Lambda_post = Lambda_a + K*R_a0
  Sigma_post = solve(Lambda_post)
  mu_post = Sigma_post %*% (Lambda_a %*% mu_a) + K*(R_a0 %*% mean_ar)     
  return(mvrnorm(1, mu_post, Sigma_post)) 
}

sample_R_a0 <-function(Sigma_a, mu_ar, mu_a0){
  #
  # Samples precision of gaussian hyperprior placed over clusters centroids
  #
  K <- dim(mu_ar)[2]
  D <- dim(mu_ar)[1]
  # Can't use the trick cov(t(mu_ar-mu_a0))*(K-1) when there is only one cluster
  scatter_matrix <- 0
  for(i in 1:K){
    scatter_matrix <- scatter_matrix + (mu_ar[,i]-mu_a0)%*%t(mu_ar[,i]-mu_a0)
  }
  wishart_dof_post = D + K
  wishart_S_post = solve(D*Sigma_a + scatter_matrix)
  return(as.matrix(rWishart(1, wishart_dof_post, wishart_S_post)[,,1]))
}

# Cluster covariances
#####################
sample_W_a0 <- function(Lambda_a, S_ar, beta_a0){
  #
  # Sample base covariance of attributes (hyperparameter)
  # 
  K <- dim(S_ar)[3]
  D <- dim(S_ar)[1]
  # TODO: why apply does not work properly?
  #scatter_matrix <- apply(S_ar, 3, sum)
  scatter_matrix <- 0
  for (i in 1:dim(S_ar)[3]){
    scatter_matrix <- scatter_matrix + S_ar[,,i]
  }
  wishart_dof_post = D + K*beta_a0
  wishart_S_post = solve(D * Lambda_a + beta_a0*scatter_matrix)
  return(as.matrix(rWishart(1, wishart_dof_post, wishart_S_post)[,,1]))
}


sample_beta_a0 <- function(S, W, init=4){
  # Sample the degrees of freedom of the Wishart
  # given a scale matrix W and some observed precision matrices S
  ars.sample_beta_a0(S, W, init=init)
}