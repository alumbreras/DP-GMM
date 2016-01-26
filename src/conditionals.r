# Sampling functions
#
library(MASS)
#source('ars_alpha.r')
#source('ars_beta.r')

sample_z <-function(){
  
}

sample_alpha <- function(){
    #
    # Sample alpha
    # by Adaptive Rejection Sampling
    # https://www1.maths.leeds.ac.uk/~wally.gilks/adaptive.rejection/web_page/Welcome.html
    # http://cran.r-project.org/web/packages/ars/ars.pdf
    #
    return(ars_sample_alpha(K = self.R, U = self.U))
}

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

sample_S_ar <- function(A_r, beta_a0, W_a0){
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
    
    scatter_matrix <- cov(t(A_r-rowMeans(A_r)))*(dim(A_r)[2]-1)
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
    return(rWishart(1, wishart_dof_post, wishart_S_post)[,,1])
}

# Cluster covariances
#####################
sample_W_a0 <- function(Lambda_a, S_ar, beta_a0){
    #
    # Sample base covariance of attributes (hyperparameter)
    # 
    K <- dim(S_ar)[3]
    D <- dim(S_ar)[1]
    scatter_matrix <- apply(S_ar, 3, sum)
    wishart_dof_post = D + K*beta_a0
    wishart_S_post = solve(D * Lambda_a + beta_a0*scatter_matrix)
    return(rWishart(1, wishart_dof_post, wishart_S_post)[,,1])
}

sample_beta_a0 <- function(z, S_ar, W_a0, init=beta_init){
    # Sample degrees of freedom (hyperparameter)
    # by Adaptive Rejection Sampling
    # https://www1.maths.leeds.ac.uk/~wally.gilks/adaptive.rejection/web_page/Welcome.html
    # http://cran.r-project.org/web/packages/ars/ars.pdf
    # https://code.google.com/p/pmtk3/source/browse/trunk/toolbox/Algorithms/mcmc/ars.m?r=2678
    
    K <- length(unique(z))
    S_ar = self.S_ar[,,1:K]         
    return(ars_sample_beta_a0(S=S_ar, W=W_a0, init=beta_init))
}