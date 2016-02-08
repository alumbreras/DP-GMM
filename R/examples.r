source('gibbs.r')

run.example <- function(dataset=c('iris', 'geyser')){
  dataset <- match.arg(dataset)

  if(dataset == 'iris'){
    data(iris)
    A <- t(iris[,1:4])
    z_init <- sample(10, dim(A)[2], replace=TRUE) # random initialization
    #z_init <- rep(1, dim(A)[2]) # uniform initialization
  }
  if(dataset == 'geyser'){
    data(geyser)  
    A <- t(geyser)
    z_init <- sample(2, dim(A)[2], replace=TRUE) # random initialization
  }

  # Run 
  res <- gibbs(A, z_init=z_init, iters=1000)
  
  # plot traces
  traces <- res$traces
  chains <- mcmc(traces)
  plot(chains)
}

set.seed(2)
run.example(dataset='geyser')