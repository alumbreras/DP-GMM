A <- matrix(rnorm(30), nrow=3, ncol=10) 
m <- rowMeans(A)
res <- 0
for(i in 1:dim(A)[2]){
  res <- res + (A[,i] - m)%*%t(A[,i] - m)
}
cat("\nResult:")
print(res)

res2 <- cov(t(A-rowMeans(A)))*(dim(A)[2]-1)
cat("\n Alternative:")
print(res2)
