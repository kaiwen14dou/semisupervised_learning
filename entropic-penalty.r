#
# Toy code for optimization using entropic penalty
#

# Synthetic data
N_unlabeled <- 5000
N_labeled <- 1000
N <- N_unlabeled + N_labeled
D <- 1000
X_unlabeled <- matrix(rnorm(N_unlabeled * D), N_unlabeled)
X_labeled <- matrix(rnorm(N_labeled * D), N_labeled)
y <- rbinom(N_labeled, 1, .2)

# Penalized log-likelihood function (logistic likelihood for labeled data)
lik_pl <- function(theta, y, X_u, X_l, lambda) {
    mu_l <- X_l %*% theta
    mu_u <- X_u %*% theta
    pr_l <- 1 / (1 + exp(-mu_l))
    pr_u <- 1 / (1 + exp(-mu_u))
    out <- -mean(dbinom(y, 1, pr_l, log = TRUE)) + 
        # Entropic penalty using unlabeled data
        lambda * dim(X_u)[1] / length(y) * mean((pr_u-1) * mu_u - log1p(exp(-mu_u))) +
        # Small ridge penalty to avoid complete separation
        1e-6 * sum(theta^2)
    out
}
init <- rnorm(D) / sqrt(D) # Initial value
lambda <- .5 # Tuning parameter
ssl_estimate <- optim(
  init, lik_pl, 
  y = y, X_u = X_unlabeled, X_l = X_labeled, lambda = lambda, 
  method = "L-BFGS-B", 
  control = list(trace= 3)
)

