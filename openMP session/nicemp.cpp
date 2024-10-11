
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <omp.h>

using namespace arma;

// sample from a multivariate normal distribution using a loop
// [[Rcpp::export]]
arma::mat rmvn_loop(
    const int n,             // a positive integer - number of draws
    const arma::vec mu,      // a vector of means
    const arma::mat sigma    // a covariance matrix
) {
  int p = mu.n_elem;
  mat out(p, n);
  mat L = chol(sigma, "lower");
  
  for (int i = 0; i < n; i++) {
    vec z = randn<arma::vec>(p);
    out.col(i) = mu + L * z;
  }
  
  return out;
}

// sample from a multivariate normal distribution using a parallel loop!
// [[Rcpp::export]]
arma::mat rmvn_par(
    const int n,             // a positive integer - number of draws
    const arma::vec mu,      // a vector of means
    const arma::mat sigma    // a covariance matrix
) {
  int p = mu.n_elem;
  mat out(p, n);
  mat L = chol(sigma, "lower");
  
  #pragma omp parallel for   // this is the only change
  for (int i = 0; i < n; i++) {
    vec z = randn<arma::vec>(p);
    out.col(i) = mu + L * z;
  }
  
  return out;
}

/*** R
set.seed(123)
N = 20
microbenchmark::microbenchmark(
  rmvn_loop(10000, rep(0, N), diag(N)),
  rmvn_par(10000, rep(0, N), diag(N)),
  times = 1000
)
*/
