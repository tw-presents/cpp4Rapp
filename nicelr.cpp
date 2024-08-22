#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
using namespace Rcpp;

// [[Rcpp::export]]
List nicelr (vec y, mat x) {
  vec beta_hat = solve(x.t() * x, x.t() * y);
  double sigma2_hat = as_scalar( (y - x * beta_hat).t() * (y - x * beta_hat) / y.n_elem );
  mat cov_beta_hat = sigma2_hat * inv_sympd(x.t() * x);
  return List::create(
    _["beta_hat"] = beta_hat, 
    _["sigma2_hat"] = sigma2_hat, 
    _["cov_beta_hat"] = cov_beta_hat
  );
}

/*** R
x = cbind(rep(1,5),1:5); y = x %*% c(1,2) + rnorm(5)
nicelr(y, x)
*/
