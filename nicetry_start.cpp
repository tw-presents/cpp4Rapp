#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;

// [[Rcpp::export]]
vec nicetry (int n) {
  vec i(n, fill::randn);
  return i;
}

/*** R
nicetry(4)
*/
