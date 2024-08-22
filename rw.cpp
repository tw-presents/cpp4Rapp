#include <Rcpp.h>
using namespace Rcpp;


// [[Rcpp::export]]
NumericVector rw(int T) {
  NumericVector rw = rnorm(T);
  return cumsum(rw);
}

/*** R
set,seed(1)
plot.ts(rw(200))
*/
