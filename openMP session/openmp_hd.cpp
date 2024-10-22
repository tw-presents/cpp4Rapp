
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
#include <omp.h>

using namespace arma;


// [[Rcpp::export]]
arma::cube bsvars_ir1 (
    arma::mat&    aux_B,              // (N, N)
    arma::mat&    aux_A,              // (N, K)
    const int     horizon,
    const int     p,
    const bool    standardise = false
) {
  
  const int       N = aux_B.n_rows;
  cube            aux_irfs(N, N, horizon + 1);  // + 0 horizons
  mat             A_bold_tmp(N * (p - 1), N * p, fill::eye);
  
  mat   irf_0         = inv(aux_B);
  if ( standardise ) {
    irf_0             = irf_0 * diagmat(pow(diagvec(irf_0), -1));
  }
  mat   A_bold        = join_cols(aux_A.cols(0, N * p - 1), A_bold_tmp);
  mat   A_bold_power  = A_bold;
  
  aux_irfs.slice(0)   = irf_0;
  
  for (int h=1; h<horizon + 1; h++) {
    aux_irfs.slice(h) = A_bold_power.submat(0, 0, N-1, N-1) * irf_0;
    A_bold_power      = A_bold_power * A_bold;
  } // END h loop
  
  return aux_irfs;
} // END bsvars_ir1


// [[Rcpp::export]]
arma::field<arma::cube> bsvars_ir_omp (
    arma::cube&   posterior_B,        // (N, N, S)
    arma::cube&   posterior_A,        // (N, K, S)
    const int     horizon,
    const int     p,
    const bool    standardise = false
) {
  
  const int       N = posterior_B.n_rows;
  const int       S = posterior_B.n_slices;
  
  cube            aux_irfs(N, N, horizon + 1);
  field<cube>     irfs(S);
  
  #pragma omp parallel for
  for (int s=0; s<S; s++) {
    irfs(s)             = bsvars_ir1( posterior_B.slice(s), posterior_A.slice(s), horizon, p , standardise);
  } // END s loop
  
  return irfs;
} // END bsvars_ir



// [[Rcpp::export]]
arma::field<arma::cube> bsvars_ir (
    arma::cube&   posterior_B,        // (N, N, S)
    arma::cube&   posterior_A,        // (N, K, S)
    const int     horizon,
    const int     p,
    const bool    standardise = false
) {
  
  const int       N = posterior_B.n_rows;
  const int       S = posterior_B.n_slices;
  
  cube            aux_irfs(N, N, horizon + 1);
  field<cube>     irfs(S);
  
  for (int s=0; s<S; s++) {
    aux_irfs            = bsvars_ir1( posterior_B.slice(s), posterior_A.slice(s), horizon, p , standardise);
    irfs(s)             = aux_irfs;
  } // END s loop
  
  return irfs;
} // END bsvars_ir


// [[Rcpp::export]]
arma::cube bsvars_structural_shocks (
    const arma::cube&     posterior_B,    // (N, N, S)
    const arma::cube&     posterior_A,    // (N, K, S)
    const arma::mat&      Y,              // NxT dependent variables
    const arma::mat&      X               // KxT dependent variables
) {
  
  const int       N = Y.n_rows;
  const int       T = Y.n_cols;
  const int       S = posterior_B.n_slices;
  
  cube            structural_shocks(N, T, S);
  
  for (int s=0; s<S; s++) {
    structural_shocks.slice(s)    = posterior_B.slice(s) * (Y - posterior_A.slice(s) * X);
  } // END s loop
  
  return structural_shocks;
} // END bsvars_structural_shocks



// [[Rcpp::export]]
void bsvars_example_omp_ir (
    arma::cube&   posterior_B,        // (N, N, S)
    arma::cube&   posterior_A,        // (N, K, S)
    arma::mat&    Y,              // NxT dependent variables
    arma::mat&    X,              // KxT dependent variables
    const int     p
) {
  const int    horizon = Y.n_cols;

  field<cube>  irfs = bsvars_ir_omp (
                        posterior_B, 
                        posterior_A, 
                        horizon, 
                        p
                      );
  cube         ss   = bsvars_structural_shocks (
                        posterior_B, 
                        posterior_A, 
                        Y, 
                        X
                      );
} // END bsvars_example




// [[Rcpp::export]]
void bsvars_example (
    arma::cube&   posterior_B,        // (N, N, S)
    arma::cube&   posterior_A,        // (N, K, S)
    arma::mat&    Y,              // NxT dependent variables
    arma::mat&    X,              // KxT dependent variables
    const int     p
) {
  const int    horizon = Y.n_cols;
  
  field<cube>  irfs = bsvars_ir (
    posterior_B, 
    posterior_A, 
    horizon, 
    p
  );
  cube         ss   = bsvars_structural_shocks (
    posterior_B, 
    posterior_A, 
    Y, 
    X
  );
} // END bsvars_example



// [[Rcpp::export]]
void bsvars_example_omp (
    arma::cube&   posterior_B,        // (N, N, S)
    arma::cube&   posterior_A,        // (N, K, S)
    const arma::mat&      Y,              // NxT dependent variables
    arma::mat&    X,              // KxT dependent variables
    const int     p
) {
  const int    horizon = Y.n_cols;
  
  #pragma omp parallel sections
  {
    #pragma omp section
    {
      field<cube>  irfs = bsvars_ir (
        posterior_B, 
        posterior_A, 
        horizon, 
        p, 
        false
      );
    }
    #pragma omp section
    {
      cube         ss   = bsvars_structural_shocks (
        posterior_B, 
        posterior_A, 
        Y, 
        X
      );
    }
  }
} // END bsvars_example

/*** R
library(bsvars)
data(us_fiscal_lsuw)
us_fiscal_lsuw |>
  specify_bsvar$new(p = 4) |>
  estimate(S = 100) |>
  estimate(S = 200) -> post

posterior_B = post$posterior$B
posterior_A = post$posterior$A
Y = post$last_draw$data_matrices$Y
X = post$last_draw$data_matrices$X

microbenchmark::microbenchmark(
  bsvars_example (posterior_B,posterior_A, Y, X, 4),
  bsvars_example_omp (posterior_B,posterior_A, Y, X, 4),
  bsvars_example_omp_ir (posterior_B,posterior_A, Y, X, 4),
  times = 50
)

# bsvars_ir_omp (posterior_B,posterior_A,8,4)

*/
