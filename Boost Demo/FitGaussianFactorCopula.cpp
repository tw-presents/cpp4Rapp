#include <RcppArmadillo.h>
#include <RcppNumerical.h> 
#include <boost/math/distributions/normal.hpp> 
#include <boost/math/quadrature/gauss_kronrod.hpp> 


using namespace Rcpp;
using namespace arma;
using namespace Numer;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppEigen)]]

// Calculate the gaussian factor copula derivative for a single observation
// [[Rcpp::export]]
Rcpp::List gaussian_derivative_single_obs(
    const arma::vec& u1, 
    const arma::vec& rho, 
    const arma::vec& nl_x2, 
    const arma::vec& wl
) {
  int d = u1.n_elem;
  if (rho.n_elem != d) {
    Rcpp::Rcout << "u1 length: " << d << ", rho length: " << rho.n_elem << std::endl;
    Rcpp::stop("The dimensions of u1 and rho must match.");
  }
  
  int nodes = wl.n_elem;
  if (nl_x2.n_elem != nodes) {
    Rcpp::Rcout << "wl length: " << nodes << ", nl_x2 length: " << nl_x2.n_elem << std::endl;
    Rcpp::stop("The dimensions of wl and nl_x2 must match.");
  }
  
  arma::vec rho_squared = arma::square(rho);
  arma::vec one_minus_rho_squared = 1.0 - rho_squared;
  
  boost::math::normal normal_dist(0.0, 1.0);
  arma::vec x1(d);
  for (int i = 0; i < d; i++) {
    x1[i] = quantile(normal_dist, u1[i]);
  }
  
  arma::mat x2 = arma::repmat(nl_x2.t(), d, 1);  
  arma::mat x1_mat = arma::repmat(x1, 1, nodes);
  arma::mat rho_mat = arma::repmat(rho, 1, nodes);
  arma::mat rho_squared_mat = arma::repmat(rho_squared, 1, nodes);
  arma::mat one_minus_rho_squared_mat = arma::repmat(one_minus_rho_squared, 1, nodes);
  
  arma::mat numerator = 2 * rho_mat % x1_mat % x2 - rho_squared_mat % (arma::square(x1_mat) + arma::square(x2));
  arma::mat denominator = 2 * one_minus_rho_squared_mat;
  arma::mat exponent = numerator / denominator;
  arma::mat out = arma::exp(exponent) / arma::sqrt(one_minus_rho_squared_mat);
  
  arma::rowvec prod_vec = arma::prod(out, 0);
  
  arma::mat deriv_mul_vec = (rho_mat % one_minus_rho_squared_mat + (1 + rho_squared_mat) % x1_mat % x2 - rho_mat % (arma::square(x1_mat) + arma::square(x2))) / arma::square(one_minus_rho_squared_mat);
  
  arma::rowvec weighted_prod_vec = prod_vec % wl.t();
  arma::mat num = deriv_mul_vec.each_row() % weighted_prod_vec;
  arma::vec num_sum = arma::sum(num, 1);
  
  double denom = arma::as_scalar(prod_vec * wl);
  
  if (denom <= 0) {
    Rcpp::stop("Denominator is non-positive, which is not valid.");
  }
  
  
  arma::vec result = num_sum / denom; 
  
  return Rcpp::List::create(
    Rcpp::Named("log_likelihood") = std::log(denom),  
    Rcpp::Named("derivatives") = result               
  );
}

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::export]]
List gaussian_derivative(const arma::mat& unif_df, const arma::vec& rho, const arma::vec& nl_x2, const arma::vec& wl) {
  
  int n_obs = unif_df.n_rows;
  int d = unif_df.n_cols; 
  
  arma::vec total_derivatives = arma::zeros(d); 
  double total_log_likelihood = 0.0;  
  
  for (int i = 0; i < n_obs; i++) {
    arma::vec u1 = unif_df.row(i).t();
    
    List result = gaussian_derivative_single_obs(u1, rho, nl_x2, wl);
    
    total_log_likelihood += as<double>(result["log_likelihood"]);
    
    arma::vec derivatives = as<arma::vec>(result["derivatives"]);
    total_derivatives += derivatives; 
  }
  
  return List::create(
    Named("log_likelihood") = total_log_likelihood,
    Named("derivatives") = total_derivatives
  );
}

// [[Rcpp::depends(RcppArmadillo, RcppNumerical)]]
// [[Rcpp::depends(BH)]]
// Define the negative log-likelihood class for use with RcppNumerical optimization
class NegLogGaussianLikelihood : public MFuncGrad {
private:
  const arma::mat& unif_df;
  const arma::vec& nl_x2;
  const arma::vec& wl;
  
public:
  NegLogGaussianLikelihood(const arma::mat& unif_df_, const arma::vec& nl_x2_, const arma::vec& wl_)
    : unif_df(unif_df_), nl_x2(nl_x2_), wl(wl_) {}
  
  double f_grad(Constvec& rho, Refvec grad) override {
    arma::vec rho_arma(rho.size());
    std::copy(rho.data(), rho.data() + rho.size(), rho_arma.begin());
    
    if ((arma::max(rho_arma) > 0.9999999) || (arma::min(rho_arma) < -0.9999999) || rho_arma[0] < 0) {
      return 9e5; 
    }
    
    List result = gaussian_derivative(unif_df, rho_arma, nl_x2, wl);
    double log_likelihood = -1 * as<double>(result["log_likelihood"]);
    arma::vec derivatives = -1 * as<arma::vec>(result["derivatives"]);
    
    for (int i = 0; i < rho.size(); i++) {
      grad[i] = derivatives[i];
    }
    
    return log_likelihood;
  }
};

std::pair<arma::vec, arma::vec> gauss_legendre(int n_nodes) {
  boost::math::quadrature::gauss_kronrod<double, 15> integrator;
  
  const auto& abs = integrator.abscissa();
  const auto& wts = integrator.weights();
  
  arma::vec nodes(n_nodes);
  arma::vec weights(n_nodes);
  
  n_nodes = std::min(n_nodes, static_cast<int>(abs.size()));
  
  for (int i = 0; i < n_nodes; ++i) {
    nodes[i] = abs[i];    
    weights[i] = wts[i];  
  }
  
  return std::make_pair(nodes, weights);
}

// [[Rcpp::export]]
List fit_gaussian_copula(const arma::mat& unif_df, arma::vec rho, int n_nodes) {
  std::pair<arma::vec, arma::vec> gl = gauss_legendre(n_nodes);
  arma::vec nl_x2 = gl.first;  
  arma::vec wl = gl.second; 
  
  NegLogGaussianLikelihood nll(unif_df, nl_x2, wl);
  
  Eigen::VectorXd rho_eigen = Eigen::Map<Eigen::VectorXd>(rho.memptr(), rho.n_elem);
  
  double f_min;
  int result = optim_lbfgs(nll, rho_eigen, f_min);
  
  arma::vec rho_optimized(rho_eigen.size());
  std::copy(rho_eigen.data(), rho_eigen.data() + rho_eigen.size(), rho_optimized.begin());
  
  return List::create(
    Named("rho") = rho_optimized,
    Named("log_likelihood") = f_min
  );
}


