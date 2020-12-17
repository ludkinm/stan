#ifndef STAN_MCMC_HOP_BASE_HOP_HPP
#define STAN_MCMC_HOP_BASE_HOP_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/hmc/base_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace stan {
namespace mcmc {
/**
 * Hop
 */
template <class Model, template <class, class> class Hamiltonian,
class BaseRNG>
class base_hop
    : public base_mcmc {
 public:
  base_hop(const Model& model, BaseRNG& rng)
    : base_mcmc(),
        point(model.num_params_r()),
        hamiltonian(model),
        rand_int(rng),
        rand_uniform(rand_int)
  {
    set_lambda_kappa(10.0, 1.0);
  }

  ~base_hop() {}

  void set_metric(const Eigen::MatrixXd& inv_e_metric) {
    point.set_metric(inv_e_metric);
  }

  void set_metric(const Eigen::VectorXd& inv_e_metric) {
    point.set_metric(inv_e_metric);
  }

  sample transition(sample& init_sample, callbacks::logger& logger) {
    // Shoehorning into the HMC framework so I don't have to roll
    // my own preconditioners...

    // Set (q,p) to the sample and a draw from the distribution at q
    point.q = init_sample.cont_params();
    hamiltonian.sample_p(point, rand_int);
    // draw p ~ N(0, Mass)
    // updates the potential and gradient in z
    hamiltonian.init(point, logger);
    
    // store the current point incase we reject
    ps_point curr_point(point);

    // calculate proposal
    Eigen::VectorXd Sigma_gc = hamiltonian.metric_times_grad(point, logger);
    double tau2c = point.g.dot(Sigma_gc);
    double rho2c = tau2c < 1.0 ? 1.0 : tau2c;
    double rhoc = sqrt(rho2c);

    double g_dot_rc = point.g.dot(point.p)/tau2c;
        
    double H0 = hamiltonian.H(point) - point.q.size() * log(rhoc);

    // new parameter values
    point.q += mu/rho2c * (point.p + (gamma-1.0) * g_dot_rc * Sigma_gc);
    // update logpi and grad_logpi
    hamiltonian.update_potential_gradient(point, logger);

    // proposed point values
    Eigen::VectorXd Sigma_gp = hamiltonian.metric_times_grad(point, logger);
    double tau2p = point.g.dot(Sigma_gp);
    double rhop = tau2p < 1.0 ? 1.0 : sqrt(tau2p);

    double g_dot_rp = point.g.dot(point.p)/tau2p;

    // what would p need to be to have proposed curr_point?
    point.p = -rhop/rhoc * (point.p + (gamma-1.0) * g_dot_rc * Sigma_gc + (1.0/gamma - 1.0) * (g_dot_rp + (gamma-1) * g_dot_rc * point.g.dot(Sigma_gc) / tau2p) * Sigma_gp);
    
    // proposed Hamiltonian values
    double H1 = hamiltonian.H(point) - point.q.size() * log(rhop);
    double acceptProb = std::exp(H0 - H1);
    // Should we reject and reset point to curr_point?
    if (acceptProb < 1 && rand_uniform() > acceptProb)
      point.ps_point::operator=(curr_point);
    acceptProb = acceptProb > 1 ? 1 : acceptProb;

    // return logpi  = -potential = -V(z)
    return sample(point.q, -hamiltonian.V(point), acceptProb);
  }

  void get_sampler_param_names(std::vector<std::string>& names) {
    names.push_back("lambda__");
    names.push_back("kappa__");
    names.push_back("logpi__");
  }

  void get_sampler_params(std::vector<double>& values) {
    values.push_back(lambda);
    values.push_back(kappa);
    values.push_back(point.V);
  }

  void set_lambda_kappa(double l, double k) {
    lambda = l;
    kappa = k;
    mu = sqrt(lambda * kappa);
    gamma = lambda/mu;
  }

  double get_mu() const { return mu; }

  double get_gamma() const { return gamma; }

  double get_lambda() const { return lambda; }

  double get_kappa() const { return kappa; }
    
protected:
  typename Hamiltonian<Model, BaseRNG>::PointType point;
  Hamiltonian<Model, BaseRNG> hamiltonian;
  BaseRNG& rand_int;
  boost::uniform_01<BaseRNG&> rand_uniform;
  double lambda;
  double kappa;
  double mu;
  double gamma;
};

}  // namespace mcmc
}  // namespace stan
#endif
