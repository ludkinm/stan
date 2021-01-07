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
  template <class Model, template <class, class> class Hamiltonian, class BaseRNG>
  class base_hop : public base_mcmc {
  public:

    using PointType = typename Hamiltonian<Model, BaseRNG>::PointType;

    // Hacks so stan/services/util/run_adaptive_sampler.hpp
    PointType& z() { return point; }

    // Hacks so stan/services/util/run_adaptive_sampler.hpp
    void init_stepsize(callbacks::logger& logger){
      set_lambda_kappa();
    }

    
    void write_sampler_params(callbacks::writer& writer) {
      std::stringstream ss;
      ss << "(lam,kap,mu,gam) = " << get_lambda() << ',' << get_kappa() << ',' << get_mu() << ',' << get_gamma();
      writer(ss.str());
    }

    /**
     * write elements of mass matrix
     */
    void write_sampler_metric(callbacks::writer& writer) {
      z().write_metric(writer);
    }

    void write_sampler_state(callbacks::writer& writer) override {
      write_sampler_params(writer);
      write_sampler_metric(writer);
    }
  
    base_hop(const Model& model, BaseRNG& rng)
      : base_mcmc(),
        point(model.num_params_r()),
        hamiltonian(model),
        rand_int(rng),
        rand_uniform(rand_int) {
      set_lambda_kappa();
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

      // Set (q,p) to the sample and a draw from the "velocity" distribution at q
      point.q = init_sample.cont_params();
      hamiltonian.sample_p(point, rand_int);
      // draw p ~ N(0, Energy_Metric)
      // updates the potential and gradient in z
      hamiltonian.init(point, logger);

      // store the current point incase we reject
      ps_point curr_point = point;

      // calculate proposal
      Eigen::VectorXd Sigma_gc = hamiltonian.metric_times_grad(point, logger);
      double tau2c = point.g.dot(Sigma_gc);
      double rho2c = tau2c < 1.0 ? 1.0 : tau2c;
      double rhoc = sqrt(rho2c);

      double g_dot_rc = point.g.dot(point.p) / tau2c;

      double H0 = hamiltonian.H(point) - point.q.size() * log(rhoc);

      // new parameter values
      point.q += mu / rhoc * (point.p + (gamma - 1.0) * g_dot_rc * Sigma_gc);
      // update logpi and grad_logpi
      hamiltonian.update_potential_gradient(point, logger);

      // proposed point values
      Eigen::VectorXd Sigma_gp = hamiltonian.metric_times_grad(point, logger);
      double tau2p = point.g.dot(Sigma_gp);
      double rhop = tau2p < 1.0 ? 1.0 : sqrt(tau2p);

      double g_dot_rp = point.g.dot(point.p) / tau2p;

      // what would p need to be to have proposed curr_point?
      point.p= -rhop / rhoc * (point.p + (gamma-1) * g_dot_rc * Sigma_gc + 
           (1.0/gamma - 1.0) * (g_dot_rc + (gamma - 1.0) * point.g.dot(Sigma_gc) / tau2p * g_dot_rp) * Sigma_gp);

      // proposed Hamiltonian values
      double H1 = hamiltonian.H(point) - point.q.size() * log(rhop);
      alpha = std::exp(H0 - H1);
      // Should we reject and reset point to curr_point?
      if (alpha < 1 && rand_uniform() > alpha) {
        tau2 = tau2c;
        point.ps_point::operator=(curr_point);
      } else {
        tau2 = tau2p;
      }
      alpha = alpha > 1.0 ? 1.0 : alpha;

      // return logpi  = -potential = -V(z)
      return sample(point.q, -hamiltonian.V(point), alpha);
    }

    void get_sampler_param_names(std::vector<std::string>& names) {
      names.push_back("lambda");
      names.push_back("kappa");
      names.push_back("mu");
      names.push_back("gamma");
    }

    void get_sampler_params(std::vector<double>& values) {
      values.push_back(lambda);
      values.push_back(kappa);
      values.push_back(mu);
      values.push_back(gamma);
    }

    void get_sampler_diagnostic_names(std::vector<std::string>& model_names,
                                      std::vector<std::string>& names) {
      point.get_param_names(model_names, names);
      names.push_back("tau2");
      names.push_back("alpha");
    }

    void get_sampler_diagnostics(std::vector<double>& values) {
      point.get_params(values);
      values.push_back(tau2);
      values.push_back(alpha);
    }

    void set_lambda_kappa(double l=10.0, double k=1.0) {
      lambda = l;
      kappa = k;
      mu = sqrt(lambda * kappa);
      gamma = lambda / mu;
    }
    
    double get_mu() const { return mu; }

    double get_gamma() const { return gamma; }

    double get_lambda() const { return lambda; }

    double get_kappa() const { return kappa; }
  
  protected:
    PointType point;
    Hamiltonian<Model, BaseRNG> hamiltonian;
    BaseRNG& rand_int;
    boost::uniform_01<BaseRNG&> rand_uniform;
    double lambda;
    double kappa;
    double mu;
    double gamma;
    double tau2;
    double alpha;
  };

}  // namespace mcmc
}  // namespace stan
#endif
