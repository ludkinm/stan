#ifndef STAN_MCMC_HOP_ADAPT_DIAG_E_STATIC_HOP_HPP
#define STAN_MCMC_HOP_ADAPT_DIAG_E_STATIC_HOP_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/hop/diag_e_hop.hpp>
#include <stan/mcmc/hop/hop_adapter.hpp>

namespace stan {
namespace mcmc {
/**
 * Hamiltonian Monte Carlo implementation using the endpoint
 * of trajectories with a static integration time with a
 * Gaussian-Euclidean disintegration and diag metric and
 * adaptive step size
 */
template <class Model, class BaseRNG>
class adapt_diag_e_hop : public diag_e_hop<Model, BaseRNG>,
                         public hop_var_adapter {
 public:
  adapt_diag_e_hop(const Model& model, BaseRNG& rng)
    : diag_e_hop<Model, BaseRNG>(model, rng),
      hop_var_adapter(model.num_params_r()) {
    this->hop_adaptation_.set_mu(3); // the target shrinkage for lambda
  }

  ~adapt_diag_e_hop() {}

  sample transition(sample& init_sample, callbacks::logger& logger) {
    // do a hop 
    sample s
        = diag_e_hop<Model, BaseRNG>::transition(init_sample, logger);

    // if still adapting adapt lambda and preconditioner
    if (this->adapt_flag_) {
      this->hop_adaptation_.learn_log_param(this->lambda, s.accept_stat());
      this->set_lambda_kappa(this->lambda);
      bool update = this->var_adaptation_.learn_variance(this->z().inv_e_metric_,
                                                         this->z().q);
      if (update) {
        this->hop_adaptation_.set_mu(3); // the target shrinkage for lambda
        this->hop_adaptation_.restart();
      }
    }
    return s;
  }
  
  void disengage_adaptation() {
    base_adapter::disengage_adaptation();
    this->hop_adaptation_.complete_adaptation(this->lambda);
    this->set_lambda_kappa(this->lambda);
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
