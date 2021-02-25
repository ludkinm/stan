#ifndef STAN_MCMC_HOP_ADAPT_UNIT_E_STATIC_HOP_HPP
#define STAN_MCMC_HOP_ADAPT_UNIT_E_STATIC_HOP_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/hop/unit_e_hop.hpp>
#include <stan/mcmc/hop/hop_adapter.hpp>

namespace stan {
namespace mcmc {
/**
 * Hamiltonian Monte Carlo implementation using the endpoint
 * of trajectories with a static integration time with a
 * Gaussian-Euclidean disintegration and unit metric and
 * adaptive step size
 */
template <class Model, class BaseRNG>
class adapt_unit_e_hop : public unit_e_hop<Model, BaseRNG>,
                         public hop_adapter {
 public:
  adapt_unit_e_hop(const Model& model, BaseRNG& rng)
    : unit_e_hop<Model, BaseRNG>(model, rng),
      hop_adapter() {
    this->hop_adaptation_.set_mu(3); // the target shrinkage for gamma
  }

  ~adapt_unit_e_hop() {}
  
  sample transition(sample& init_sample, callbacks::logger& logger) {
    // do a hop 
    sample s
        = unit_e_hop<Model, BaseRNG>::transition(init_sample, logger);

    // if still adapting adapt lambda and preconditioner
    if (this->adapt_flag_) {
      this->hop_adaptation_.learn_log_param(this->gamma, s.accept_stat());
      this->set_gamma(this->gamma);
    }
    return s;
  }
  
  void disengage_adaptation() {
    base_adapter::disengage_adaptation();
    this->hop_adaptation_.complete_adaptation(this->gamma);
    this->set_gamma(this->gamma);
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
