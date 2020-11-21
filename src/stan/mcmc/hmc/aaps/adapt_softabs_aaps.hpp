#ifndef STAN_MCMC_HMC_AAPS_ADAPT_SOFTABS_AAPS_HPP
#define STAN_MCMC_HMC_AAPS_ADAPT_SOFTABS_AAPS_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/hmc/aaps/softabs_aaps.hpp>
#include <stan/mcmc/stepsize_adapter.hpp>

namespace stan {
namespace mcmc {
/**
 * Hamiltonian Monte Carlo implementation using the endpoint
 * of trajectories with a static integration time with a
 * Gaussian-Riemannian disintegration and SoftAbs metric and
 * adaptive step size
 */
template <class Model, class BaseRNG>
class adapt_softabs_aaps : public softabs_aaps<Model, BaseRNG>,
                           public stepsize_adapter {
 public:
  adapt_softabs_aaps(const Model& model, BaseRNG& rng)
      : softabs_aaps<Model, BaseRNG>(model, rng) {}

  ~adapt_softabs_aaps() {}

  sample transition(sample& init_sample, callbacks::logger& logger) {
    sample s = softabs_aaps<Model, BaseRNG>::transition(init_sample, logger);

    if (this->adapt_flag_) {
      this->stepsize_adaptation_.learn_stepsize(this->nom_epsilon_,
                                                s.accept_stat());
      this->update_L_();
    }

    return s;
  }

  void disengage_adaptation() {
    base_adapter::disengage_adaptation();
    this->stepsize_adaptation_.complete_adaptation(this->nom_epsilon_);
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
