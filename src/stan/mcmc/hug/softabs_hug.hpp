#ifndef STAN_MCMC_HUG_SOFTABS_HUG_HPP
#define STAN_MCMC_HUG_SOFTABS_HUG_HPP

#include <stan/mcmc/hmc/hamiltonians/softabs_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/softabs_metric.hpp>
#include <stan/mcmc/hug/hug_bouncer.hpp>
#include <stan/mcmc/hmc/static/base_static_hmc.hpp>

namespace stan {
namespace mcmc {
/**
 * Hamiltonian Monte Carlo implementation using the endpoint
 * of trajectories with a static integration time with a
 * Gaussian-Riemannian disintegration and SoftAbs metric
 */
template <class Model, class BaseRNG>
class softabs_hug
    : public base_static_hmc<Model, softabs_metric, hug_bouncer, BaseRNG> {
 public:
  softabs_hug(const Model& model, BaseRNG& rng)
      : base_static_hmc<Model, softabs_metric, hug_bouncer, BaseRNG>(model,
                                                                     rng) {}
};

}  // namespace mcmc
}  // namespace stan
#endif
