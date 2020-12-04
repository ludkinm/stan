#ifndef STAN_MCMC_HUG_DENSE_E_HUG_HPP
#define STAN_MCMC_HUG_DENSE_E_HUG_HPP

#include <stan/mcmc/hmc/static/base_static_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_metric.hpp>
#include <stan/mcmc/hug/hug_bouncer.hpp>

namespace stan {
namespace mcmc {
/**
 * Hamiltonian Monte Carlo implementation using the endpoint
 * of trajectories with a static integration time with a
 * Gaussian-Euclidean disintegration and dense metric
 */
template <class Model, class BaseRNG>
class dense_e_hug
    : public base_static_hmc<Model, dense_e_metric, hug_bouncer, BaseRNG> {
 public:
  dense_e_hug(const Model& model, BaseRNG& rng)
      : base_static_hmc<Model, dense_e_metric, hug_bouncer, BaseRNG>(model,
                                                                     rng) {}
};

}  // namespace mcmc
}  // namespace stan
#endif
