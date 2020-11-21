#ifndef STAN_MCMC_HMC_AAPS_DENSE_E_AAPS_HMC_HPP
#define STAN_MCMC_HMC_AAPS_DENSE_E_AAPS_HMC_HPP

#include <stan/mcmc/hmc/aaps/base_aaps.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/dense_e_metric.hpp>

namespace stan {
namespace mcmc {
/**
 * Hamiltonian Monte Carlo implementation using the endpoint
 * of trajectories with a static integration time with a
 * Gaussian-Euclidean disintegration and dense metric
 */
template <class Model, class BaseRNG>
class dense_e_aaps
    : public base_aaps<Model, dense_e_metric, aaps_position_leapfrog, BaseRNG> {
 public:
  dense_e_aaps(const Model& model, BaseRNG& rng)
      : base_aaps<Model, dense_e_metric, aaps_position_leapfrog, BaseRNG>(
          model, rng) {}
};

}  // namespace mcmc
}  // namespace stan
#endif
