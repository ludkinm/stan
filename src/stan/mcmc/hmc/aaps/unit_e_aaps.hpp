#ifndef STAN_MCMC_HMC_AAPS_UNIT_E_AAPS_HPP
#define STAN_MCMC_HMC_AAPS_UNIT_E_AAPS_HPP

#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_metric.hpp>
#include <stan/mcmc/hmc/aaps/base_aaps.hpp>

namespace stan {
namespace mcmc {
/**
 * Hamiltonian Monte Carlo implementation using the endpoint
 * of trajectories with a static integration time with a
 * Gaussian-Euclidean disintegration and unit metric
 */
template <class Model, class BaseRNG>
class unit_e_aaps
    : public base_aaps<Model, unit_e_metric, aaps_position_leapfrog, BaseRNG> {
 public:
  unit_e_aaps(const Model& model, BaseRNG& rng)
      : base_aaps<Model, unit_e_metric, aaps_position_leapfrog, BaseRNG>(model,
                                                                         rng) {}
};

}  // namespace mcmc
}  // namespace stan
#endif
