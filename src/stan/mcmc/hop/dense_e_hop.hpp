#ifndef STAN_MCMC_HOP_DENSE_E_HOP_HPP
#define STAN_MCMC_HOP_DENSE_E_HOP_HPP

#include <stan/mcmc/hop/base_hop.hpp>
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
class dense_e_hop
    : public base_hop<Model, dense_e_metric, BaseRNG> {
 public:
  dense_e_hop(const Model& model, BaseRNG& rng)
      : base_hop<Model, dense_e_metric, BaseRNG>(model, rng) {}
};

}  // namespace mcmc
}  // namespace stan
#endif
