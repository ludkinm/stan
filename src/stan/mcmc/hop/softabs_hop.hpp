#ifndef STAN_MCMC_HOP_SOFTABS_HOP_HPP
#define STAN_MCMC_HOP_SOFTABS_HOP_HPP

#include <stan/mcmc/hmc/hamiltonians/softabs_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/softabs_metric.hpp>
#include <stan/mcmc/hop/base_hop.hpp>

namespace stan {
namespace mcmc {
/**
 * Hamiltonian Monte Carlo implementation using the endpoint
 * of trajectories with a static integration time with a
 * Gaussian-Riemannian disintegration and SoftAbs metric
 */
template <class Model, class BaseRNG>
class softabs_hop
    : public base_hop<Model, softabs_metric, BaseRNG> {
 public:
  softabs_hop(const Model& model, BaseRNG& rng)
      : base_hop<Model, softabs_metric, BaseRNG>(model, rng) {}
};

}  // namespace mcmc
}  // namespace stan
#endif
