#ifndef STAN_MCMC_HOP_UNIT_E_STATIC_HOP_HPP
#define STAN_MCMC_HOP_UNIT_E_STATIC_HOP_HPP

#include <stan/mcmc/hmc/hamiltonians/unit_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/unit_e_metric.hpp>
#include <stan/mcmc/hop/base_hop.hpp>

namespace stan {
namespace mcmc {

template <class Model, class BaseRNG>
class unit_e_hop
    : public base_hop<Model, unit_e_metric, BaseRNG> {
 public:
  unit_e_hop(const Model& model, BaseRNG& rng)
      : base_hop<Model, unit_e_metric, BaseRNG>(model, rng) {}
};

}  // namespace mcmc
}  // namespace stan
#endif
