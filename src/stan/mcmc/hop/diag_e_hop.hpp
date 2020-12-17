#ifndef STAN_MCMC_HOP_DIAG_E_HOP_HPP
#define STAN_MCMC_HOP_DIAG_E_HOP_HPP

#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp>
#include <stan/mcmc/hop/base_hop.hpp>

namespace stan {
namespace mcmc {

template <class Model, class BaseRNG>
class diag_e_hop
    : public base_hop<Model, diag_e_metric, BaseRNG> {
 public:
  diag_e_hop(const Model& model, BaseRNG& rng)
      : base_hop<Model, diag_e_metric, BaseRNG>(model, rng) {}
};

}  // namespace mcmc
}  // namespace stan
#endif
