#ifndef STAN_MCMC_HUG_DIAG_E_HUG_HPP
#define STAN_MCMC_HUG_DIAG_E_HUG_HPP

#include <stan/mcmc/hmc/hamiltonians/diag_e_point.hpp>
#include <stan/mcmc/hmc/hamiltonians/diag_e_metric.hpp>
#include <stan/mcmc/hmc/static/base_static_hmc.hpp>
#include <stan/mcmc/hug/hug_bouncer.hpp>

namespace stan {
namespace mcmc {

template <class Model, class BaseRNG>
class diag_e_hug
    : public base_static_hmc<Model, diag_e_metric, hug_bouncer, BaseRNG> {
 public:
  diag_e_hug(const Model& model, BaseRNG& rng)
      : base_static_hmc<Model, diag_e_metric, hug_bouncer, BaseRNG>(model,
                                                                    rng) {}
};

}  // namespace mcmc
}  // namespace stan
#endif
