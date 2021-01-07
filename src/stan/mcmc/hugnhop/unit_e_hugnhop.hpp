#ifndef STAN_MCMC_HUGNHOP_UNIT_E_HUGNHOP_HPP
#define STAN_MCMC_HUGNHOP_UNIT_E_HUGNHOP_HPP

#include <stan/mcmc/hug/unit_e_hug.hpp>
#include <stan/mcmc/hop/unit_e_hop.hpp>
#include <stan/mcmc/base_alternator.hpp>

namespace stan {
namespace mcmc {

template <class Model, class BaseRNG>
class unit_e_hugnhop : public base_alternator<unit_e_hug<Model, BaseRNG>,
                                              unit_e_hop<Model, BaseRNG>> {
 public:
  unit_e_hugnhop(const Model& model, BaseRNG& rng)
      : base_alternator<unit_e_hug<Model, BaseRNG>, unit_e_hop<Model, BaseRNG>>{
          unit_e_hug<Model, BaseRNG>{model, rng},
          unit_e_hop<Model, BaseRNG>{model, rng}} {}
};

}  // namespace mcmc
}  // namespace stan
#endif
