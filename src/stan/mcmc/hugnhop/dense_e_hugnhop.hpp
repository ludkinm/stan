#ifndef STAN_MCMC_HUGNHOP_DENSE_E_HUGNHOP_HPP
#define STAN_MCMC_HUGNHOP_DENSE_E_HUGNHOP_HPP

#include <stan/mcmc/hug/dense_e_hug.hpp>
#include <stan/mcmc/hop/dense_e_hop.hpp>
#include <stan/mcmc/base_alternator.hpp>

namespace stan {
namespace mcmc {

template <class Model, class BaseRNG>
class dense_e_hugnhop : public base_alternator<dense_e_hug<Model, BaseRNG>,
                                               dense_e_hop<Model, BaseRNG>> {
 public:
  dense_e_hugnhop(const Model& model, BaseRNG& rng)
      : base_alternator<dense_e_hug<Model, BaseRNG>,
                        dense_e_hop<Model, BaseRNG>>{
          dense_e_hug<Model, BaseRNG>{model, rng},
          dense_e_hop<Model, BaseRNG>{model, rng}} {}
};

}  // namespace mcmc
}  // namespace stan
#endif
