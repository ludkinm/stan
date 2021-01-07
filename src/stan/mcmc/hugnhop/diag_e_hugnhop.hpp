#ifndef STAN_MCMC_HUGNHOP_DIAG_E_HUGNHOP_HPP
#define STAN_MCMC_HUGNHOP_DIAG_E_HUGNHOP_HPP

#include <stan/mcmc/hug/diag_e_hug.hpp>
#include <stan/mcmc/hop/diag_e_hop.hpp>
#include <stan/mcmc/base_alternator.hpp>

namespace stan {
namespace mcmc {

template <class Model, class BaseRNG>
class diag_e_hugnhop : public base_alternator<diag_e_hug<Model, BaseRNG>,
                                              diag_e_hop<Model, BaseRNG>> {
 public:
  diag_e_hugnhop(const Model& model, BaseRNG& rng)
      : base_alternator<diag_e_hug<Model, BaseRNG>, diag_e_hop<Model, BaseRNG>>{
          diag_e_hug<Model, BaseRNG>{model, rng},
          diag_e_hop<Model, BaseRNG>{model, rng}} {}
};

}  // namespace mcmc
}  // namespace stan
#endif
