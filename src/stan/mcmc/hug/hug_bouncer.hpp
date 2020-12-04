#ifndef STAN_MCMC_HUG_HUG_BOUNCER_HPP
#define STAN_MCMC_HUG_HUG_BOUNCER_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/hmc/integrators/base_integrator.hpp>
#include <stan/math/prim/fun/Eigen.hpp>

namespace stan {
namespace mcmc {

template <class Hamiltonian>
class hug_bouncer : public base_integrator<Hamiltonian> {
 public:
  hug_bouncer() : base_integrator<Hamiltonian>() {}

  void evolve(typename Hamiltonian::PointType& z, Hamiltonian& hamiltonian,
              const double epsilon, callbacks::logger& logger) {
    logger.info("hug_bouncer::evolve begin");
    // step in q
    z.q -= epsilon * hamiltonian.dtau_dp(z);
    hamiltonian.update_potential_gradient(z, logger);
    // reflect through gradient
    const Eigen::VectorXd g = hamiltonian.dphi_dq(z, logger);
    z.p -= 2 * g.dot(z.p) / g.dot(g) * g;
    // step in q, again, update the gradient to be safe
    z.q -= epsilon * hamiltonian.dtau_dp(z);
    hamiltonian.update_potential_gradient(z, logger);
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
