#ifndef STAN_SERVICES_SAMPLE_HUGNHOP_UNIT_E_HPP
#define STAN_SERVICES_SAMPLE_HUGNHOP_UNIT_E_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/mcmc/fixed_param_sampler.hpp>
#include <stan/mcmc/hugnhop/unit_e_hugnhop.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/util/run_sampler.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/initialize.hpp>
#include <vector>

namespace stan {
namespace services {
namespace sample {
template <class Model>
int hugnhop_unit_e(Model& model, const stan::io::var_context& init,
                   unsigned int random_seed, unsigned int chain,
                   double init_radius, int num_warmup, int num_samples,
                   int num_thin, bool save_warmup, int refresh, double stepsize,
                   double stepsize_jitter, double int_time, double lambda,
                   double kappa, callbacks::interrupt& interrupt,
                   callbacks::logger& logger, callbacks::writer& init_writer,
                   callbacks::writer& sample_writer,
                   callbacks::writer& diagnostic_writer) {
  boost::ecuyer1988 rng = util::create_rng(random_seed, chain);

  std::vector<int> disc_vector;
  std::vector<double> cont_vector = util::initialize(
      model, init, rng, init_radius, true, logger, init_writer);

  stan::mcmc::unit_e_hugnhop<Model, boost::ecuyer1988> sampler(model, rng);

  sampler.kernel0.set_nominal_stepsize_and_T(stepsize, int_time);
  sampler.kernel0.set_stepsize_jitter(stepsize_jitter);
  sampler.kernel1.set_lambda_kappa(lambda, kappa);

  util::run_sampler(sampler, model, cont_vector, num_warmup, num_samples,
                    num_thin, refresh, save_warmup, rng, interrupt, logger,
                    sample_writer, diagnostic_writer);

  return error_codes::OK;
}

}  // namespace sample
}  // namespace services
}  // namespace stan
#endif
