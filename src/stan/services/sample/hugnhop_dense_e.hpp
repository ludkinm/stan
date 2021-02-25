#ifndef STAN_SERVICES_SAMPLE_HUGNHOP_DENSE_E_HPP
#define STAN_SERVICES_SAMPLE_HUGNHOP_DENSE_E_HPP

#include <stan/callbacks/interrupt.hpp>
#include <stan/callbacks/logger.hpp>
#include <stan/callbacks/writer.hpp>
#include <stan/io/var_context.hpp>
#include <stan/math/prim.hpp>
#include <stan/mcmc/hugnhop/dense_e_hugnhop.hpp>
#include <stan/services/error_codes.hpp>
#include <stan/services/util/run_sampler.hpp>
#include <stan/services/util/create_rng.hpp>
#include <stan/services/util/initialize.hpp>
#include <stan/services/util/inv_metric.hpp>
#include <vector>

namespace stan {
namespace services {
namespace sample {
template <class Model>
int hugnhop_dense_e(Model& model, const stan::io::var_context& init,
                    const stan::io::var_context& init_inv_metric,
                    unsigned int random_seed, unsigned int chain,
                    double init_radius, int num_warmup, int num_samples,
                    int num_thin, bool save_warmup, int refresh,
                    double stepsize, double stepsize_jitter, double int_time,
                    double lambda,
                    callbacks::interrupt& interrupt, callbacks::logger& logger,
                    callbacks::writer& init_writer,
                    callbacks::writer& sample_writer,
                    callbacks::writer& diagnostic_writer) {
  boost::ecuyer1988 rng = util::create_rng(random_seed, chain);

  std::vector<int> disc_vector;
  std::vector<double> cont_vector = util::initialize(
      model, init, rng, init_radius, true, logger, init_writer);

  Eigen::MatrixXd inv_metric;
  try {
    inv_metric = util::read_dense_inv_metric(init_inv_metric,
                                             model.num_params_r(), logger);
    util::validate_dense_inv_metric(inv_metric, logger);
  } catch (const std::domain_error& e) {
    return error_codes::CONFIG;
  }

  stan::mcmc::dense_e_hugnhop<Model, boost::ecuyer1988> sampler(model, rng);

  sampler.kernel0.set_metric(inv_metric);
  sampler.kernel0.set_nominal_stepsize_and_T(stepsize, int_time);
  sampler.kernel0.set_stepsize_jitter(stepsize_jitter);

  sampler.kernel1.set_metric(inv_metric);
  sampler.kernel1.set_gamma(lambda);

  util::run_sampler(sampler, model, cont_vector, num_warmup, num_samples,
                    num_thin, refresh, save_warmup, rng, interrupt, logger,
                    sample_writer, diagnostic_writer);

  return error_codes::OK;
}

template <class Model>
int hugnhop_dense_e(Model& model, const stan::io::var_context& init,
                    unsigned int random_seed, unsigned int chain,
                    double init_radius, int num_warmup, int num_samples,
                    int num_thin, bool save_warmup, int refresh,
                    double stepsize, double stepsize_jitter, double int_time,
                    double lambda, 
                    callbacks::interrupt& interrupt, callbacks::logger& logger,
                    callbacks::writer& init_writer,
                    callbacks::writer& sample_writer,
                    callbacks::writer& diagnostic_writer) {
  stan::io::dump dmp
      = util::create_unit_e_dense_inv_metric(model.num_params_r());
  stan::io::var_context& unit_e_metric = dmp;

  return hugnhop_dense_e(model, init, unit_e_metric, random_seed, chain,
                         init_radius, num_warmup, num_samples, num_thin,
                         save_warmup, refresh, stepsize, stepsize_jitter,
                         int_time, lambda, interrupt, logger,
                         init_writer, sample_writer, diagnostic_writer);
}

}  // namespace sample
}  // namespace services
}  // namespace stan
#endif