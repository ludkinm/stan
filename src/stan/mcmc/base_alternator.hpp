#ifndef STAN_MCMC_BASE_ALTERNATOR_HPP
#define STAN_MCMC_BASE_ALTERNATOR_HPP

#include <iterator>
#include <stan/mcmc/base_mcmc.hpp>

namespace stan {
namespace mcmc {

template <class K0, class K1>
class base_alternator : public base_mcmc {
 public:
  double alpha0;
  K0 kernel0;
  K1 kernel1;

  base_alternator(K0 const& k0, K1 const& k1) : kernel0{k0}, kernel1{k1} {}

  sample transition(sample& init_sample, callbacks::logger& logger) override {
    sample x = kernel0.transition(init_sample, logger);
    alpha0 = x.accept_stat();
    x = kernel1.transition(x, logger);
    return x;
  }

  void get_sampler_param_names(std::vector<std::string>& names) override {
    std::vector<std::string> k0_names;
    kernel0.get_sampler_param_names(k0_names);
    std::vector<std::string> k1_names;
    kernel1.get_sampler_param_names(k1_names);
    for (auto& n : k0_names)
      n = "K0:" + n;
    for (auto& n : k1_names)
      n = "K1:" + n;
    std::copy(k0_names.begin(), k0_names.end(), std::back_inserter(names));
    std::copy(k1_names.begin(), k1_names.end(), std::back_inserter(names));
  }

  void get_sampler_params(std::vector<double>& values) {
    kernel0.get_sampler_params(values);
    kernel1.get_sampler_params(values);
  }

  void write_sampler_state(callbacks::writer& writer) {
    kernel0.write_sampler_state(writer);
    kernel1.write_sampler_state(writer);
  }

  void get_sampler_diagnostic_names(std::vector<std::string>& model_names,
                                    std::vector<std::string>& names) override {
    std::vector<std::string> k0_names;
    kernel0.get_sampler_diagnostic_names(model_names, k0_names);
    std::vector<std::string> k1_names;
    kernel1.get_sampler_diagnostic_names(model_names, k1_names);
    for (auto& n : k0_names)
      n = "K0:" + n;
    for (auto& n : k1_names)
      n = "K1:" + n;
    std::copy(k0_names.begin(), k0_names.end(), std::back_inserter(names));
    names.push_back("K0:alpha");
    std::copy(k1_names.begin(), k1_names.end(), std::back_inserter(names));
  }

  void get_sampler_diagnostics(std::vector<double>& values) {
    kernel0.get_sampler_diagnostics(values);
    values.push_back(alpha0);
    kernel1.get_sampler_diagnostics(values);
  }
};

}  // namespace mcmc
}  // namespace stan
#endif
