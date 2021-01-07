#ifndef STAN_MCMC_HOP_HOP_ADAPTER_HPP
#define STAN_MCMC_HOP_HOP_ADAPTER_HPP

#include <stan/callbacks/logger.hpp>
#include <stan/mcmc/base_adaptation.hpp>
#include <stan/mcmc/nesterov_dual_avg_adaptation.hpp>
#include <stan/mcmc/base_adapter.hpp>
#include <stan/mcmc/var_adaptation.hpp>

namespace stan {
namespace mcmc {

struct hop_adaptation : nesterov_adaptation{};
  
class hop_adapter : public base_adapter {
public:
  
  hop_adapter() {}

  hop_adaptation& get_hop_adaptation() {
    return hop_adaptation_;
  }

  const hop_adaptation& get_hop_adaptation() const noexcept {
    return hop_adaptation_;
  }

 protected:
  hop_adaptation hop_adaptation_;
};


class hop_var_adapter : public base_adapter {
 public:
  explicit hop_var_adapter(int n) : var_adaptation_(n) {}

  hop_adaptation& get_hop_adaptation() {
    return hop_adaptation_;
  }

  const hop_adaptation& get_hop_adaptation() const noexcept {
    return hop_adaptation_;
  }

  var_adaptation& get_var_adaptation() { return var_adaptation_; }

  void set_window_params(unsigned int num_warmup, unsigned int init_buffer,
                         unsigned int term_buffer, unsigned int base_window,
                         callbacks::logger& logger) {
    var_adaptation_.set_window_params(num_warmup, init_buffer, term_buffer,
                                      base_window, logger);
  }

 protected:
  hop_adaptation hop_adaptation_;
  var_adaptation var_adaptation_;
};

}  // namespace mcmc

}  // namespace stan

#endif
