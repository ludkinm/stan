#ifndef STAN_MCMC_HMC_AAPS_BASE_AAPS_HPP
#define STAN_MCMC_HMC_AAPS_BASE_AAPS_HPP

#include <iterator>
#include <numeric>
#include <sstream>
#include <stan/callbacks/logger.hpp>
#include <stan/math/prim.hpp>
#include <stan/mcmc/hmc/base_hmc.hpp>
#include <stan/mcmc/hmc/hamiltonians/ps_point.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace stan {
namespace mcmc {
namespace aaps_details {
// numerically stable conversion of a vector (x_i)
// to vector with element i = log(sum_{j=1}^i exp(x_j))
void log_cum_sum_exp(std::vector<double>& x) {
  // find max of whole vector
  double max = *std::max_element(x.begin(), x.end());
  // convert to x[i] = exp(x[i]  - max)
  std::transform(x.begin(), x.end(), x.begin(),
                 [max](double i) { return exp(i - max); });
  // convert to x[i] = sum(x[0:i])
  std::partial_sum(x.begin(), x.end(), x.begin());
  // convert to x[i] = max + log(sum(x[i]))
  std::transform(x.begin(), x.end(), x.begin(),
                 [max](double i) { return max + log(i); });
}
}  // namespace aaps_details

void print(std::string const& header, std::vector<double> const& vec,
           callbacks::logger& logger) {
  std::stringstream ss;
  ss << header;
  std::for_each(vec.begin(), vec.end(), [&ss](double r) { ss << r << ' '; });
  logger.info(ss);
}

void print(std::string const& header, Eigen::VectorXd const& vec,
           callbacks::logger& logger) {
  std::stringstream ss;
  ss << header;
  for (std::size_t i{0}; i < vec.size(); ++i)
    ss << vec[i] << ' ';
  logger.info(ss);
}

template <class Hamiltonian>
class aaps_position_leapfrog {
 public:
  bool evolve(typename Hamiltonian::PointType& z, Hamiltonian& hamiltonian,
              const double epsilon, callbacks::logger& logger) {
    logger.info("evolve begin");
    print("z.q=", z.q, logger);
    print("z.p=", z.p, logger);

    logger.info("update position");
    z.q += 0.5 * epsilon * hamiltonian.dtau_dp(z);
    hamiltonian.update_potential_gradient(z, logger);
    print("z.q=", z.q, logger);

    logger.info("get gradient");
    Eigen::VectorXd g = hamiltonian.dphi_dq(z, logger);
    double pre_angle = g.dot(z.p);

    print("g=", g, logger);
    print("z.p=", z.p, logger);
    logger.info("pre_angle = " + std::to_string(pre_angle));

    logger.info("update momenta");
    z.p -= epsilon * g;
    double post_angle = g.dot(z.p);

    print("z.p=", z.p, logger);
    logger.info("post_angle = " + std::to_string(post_angle));

    logger.info("update position");
    z.q += 0.5 * epsilon * hamiltonian.dtau_dp(z);
    hamiltonian.update_potential_gradient(z, logger);

    logger.info("evolve end");
    print("z.q=", z.q, logger);
    print("z.p=", z.p, logger);

    // return true if same sign, i.e. **not** an apogee
    return std::signbit(pre_angle) == std::signbit(post_angle);
  }
};

/**
 * The Apogee-Apogee-Path-Sampler (AAPS)
 */
template <class Model, template <class, class> class Hamiltonian,
          template <class> class Integrator, class BaseRNG>
class base_aaps : public base_hmc<Model, Hamiltonian, Integrator, BaseRNG> {
 public:
  typedef typename Hamiltonian<Model, BaseRNG>::PointType PointType;

  base_aaps(const Model& model, BaseRNG& rng)
      : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, rng),
        path_length_(0),
        index_(0),
        energy_(0) {}

  /**
   * specialized constructor for specified diag mass matrix
   */
  base_aaps(const Model& model, BaseRNG& rng, Eigen::VectorXd& inv_e_metric)
      : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, rng,
                                                          inv_e_metric),
        path_length_(0),
        index_(0),
        energy_(0) {}

  /**
   * specialized constructor for specified dense mass matrix
   */
  base_aaps(const Model& model, BaseRNG& rng, Eigen::MatrixXd& inv_e_metric)
      : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, rng,
                                                          inv_e_metric),
        path_length_(0),
        index_(0),
        energy_(0) {}

  ~base_aaps() {}

  void set_metric(const Eigen::MatrixXd& inv_e_metric) {
    this->z_.set_metric(inv_e_metric);
  }

  void set_metric(const Eigen::VectorXd& inv_e_metric) {
    this->z_.set_metric(inv_e_metric);
  }

  std::size_t sample_from_path(PointType& z_propose, double& log_propose_energy,
                               std::vector<PointType>& path,
                               callbacks::logger& logger) {
    logger.info("~~sample_from_path~~");
    std::vector<double> log_rhos(path.size());
    print("set log_rhos to vector of 0s:", log_rhos, logger);
    logger.info("calculating log_rho = negative hamiltonian at each point");
    std::transform(path.begin(), path.end(), log_rhos.begin(),
                   [this](PointType& z) { return -this->hamiltonian_.H(z); });
    print("log_rhos=", log_rhos, logger);
    logger.info("converting rhos to log_cum_sum_exp(log_rhos)");
    aaps_details::log_cum_sum_exp(log_rhos);
    print("log_cum_sum_exp(log_rhos)=", log_rhos, logger);
    logger.info("sampling log_u");
    double log_u = log(this->rand_uniform_());
    logger.info("got log_u = " + std::to_string(log_u));
    log_u += log_rhos.back();
    logger.info("got log_sum_rhos + log_u = " + std::to_string(log_u));
    logger.info(
        "find_if'ing first element: in u + log_sum_exp(-Hs) + log_u < "
        "log_cum_sum(-Hs)");
    auto it = std::find_if(log_rhos.begin(), log_rhos.end(),
                           [log_u](double r) { return log_u < r; });
    logger.info("got *it = " + std::to_string(*it));
    logger.info("finding index based on this iterator");
    std::size_t index = std::distance(log_rhos.begin(), it);
    logger.info("about to set z_propose to index = " + std::to_string(index));
    z_propose = path.at(index);
    logger.info("about to set log_rho_propose");
    log_propose_energy = -log_rhos.at(index);
    return index;
  }

  std::size_t build_path(PointType& z, std::vector<PointType>& path,
                         callbacks::logger& logger) {
    // makes path of points -B, -B+1, ..., -1, 0, 1,..., F-1, F
    // where 0 is the input state z
    // B and F are the points just before an apogge going backwards/forwards
    // resp.
    logger.info("~~build_path~~");
    bool not_apogee{true};
    do {  // backwards
      path.push_back(z);
      not_apogee = this->integrator_.evolve(z, this->hamiltonian_,
                                            -this->epsilon_, logger);
    } while (not_apogee);
    // reorder so its -B, -B+1, ... -1, 0
    std::size_t bck_length{path.size()};
    std::reverse(std::begin(path), std::end(path));
    logger.info("backwards length = " + std::to_string(bck_length));
    // move z back to start point
    z = path[0];
    not_apogee = not_apogee = this->integrator_.evolve(z, this->hamiltonian_,
                                                       this->epsilon_, logger);
    while (not_apogee) {
      path.push_back(z);
      not_apogee = this->integrator_.evolve(z, this->hamiltonian_,
                                            this->epsilon_, logger);
    }
    logger.info("forwards length = "
                + std::to_string(path.size() - bck_length));
    return bck_length;
  }

  sample transition(sample& init_sample, callbacks::logger& logger) {
    logger.info("\n\n~aaps::transition~");
    // Initialize the algoritm
    this->sample_stepsize();

    print("init_sample.cont_params() = ", init_sample.cont_params(), logger);
    // this sets this->z.q to the continuous parameters in init_sample
    this->seed(init_sample.cont_params());

    // sample a new momenta
    this->hamiltonian_.sample_p(this->z_, this->rand_int_);

    // log stuff to logger
    this->hamiltonian_.init(this->z_, logger);

    // init paths
    logger.info("about to build path");
    std::vector<PointType> path;
    std::size_t bck_length = build_path(this->z_, path, logger);
    this->path_length_ = path.size();
    this->index_ = 1 + sample_from_path(this->z_, this->energy_, path, logger)
                   - bck_length;

    logger.info("setting infos");
    // store energy for the sampled point
    this->energy_ = this->hamiltonian_.H(this->z_);
    logger.info("returning");
    // return sampled point, with logpi and
    return sample(this->z_.q, -this->hamiltonian_.V(this->z_), 1.0);
  }

  void get_sampler_param_names(std::vector<std::string>& names) {
    names.push_back("stepsize__");
    names.push_back("pathlength__");
    names.push_back("index__");
    names.push_back("energy__");
  }

  void get_sampler_params(std::vector<double>& values) {
    values.push_back(this->epsilon_);
    values.push_back(this->path_length_);
    values.push_back(this->index_);
    values.push_back(this->energy_);
  }

 private:
  int path_length_;
  int index_;
  double energy_;
};  // namespace mcmc

}  // namespace mcmc
}  // namespace stan
#endif
