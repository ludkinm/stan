#ifndef STAN_MCMC_HMC_AAPS_BASE_AAPS_HPP
#define STAN_MCMC_HMC_AAPS_BASE_AAPS_HPP

#include <iterator>
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

template <class Hamiltonian>
class aaps_position_leapfrog {
 public:
  bool evolve(typename Hamiltonian::PointType& z, Hamiltonian& hamiltonian,
              const double epsilon, callbacks::logger& logger) {
    // update position
    z.q += 0.5 * epsilon * hamiltonian.dtau_dp(z);
    hamiltonian.update_potential_gradient(z, logger);

    // momenta
    Eigen::VectorXd g = hamiltonian.dphi_dq(z, logger);
    double pre_angle = g.dot(z.p);
    z.p -= epsilon * g;
    double post_angle = g.dot(z.p);

    // update position
    z.q += 0.5 * epsilon * hamiltonian.dtau_dp(z);
    hamiltonian.update_potential_gradient(z, logger);

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
        energy_(0) {}

  /**
   * specialized constructor for specified diag mass matrix
   */
  base_aaps(const Model& model, BaseRNG& rng, Eigen::VectorXd& inv_e_metric)
      : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, rng,
                                                          inv_e_metric),
        path_length_(0),
        energy_(0) {}

  /**
   * specialized constructor for specified dense mass matrix
   */
  base_aaps(const Model& model, BaseRNG& rng, Eigen::MatrixXd& inv_e_metric)
      : base_hmc<Model, Hamiltonian, Integrator, BaseRNG>(model, rng,
                                                          inv_e_metric),
        path_length_(0),
        energy_(0) {}

  ~base_aaps() {}

  void set_metric(const Eigen::MatrixXd& inv_e_metric) {
    this->z_.set_metric(inv_e_metric);
  }

  void set_metric(const Eigen::VectorXd& inv_e_metric) {
    this->z_.set_metric(inv_e_metric);
  }

  enum Direction { backwards, forwards };

  void sample_from_path(PointType& z_propose, double& p_propose,
                        std::vector<PointType>& path, double& total_ps,
                        callbacks::logger& logger) {
    // if this function is called, we know path is non-empty
    // storage for exp(-Hs)
    logger.info("sample_from_path");
    logger.info("about to make vector of rhos");
    std::vector<double> rhos(path.size());
    logger.info("about to calculate the hamiltonian for each point in path");
    // go along path and calculate exp(-hamiltonian)
    std::transform(
        path.begin(), path.end(), rhos.begin(),
        [this](PointType& z) { return exp(-this->hamiltonian_.H(z)); });
    // annoyingly non const!
    // cum_sum(rhos)
    logger.info("about to cum sum");
    std::partial_sum(rhos.begin(), rhos.end(), rhos.begin(), std::plus<>());
    // find the first rho for which rho/max_rho < Unif(0,1)
    // sample a point
    logger.info("about to get rhos.back");
    total_ps = rhos.back();
    logger.info("about to sample from rand_uniform_");
    double u = this->rand_uniform_();
    logger.info("got u = ");
    logger.info(std::to_string(u));
    logger.info("about to find_if the first time rho/total_rho is less than u");
    auto it = std::find_if(rhos.begin(), rhos.end(), [u, total_ps](double r) {
      return u < r / total_ps;
    });
    logger.info("about to get distance between that position and rhos.begin()");
    std::size_t index = std::distance(rhos.begin(), it);
    logger.info("about to set z_propose to index = ");
    logger.info(std::to_string(index));
    z_propose = path.at(index);
    logger.info("about to set p_propose");
    p_propose = rhos.at(index);
  }

  void build_path(Direction dir, PointType& z, std::vector<PointType>& path,
                  callbacks::logger& logger) {
    // samples a point from the `dir`-path starting as z.
    // returns the proposal in `z_propose`
    // returns the sum_{z in dir-path} exp(-H(z)) in `sum_rho`
    double eps = ((dir == Direction::forwards) ? 1 : -1) * this->epsilon_;

    // build upto first apogee
    bool not_apogee{true};
    do {
      not_apogee = this->integrator_.evolve(z, this->hamiltonian_, eps, logger);
      path.push_back(z);
    } while (not_apogee);
  }

  sample transition(sample& init_sample, callbacks::logger& logger) {
    logger.info("Made it to AAPS transition");
    // Initialize the algoritm
    this->sample_stepsize();

    // this sets this->z.q to the continuous parameters in init_sample
    this->seed(init_sample.cont_params());

    // sample a new momenta
    this->hamiltonian_.sample_p(this->z_, this->rand_int_);

    // log stuff to logger
    this->hamiltonian_.init(this->z_, logger);

    logger.info("about to init paths");
    // init paths
    PointType z_fwd(this->z_);  // State at forward end of trajectory
    PointType z_bck(z_fwd);     // State at backward end of trajectory
    std::vector<PointType> path_fwd;
    std::vector<PointType> path_bck;

    logger.info("about to build paths");
    // build paths
    build_path(Direction::forwards, z_fwd, path_fwd, logger);
    build_path(Direction::backwards, z_bck, path_bck, logger);

    std::stringstream ss;
    ss << "about to sample from paths. Sizes are f = " << path_fwd.size()
       << " b = " << path_bck.size();
    logger.info(ss);
    double accept_prob = 0.0;
    if (path_fwd.size() == 0 && path_bck.size() == 0) {
      logger.info("case: both emtpty!");
      // didn't move
    } else if (path_bck.size() == 0) {
      logger.info("case: path_bck is empty!");
      // back path is non-empty
      double p_fwd{0.0};
      double fwd_total_p{0.0};
      sample_from_path(z_fwd, p_fwd, path_fwd, fwd_total_p, logger);
      this->z_ = z_fwd;
      accept_prob = p_fwd / fwd_total_p;
    } else if (path_fwd.size() == 0) {
      logger.info("case: path_fwd is empty!");
      // back path is non-empty
      double p_bck{0.0};
      double bck_total_p{0.0};
      sample_from_path(z_bck, p_bck, path_bck, bck_total_p, logger);
      this->z_ = z_bck;
      accept_prob = p_bck / bck_total_p;
    } else {
      logger.info("case: neither path is empty!");
      // sample a point on each path...
      double p_fwd{0.0};
      double p_bck{0.0};
      double fwd_total_p{0.0};
      double bck_total_p{0.0};
      sample_from_path(z_fwd, p_fwd, path_fwd, fwd_total_p, logger);
      sample_from_path(z_bck, p_bck, path_bck, bck_total_p, logger);
      // then choose between them proportional to the relative
      // total probability on their paths
      accept_prob = fwd_total_p / (fwd_total_p + bck_total_p);
      if (accept_prob < 1 && this->rand_uniform_() > accept_prob) {
        this->z_ = z_fwd;
      } else {
        this->z_ = z_bck;
      }
    }
    logger.info("setting infos");
    this->path_length_ = path_fwd.size() + path_bck.size();
    // store energy for the sampled point
    this->energy_ = this->hamiltonian_.H(this->z_);
    logger.info("returning");
    // return sampled point, with logpi and
    return sample(this->z_.q, -this->hamiltonian_.V(this->z_), accept_prob);
  }

  void get_sampler_param_names(std::vector<std::string>& names) {
    names.push_back("stepsize__");
    names.push_back("pathlength__");
    names.push_back("energy__");
  }

  void get_sampler_params(std::vector<double>& values) {
    values.push_back(this->epsilon_);
    values.push_back(this->path_length_);
    values.push_back(this->energy_);
  }

 private:
  int path_length_;
  double energy_;
};

}  // namespace mcmc
}  // namespace stan
#endif
