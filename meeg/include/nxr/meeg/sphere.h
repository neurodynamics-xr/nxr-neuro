// nxr::meeg — spheres fitted to a head surface (docs/specs/2026-11-10-neuro-forward-inverse.md, section 8).
// Pure: arrays in, arrays out; Eigen only. The least-squares fit is minimised with the Nelder-Mead simplex (minimise.h).
#pragma once
#include <Eigen/Dense>
#include <vector>

#include "nxr/meeg/minimise.h"

namespace nxr::meeg {

struct Sphere {
  Eigen::Vector3d centre = Eigen::Vector3d::Zero();
  double radius = 0;
};

// The cost a sphere fit minimises, over the points x_i with weights w_i.
enum class SphereCost {
  Geometric,   // sum w_i (|x_i - c| - R)^2: the distances to the sphere, R the weighted mean distance for the centre c
  Algebraic    // sum w_i (|x_i|^2 - 2 c.x_i - k)^2, R^2 = k + |c|^2: linear, closed form (the start of the geometric fit)
};
struct SphereFitOptions {
  SphereCost cost = SphereCost::Geometric;
  NelderMeadOptions minimiser = []{ NelderMeadOptions o; o.zeroStep = 0.05; return o; }();   // the centre is searched in units of the points' rms spread about their mean
};

// The sphere fitted to the points [n x 3] (weights [n], default all 1; non-negative). std::invalid_argument for fewer than 4 points, a NaN, a negative or
// all-zero weight, or weights of the wrong length.
Sphere fitSphere(const Eigen::MatrixXd& points, const Eigen::VectorXd& weights = Eigen::VectorXd(), const SphereFitOptions& opt = {});

// The overlapping spheres of Huang, Mosher & Leahy (1999): one sphere per sensor, fitted to the head surface points with weights that fall with the distance to the sensor.
// ACCURACY: up to 3.4% from Brainstorm's bst_os (max-norm gain difference on the synthetic parity case); its fit needs a realistic-head (BEM) lead field; to be revisited when a BEM exists.
struct SensorSphereOptions {
  double weightExponent = 4;      // w_i = sum over the sensor's points k of |weight_k| d_ik^(-p): p = 4. Chosen on the reference's output (spheres of 13 synthetic sensors, p from 1 to 8:
                                  // the smallest gain mismatch is at p = 4); the reference's own weighting is not published in a form that reproduces its spheres exactly (spec section 8)
  SphereFitOptions fit;
};
// `sensorPoints` [P x 3] with `sensorWeights` [P] and `sensorChannel` [P] (the channel each point belongs to, 0..channels-1); returns one sphere per channel.
// A channel without a point gets the sphere fitted with uniform weights. std::invalid_argument for sizes that disagree or a point on a surface vertex.
std::vector<Sphere> sensorSpheres(const Eigen::MatrixXd& surface, const Eigen::MatrixXd& sensorPoints, const Eigen::VectorXd& sensorWeights, const std::vector<int>& sensorChannel,
                                  int channels, const SensorSphereOptions& opt = {});

}  // namespace nxr::meeg
