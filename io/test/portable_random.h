// Random draws that are the same on every platform. The standard leaves std::normal_distribution and std::uniform_real_distribution implementation-defined, and
// libstdc++ and libc++ return different values from the same std::mt19937 seed (libstdc++'s Marsaglia polar method returns the second coordinate of the pair
// first, libc++'s the first), so a synthetic fixture made with them is a different fixture on each platform and a threshold can fall on either side of it.
// std::mt19937's own output IS fixed by the standard. These two draw from it by the same closed forms as libc++ (the values the fixtures were calibrated on):
//   uniform: (hi - lo) * (w0 + w1 * 2^32) / 2^64 + lo, from two 32-bit words, as std::generate_canonical<double, 53> does for a 32-bit generator;
//   normal:  the Marsaglia polar method on uniforms (-1, 1), rejecting S = u^2 + v^2 outside (0, 1], returning u * f and keeping v * f for the next call,
//            f = sqrt(-2 ln S / S).
// Only sqrt and log of the platform's libm enter, which agree to the last bit or two: far below anything a fixture's threshold resolves.
#pragma once
#include <cmath>
#include <cstdint>

namespace nxr::testing {

struct PortableUniform {
  double lo, hi;
  explicit PortableUniform(double l = 0.0, double h = 1.0) : lo(l), hi(h) {}
  template <class G> double operator()(G& g) const {
    const double w0 = static_cast<double>(g());
    const double w1 = static_cast<double>(g());
    double u = (w0 + w1 * 4294967296.0) / 18446744073709551616.0;
    if (u >= 1.0) u = std::nextafter(1.0, 0.0);
    return (hi - lo) * u + lo;
  }
};

struct PortableNormal {
  double mean, sd;
  explicit PortableNormal(double m = 0.0, double s = 1.0) : mean(m), sd(s) {}
  template <class G> double operator()(G& g) {
    double u1;
    if (hot_) { hot_ = false; u1 = saved_; }
    else {
      const PortableUniform sym(-1.0, 1.0);
      double u, v, s;
      do { u = sym(g); v = sym(g); s = u * u + v * v; } while (s > 1.0 || s == 0.0);
      const double f = std::sqrt(-2.0 * std::log(s) / s);
      saved_ = v * f; hot_ = true; u1 = u * f;
    }
    return u1 * sd + mean;
  }
 private:
  double saved_ = 0.0;
  bool hot_ = false;
};

}  // namespace nxr::testing
