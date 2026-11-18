// A standard normal generator that gives the same numbers on every platform. std::normal_distribution is implementation-defined (libstdc++ and libc++ draw different
// values from the same engine), so a test that plants noise and checks a threshold must not use it. std::mt19937's own output is fixed by the standard.
#pragma once
#include <cmath>
#include <cstdint>
#include <random>
namespace nxr_test {
class PortableNormal {
 public:
  explicit PortableNormal(std::uint32_t seed) : rng_(seed) {}
  double operator()() {  // Box-Muller, both values of a pair used in turn
    if (have_) { have_ = false; return spare_; }
    const double u1 = (static_cast<double>(rng_()) + 0.5) / 4294967296.0, u2 = (static_cast<double>(rng_()) + 0.5) / 4294967296.0;
    const double r = std::sqrt(-2.0 * std::log(u1)), a = 6.283185307179586 * u2;
    spare_ = r * std::sin(a); have_ = true;
    return r * std::cos(a);
  }
 private:
  std::mt19937 rng_;
  double spare_ = 0.0;
  bool have_ = false;
};
}  // namespace nxr_test
