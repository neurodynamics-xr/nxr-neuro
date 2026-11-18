// Shared by the old-code golden generators (gen_*_golden.cpp, compiled against the code BEFORE a refactor) and the tests that compare the new code to them:
// a matrix is recorded as "name rows cols hash sum" where hash is the FNV-1a 64 of its raw doubles (equal hash = bitwise equal) and sum is its element sum (%.17g, to read a difference).
#pragma once
#include <Eigen/Dense>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace goldenhash {
inline std::uint64_t fnv(const double* p, std::size_t n) {
    std::uint64_t h = 1469598103934665603ull; const unsigned char* b = reinterpret_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n * sizeof(double); ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
struct Entry { long rows = 0, cols = 0; std::uint64_t hash = 0; double sum = 0; };
inline Entry entryOf(const Eigen::MatrixXd& A) { Eigen::MatrixXd c = A; Entry e; e.rows = c.rows(); e.cols = c.cols(); e.hash = fnv(c.data(), static_cast<std::size_t>(c.size())); e.sum = c.sum(); return e; }
class Writer {
public:
    explicit Writer(const std::string& path) : f_(path) {}
    template <class D> void put(const std::string& name, const Eigen::MatrixBase<D>& A) {
        if constexpr (Eigen::NumTraits<typename D::Scalar>::IsComplex) { putReal(name + ".re", Eigen::MatrixXd(A.real())); putReal(name + ".im", Eigen::MatrixXd(A.imag())); }
        else putReal(name, Eigen::MatrixXd(A));
    }
    void put(const std::string& name, double v) { putReal(name, Eigen::MatrixXd::Constant(1, 1, v)); }
    template <class V> void putAll(const std::string& name, const V& list) { for (std::size_t i = 0; i < list.size(); ++i) put(name + "." + std::to_string(i), list[i]); }
private:
    void putReal(const std::string& name, const Eigen::MatrixXd& A) { const Entry e = entryOf(A); char b[256]; std::snprintf(b, sizeof b, "%s %ld %ld %llu %.17g\n", name.c_str(), e.rows, e.cols, static_cast<unsigned long long>(e.hash), e.sum); f_ << b; }
    std::ofstream f_;
};
class Reader {
public:
    explicit Reader(const std::string& path) { std::ifstream f(path); std::string l; while (std::getline(f, l)) { std::istringstream s(l); std::string n; Entry e; unsigned long long h; if (s >> n >> e.rows >> e.cols >> h >> e.sum) { e.hash = h; m_[n] = e; } } }
    bool good() const { return !m_.empty(); }
    // 0: equal bitwise; 1: differs (the message says how)
    template <class D> bool same(const std::string& name, const Eigen::MatrixBase<D>& A, std::string* why = nullptr) const {
        if constexpr (Eigen::NumTraits<typename D::Scalar>::IsComplex) return sameReal(name + ".re", Eigen::MatrixXd(A.real()), why) && sameReal(name + ".im", Eigen::MatrixXd(A.imag()), why);
        else return sameReal(name, Eigen::MatrixXd(A), why);
    }
    bool same(const std::string& name, double v, std::string* why = nullptr) const { return sameReal(name, Eigen::MatrixXd::Constant(1, 1, v), why); }
private:
    bool sameReal(const std::string& name, const Eigen::MatrixXd& A, std::string* why) const {
        auto it = m_.find(name); if (it == m_.end()) { if (why) *why = name + ": not in the golden"; return false; }
        const Entry e = entryOf(A);
        if (e.rows != it->second.rows || e.cols != it->second.cols) { if (why) *why = name + ": shape"; return false; }
        if (e.hash != it->second.hash) { if (why) { std::ostringstream s; s.precision(17); s << name << ": sum " << e.sum << " vs golden " << it->second.sum; *why = s.str(); } return false; }
        return true;
    }

    std::map<std::string, Entry> m_;
};
}  // namespace goldenhash
