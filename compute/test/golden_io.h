// Shared by the part-2 filterbank tests: read a golden text file ("name rows cols\n values...") token by token.
#pragma once
#include <Eigen/Dense>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

struct Golden {
    std::ifstream f;
    explicit Golden(const std::string& path) : f(path) {}
    bool good() const { return f.good(); }
    std::string tok() { std::string t; if (!(f >> t)) { std::cout << "  [FAIL] golden ends early\n"; std::exit(1); } return t; }
    double num() { return std::strtod(tok().c_str(), nullptr); }
    Eigen::MatrixXd mat(const std::string& name) {
        const std::string t = tok();
        if (t != name) { std::cout << "  [FAIL] golden: expected '" << name << "', got '" << t << "'\n"; std::exit(1); }
        const int r = static_cast<int>(num()), c = static_cast<int>(num());
        Eigen::MatrixXd A(r, c);
        for (int i = 0; i < r; ++i) for (int j = 0; j < c; ++j) A(i, j) = num();
        return A;
    }
    Eigen::MatrixXcd cmat(const std::string& re, const std::string& im) { Eigen::MatrixXd a = mat(re), b = mat(im); return Eigen::MatrixXcd(a.cast<std::complex<double>>() + std::complex<double>(0, 1) * b.cast<std::complex<double>>()); }
};

// max |a-b| / max(1, max|b|)
template <class A, class B> double relErr(const A& a, const B& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return 1e9;
    double d = 0, m = 1;
    for (Eigen::Index j = 0; j < a.cols(); ++j) for (Eigen::Index i = 0; i < a.rows(); ++i) { d = std::max(d, std::abs(a(i, j) - b(i, j))); m = std::max(m, std::abs(b(i, j))); }
    return d / m;
}
inline std::string sci(double v) { std::ostringstream s; s.precision(2); s << std::scientific << v; return s.str(); }
