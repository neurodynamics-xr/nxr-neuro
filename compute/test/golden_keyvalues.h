// A golden file of one array per line: `key v v v ...` (complex numbers as re im pairs), as written by gen_signal_design_golden.py and gen_signal_grid_golden.py.
#pragma once
#include <Eigen/Dense>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

struct KeyValues {
    std::map<std::string, std::vector<double>> v;
    explicit KeyValues(const std::string& path) {
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream s(line);
            std::string key; s >> key;
            std::vector<double> a; double x;
            while (s >> x) a.push_back(x);
            v[key] = a;
        }
    }
    bool has(const std::string& k) const { return v.count(k) > 0; }
    const std::vector<double>& at(const std::string& k) const {
        auto it = v.find(k);
        if (it == v.end()) { std::cout << "  [FAIL] golden has no '" << k << "'\n"; std::exit(1); }
        return it->second;
    }
    Eigen::VectorXd vec(const std::string& k) const { const auto& a = at(k); return Eigen::Map<const Eigen::VectorXd>(a.data(), static_cast<Eigen::Index>(a.size())); }
    Eigen::VectorXcd cvec(const std::string& k) const {
        const auto& a = at(k); Eigen::VectorXcd z(a.size() / 2);
        for (Eigen::Index i = 0; i < z.size(); ++i) z(i) = {a[2 * i], a[2 * i + 1]};
        return z;
    }
    // row-major rows x cols
    Eigen::MatrixXd mat(const std::string& k, Eigen::Index rows, Eigen::Index cols) const {
        const auto& a = at(k); Eigen::MatrixXd M(rows, cols);
        for (Eigen::Index i = 0; i < rows; ++i) for (Eigen::Index j = 0; j < cols; ++j) M(i, j) = a[i * cols + j];
        return M;
    }
    Eigen::MatrixXcd cmat(const std::string& k, Eigen::Index rows, Eigen::Index cols) const {
        const auto& a = at(k); Eigen::MatrixXcd M(rows, cols);
        for (Eigen::Index i = 0; i < rows; ++i) for (Eigen::Index j = 0; j < cols; ++j) M(i, j) = {a[2 * (i * cols + j)], a[2 * (i * cols + j) + 1]};
        return M;
    }
};
