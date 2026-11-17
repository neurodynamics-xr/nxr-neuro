// Arrays across the boundary: NumPy <-> Eigen, and nothing else. A binding binds; the maths is in the libraries.
//
// Signals are [channels, samples], C-ordered (the library's [channels x samples]). A float32 or float64 array is accepted (other numeric types are
// converted to float64 by NumPy's rules); the library computes in float64, and the result comes back in the input's precision: float64 results are
// handed over without a copy (they may then be Fortran-ordered), float32 results are cast.
#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace nxrpy {

template <class T>
using Array1 = nb::ndarray<nb::numpy, const T, nb::ndim<1>, nb::c_contig>;
template <class T>
using Array2 = nb::ndarray<nb::numpy, const T, nb::ndim<2>, nb::c_contig>;

using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

/** A [rows x cols] C-ordered array as a column-major double matrix (one copy: the library's layout). */
template <class T>
Eigen::MatrixXd to_matrix(const Array2<T>& a) {
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> map(a.data(), static_cast<Eigen::Index>(a.shape(0)),
                                                                                          static_cast<Eigen::Index>(a.shape(1)));
  return map.template cast<double>();
}

template <class T>
Eigen::VectorXd to_vector(const Array1<T>& a) {
  Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> map(a.data(), static_cast<Eigen::Index>(a.shape(0)));
  return map.template cast<double>();
}

/** A float64 matrix handed over without a copy (the array is Fortran-ordered and owns the matrix). */
inline nb::object hand_over(Eigen::MatrixXd&& m) {
  auto* owned = new Eigen::MatrixXd(std::move(m));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<Eigen::MatrixXd*>(p); });
  const size_t shape[2] = {static_cast<size_t>(owned->rows()), static_cast<size_t>(owned->cols())};
  const int64_t strides[2] = {1, static_cast<int64_t>(owned->rows())};
  return nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<2>>(owned->data(), 2, shape, owner, strides));
}

inline nb::object hand_over(Eigen::VectorXd&& v) {
  auto* owned = new Eigen::VectorXd(std::move(v));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<Eigen::VectorXd*>(p); });
  const size_t shape[1] = {static_cast<size_t>(owned->size())};
  return nb::cast(nb::ndarray<nb::numpy, double, nb::ndim<1>>(owned->data(), 1, shape, owner));
}

/** A vector or matrix in the input's precision T: float64 without a copy, float32 cast (C-ordered). */
template <class T>
nb::object give(Eigen::MatrixXd&& m) {
  if constexpr (std::is_same_v<T, double>) {
    return hand_over(std::move(m));
  } else {
    auto* out = new T[static_cast<size_t>(m.size())];
    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(out, m.rows(), m.cols()) = m.template cast<T>();
    const size_t shape[2] = {static_cast<size_t>(m.rows()), static_cast<size_t>(m.cols())};
    return nb::cast(nb::ndarray<nb::numpy, T, nb::ndim<2>>(out, 2, shape, owner));
  }
}

template <class T>
nb::object give(Eigen::VectorXd&& v) {
  if constexpr (std::is_same_v<T, double>) {
    return hand_over(std::move(v));
  } else {
    auto* out = new T[static_cast<size_t>(v.size())];
    nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<T*>(p); });
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>(out, v.size()) = v.template cast<T>();
    const size_t shape[1] = {static_cast<size_t>(v.size())};
    return nb::cast(nb::ndarray<nb::numpy, T, nb::ndim<1>>(out, 1, shape, owner));
  }
}

/** A stack of equally sized matrices as one C-ordered [n, rows, cols] array of precision T. */
template <class T>
nb::object give_stack(const std::vector<Eigen::MatrixXd>& stack, size_t rows, size_t cols) {
  auto* out = new T[stack.size() * rows * cols];
  nb::capsule owner(out, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  for (size_t g = 0; g < stack.size(); ++g)
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(out + g * rows * cols, rows, cols) = stack[g].template cast<T>();
  const size_t shape[3] = {stack.size(), rows, cols};
  return nb::cast(nb::ndarray<nb::numpy, T, nb::ndim<3>>(out, 3, shape, owner));
}

/** A raw buffer of any element type as an owned 1-D..N-D array (used by the format readers). The shape's last index varies fastest unless `fortran`. */
template <class T>
nb::object own_array(std::vector<T>&& values, const std::vector<size_t>& shape, bool fortran = false) {
  auto* owned = new std::vector<T>(std::move(values));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  std::vector<int64_t> strides(shape.size());
  int64_t s = 1;
  if (fortran) {
    for (size_t i = 0; i < shape.size(); ++i) { strides[i] = s; s *= static_cast<int64_t>(shape[i]); }
  } else {
    for (size_t i = shape.size(); i-- > 0;) { strides[i] = s; s *= static_cast<int64_t>(shape[i]); }
  }
  return nb::cast(nb::ndarray<nb::numpy, T>(owned->data(), shape.size(), shape.data(), owner, strides.data()));
}

}  // namespace nxrpy
