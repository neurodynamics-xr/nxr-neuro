#pragma once
// Conversions shared by the first slice's commands (compute and io): mxArray <-> Eigen, strings, structs. Nothing here computes.
#include <Eigen/Dense>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "mex.h"

namespace nxr::mex::slice {

/** A real 2-D numeric array as a column-major double matrix (MATLAB's own layout); `single` says whether the input was single. */
struct Matrix {
  Eigen::MatrixXd m;
  bool single = false;
};

namespace detail {
template <class T>
void copy_as_double(const mxArray* a, Eigen::MatrixXd& out) {
  const T* p = static_cast<const T*>(mxGetData(a));
  const std::size_t n = static_cast<std::size_t>(out.size());
  double* o = out.data();
  for (std::size_t i = 0; i < n; ++i) o[i] = static_cast<double>(p[i]);
}
}  // namespace detail

inline Matrix get_matrix(const mxArray* a, const char* what) {
  if (!(mxIsNumeric(a) || mxIsLogical(a)) || mxIsComplex(a) || mxIsSparse(a) || mxGetNumberOfDimensions(a) != 2)
    throw std::invalid_argument(std::string(what) + " must be a real, dense, two-dimensional numeric array");
  Matrix r;
  r.m.resize(static_cast<Eigen::Index>(mxGetM(a)), static_cast<Eigen::Index>(mxGetN(a)));
  switch (mxGetClassID(a)) {
    case mxDOUBLE_CLASS: detail::copy_as_double<double>(a, r.m); break;
    case mxSINGLE_CLASS: detail::copy_as_double<float>(a, r.m); r.single = true; break;   // computed in double; the result goes back as single
    case mxINT8_CLASS: detail::copy_as_double<std::int8_t>(a, r.m); break;
    case mxUINT8_CLASS: detail::copy_as_double<std::uint8_t>(a, r.m); break;
    case mxINT16_CLASS: detail::copy_as_double<std::int16_t>(a, r.m); break;
    case mxUINT16_CLASS: detail::copy_as_double<std::uint16_t>(a, r.m); break;
    case mxINT32_CLASS: detail::copy_as_double<std::int32_t>(a, r.m); break;
    case mxUINT32_CLASS: detail::copy_as_double<std::uint32_t>(a, r.m); break;
    case mxINT64_CLASS: detail::copy_as_double<std::int64_t>(a, r.m); break;
    case mxUINT64_CLASS: detail::copy_as_double<std::uint64_t>(a, r.m); break;
    case mxLOGICAL_CLASS: detail::copy_as_double<mxLogical>(a, r.m); break;
    default: throw std::invalid_argument(std::string(what) + " must be single or double (or an integer or logical array, computed as double)");
  }
  return r;
}

/** Any numeric vector (row or column) as a column vector. */
inline Eigen::VectorXd get_vector(const mxArray* a, const char* what) {
  Matrix m = get_matrix(a, what);
  if (m.m.rows() != 1 && m.m.cols() != 1 && m.m.size() != 0) throw std::invalid_argument(std::string(what) + " must be a vector");
  return Eigen::Map<const Eigen::VectorXd>(m.m.data(), m.m.size());
}

inline double get_scalar(const mxArray* a, const char* what) {
  if (!(mxIsNumeric(a) || mxIsLogical(a)) || mxIsComplex(a) || mxGetNumberOfElements(a) != 1)
    throw std::invalid_argument(std::string(what) + " must be a real scalar");
  return mxGetScalar(a);
}

inline bool get_bool(const mxArray* a, const char* what) { return get_scalar(a, what) != 0.0; }

inline std::string get_string(const mxArray* a, const char* what) {
  if (!mxIsChar(a)) throw std::invalid_argument(std::string(what) + " must be a character vector or a string scalar");
  char* c = mxArrayToUTF8String(a);
  if (!c) throw std::invalid_argument(std::string(what) + " must be a character vector");
  std::string s(c);
  mxFree(c);
  return s;
}

inline bool is_unset(const mxArray* a) { return mxIsEmpty(a); }   // [] stands for "not given" (the library's std::nullopt)

/** A matrix in the input's precision. */
inline mxArray* give(const Eigen::MatrixXd& m, bool single) {
  mxArray* a = mxCreateNumericMatrix(static_cast<mwSize>(m.rows()), static_cast<mwSize>(m.cols()), single ? mxSINGLE_CLASS : mxDOUBLE_CLASS, mxREAL);
  if (single) {
    float* p = static_cast<float*>(mxGetData(a));
    for (Eigen::Index i = 0; i < m.size(); ++i) p[i] = static_cast<float>(m.data()[i]);
  } else {
    std::memcpy(mxGetData(a), m.data(), static_cast<std::size_t>(m.size()) * sizeof(double));
  }
  return a;
}

/** A vector as a row (rows = true) or a column, in the input's precision. */
inline mxArray* give_vector(const Eigen::VectorXd& v, bool single, bool row) {
  Eigen::MatrixXd m = v;
  if (row) m = v.transpose();
  return give(m, single);
}

inline mxArray* give_scalar(double x) { return mxCreateDoubleScalar(x); }
inline mxArray* give_bool(bool b) { return mxCreateLogicalScalar(b); }
inline mxArray* give_string(const std::string& s) { return mxCreateString(s.c_str()); }

inline mxArray* give_cellstr(const std::vector<std::string>& v) {
  mxArray* c = mxCreateCellMatrix(static_cast<mwSize>(v.size()), 1);
  for (std::size_t i = 0; i < v.size(); ++i) mxSetCell(c, static_cast<mwIndex>(i), give_string(v[i]));
  return c;
}

/** A 1x1 struct with the named fields, each taking ownership of its mxArray. */
inline mxArray* make_struct(const std::vector<std::pair<std::string, mxArray*>>& fields) {
  mxArray* s = mxCreateStructMatrix(1, 1, 0, nullptr);
  for (const auto& f : fields) {
    mxAddField(s, f.first.c_str());
    mxSetField(s, 0, f.first.c_str(), f.second);
  }
  return s;
}

/** A file's own name as a MATLAB field name: characters outside [A-Za-z0-9_] become _, and a name that does not start with a letter gets an x first. */
inline std::string field_name(const std::string& name) {
  std::string out;
  for (unsigned char c : name) out.push_back((std::isalnum(c) && c < 128) || c == '_' ? static_cast<char>(c) : '_');
  if (out.empty() || !std::isalpha(static_cast<unsigned char>(out[0]))) out.insert(out.begin(), 'x');
  if (out.size() > 63) out.resize(63);   // namelengthmax
  return out;
}

}  // namespace nxr::mex::slice

// The temporal events and the wavelet and filterbank functions (access_slice.cpp): each command of the dispatcher, `compute.select.*` and the new `compute.signal.*`.
// A bank arrives as a struct of the library's design fields (bindings/common/access.h); spans as a struct of columns (start and peak 1-based, stop the last sample, inclusive).
namespace nxr::mex::access {
using Command = void (*)(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void selectMedian(int, mxArray**, int, const mxArray**);
void selectMad(int, mxArray**, int, const mxArray**);
void selectCrossings(int, mxArray**, int, const mxArray**);
void selectMergeSpans(int, mxArray**, int, const mxArray**);
void selectKeepSpans(int, mxArray**, int, const mxArray**);
void selectDetectBursts(int, mxArray**, int, const mxArray**);
void selectRisefall(int, mxArray**, int, const mxArray**);
void selectStages(int, mxArray**, int, const mxArray**);
void signalBandAnalytic(int, mxArray**, int, const mxArray**);
void signalBandEnvelope(int, mxArray**, int, const mxArray**);
void signalTimeBank(int, mxArray**, int, const mxArray**);
void signalWt(int, mxArray**, int, const mxArray**);
void signalIwt(int, mxArray**, int, const mxArray**);
void signalGraphBank(int, mxArray**, int, const mxArray**);
void signalGwt(int, mxArray**, int, const mxArray**);
void signalIgwt(int, mxArray**, int, const mxArray**);
void signalJwt(int, mxArray**, int, const mxArray**);
void signalIjwt(int, mxArray**, int, const mxArray**);
}  // namespace nxr::mex::access
