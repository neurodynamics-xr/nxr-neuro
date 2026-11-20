/**
 * addon.cpp — N-API bindings for the cortical compute library.
 *
 * Exposes the compute library to Node.js/Electron. All functions accept TypedArrays (Float64Array, Int32Array) and return plain JavaScript objects with
 * TypedArray fields.
 *
 * The exports are the library's functions by their toolbox name, `<toolbox>.<function>` (`geometry.heat` calls `nxr::geometry::heat`): one name per thing across
 * the C++ function, the node and the bindings. Every export takes arrays and returns arrays, and nothing is kept between calls (no handle, no cache). A SURFACE
 * is data: the object made once by `geometry.surfaceArrays` (positions, faces and the intrinsic Delaunay triangulation, the expensive step), passed as the
 * first argument to every surface function, which rebuilds the library's SurfaceArrays from it. A factorization a function needs is made inside it.
 * An old export (`heat`, `solve`, `createContext`, ...) is refused with an error naming the new name (no alias).
 */

#include <napi.h>
#include "nxr/geometry/intrinsic_surface.h"
#include "nxr/geometry/surface_arrays.h"
#include "nxr/geometry/mesh.h"
#include "nxr/geometry/curvature.h"
#include "nxr/geometry/frames.h"
#include "nxr/geometry/geodesics.h"
#include "nxr/geometry/direction_fields.h"
#include "nxr/operators/trivial_connection.h"
#include "nxr/geometry/transport.h"
#include "nxr/geometry/parametrization.h"
#include "nxr/operators/factorization.h"
#include "nxr/operators/dec.h"
#include "nxr/operators/dirac.h"
#include "nxr/operators/laplacians.h"
#include "nxr/operators/eigen.h"
#include "nxr/signal/hodge_decomposition.h"
#include "nxr/operators/assembled.h"
#include "nxr/synthesis/fields.h"
#include "nxr/signal/surface_calculus.h"
#include "nxr/operators/registry/operator_registry.h"
#include "nxr/operators/registry/field_registry.h"

#include <atomic>
#include <cstring>
#include <memory>
#include <map>
#include <string>
#include <vector>

namespace geom = nxr::geometry;
namespace ops = nxr::operators;
namespace cl = nxr::operators::laplacian::connection;
namespace syn = nxr::synthesis;
using nxr::core::CancellationToken;
using nxr::core::ProgressObserver;
using nxr::operators::EigenResult;

// ─── Synchronous error translation ──────────────────────────
//
// Every synchronous N-API entry point catches nxr::core::Error
// and re-raises as a Napi::Error decorated with the structured
// fields the AsyncWorker path already exposes:
//
//   • .code  — string-named ErrorCode enumerator (e.g. "INVALID_INPUT")
//   • .hint  — diagnostic hint, when the source set one
//   • message — "[CODE] message [| hint: ...]" so plain catch-all
//                handlers still see a useful string.
//
// Without this, the C++ exception escapes Embind's default handler
// and JS sees an Error with the raw message but no .code, which
// breaks `if (err.code === 'INVALID_INPUT')` consumers who expect
// the same contract the AsyncWorker eigensolve path already honours.
//
// Use as: return nxrSyncCall(env, [&] -> Napi::Value { ... });
template <typename Fn>
static Napi::Value nxrSyncCall(Napi::Env env, Fn&& fn) {
    try {
        return fn();
    } catch (const nxr::core::Error& e) {
        std::string codeName(nxr::core::errorCodeName(e.code()));
        std::string hint(e.hint());
        std::string msg = "[" + codeName + "] " + e.what();
        if (!hint.empty()) msg += " | hint: " + hint;
        Napi::Error err = Napi::Error::New(env, msg);
        err.Set("code", Napi::String::New(env, codeName));
        if (!hint.empty()) err.Set("hint", Napi::String::New(env, hint));
        err.ThrowAsJavaScriptException();
        return env.Null();
    } catch (const std::exception& e) {
        Napi::Error err = Napi::Error::New(env, e.what());
        err.Set("code", Napi::String::New(env, "INTERNAL_ERROR"));
        err.ThrowAsJavaScriptException();
        return env.Null();
    }
}

// ─── Helpers: TypedArray ↔ Eigen conversion ──────────────────

// Convert a Float64Array to Eigen::VectorXd (copy)
static Eigen::VectorXd toVectorXd(const Napi::Float64Array& arr) {
    Eigen::VectorXd v(arr.ElementLength());
    std::memcpy(v.data(), arr.Data(), arr.ByteLength());
    return v;
}

// Convert Eigen::VectorXd to Float64Array (copy)
static Napi::Float64Array toFloat64Array(Napi::Env env, const Eigen::VectorXd& v) {
    auto arr = Napi::Float64Array::New(env, v.size());
    std::memcpy(arr.Data(), v.data(), v.size() * sizeof(double));
    return arr;
}

// Convert Eigen::MatrixXd (column-major) to Float64Array (row-major flat [rows*cols])
// We flatten row-major so JavaScript can index as [v*k + k_idx]
static Napi::Float64Array matrixToFloat64Array(Napi::Env env, const Eigen::MatrixXd& m) {
    int rows = static_cast<int>(m.rows());
    int cols = static_cast<int>(m.cols());
    auto arr = Napi::Float64Array::New(env, rows * cols);
    double* data = arr.Data();
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            data[r * cols + c] = m(r, c);
        }
    }
    return arr;
}

// Convert Eigen::MatrixXf (column-major) to Float32Array (row-major flat [rows*cols]).
// Used by time-varying generators that return [T, V] activity-shaped arrays —
// row-major flattening matches the Zarr `recordings/.../activity` schema so
// the renderer can slot the result directly into the activity store.
static Napi::Float32Array matrixToFloat32Array(Napi::Env env, const Eigen::MatrixXf& m) {
    int rows = static_cast<int>(m.rows());
    int cols = static_cast<int>(m.cols());
    auto arr = Napi::Float32Array::New(env, rows * cols);
    float* data = arr.Data();
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            data[r * cols + c] = m(r, c);
        }
    }
    return arr;
}

// Convert sparse matrix to COO TypedArrays: { row, col, data, shape }
static Napi::Object sparseToCOO(Napi::Env env, const Eigen::SparseMatrix<double>& M) {
    int nnz = static_cast<int>(M.nonZeros());

    auto rowArr = Napi::Int32Array::New(env, nnz);
    auto colArr = Napi::Int32Array::New(env, nnz);
    auto dataArr = Napi::Float64Array::New(env, nnz);

    int32_t* rowData = rowArr.Data();
    int32_t* colData = colArr.Data();
    double* valData = dataArr.Data();

    int k = 0;
    for (int outer = 0; outer < M.outerSize(); outer++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, outer); it; ++it) {
            rowData[k] = static_cast<int32_t>(it.row());
            colData[k] = static_cast<int32_t>(it.col());
            valData[k] = it.value();
            k++;
        }
    }

    auto result = Napi::Object::New(env);
    result.Set("row", rowArr);
    result.Set("col", colArr);
    result.Set("data", dataArr);
    result.Set("rows", Napi::Number::New(env, M.rows()));
    result.Set("cols", Napi::Number::New(env, M.cols()));
    result.Set("nnz", Napi::Number::New(env, nnz));
    return result;
}

// JS COO object {row:Int32Array, col:Int32Array, data:Float64Array, rows, cols}
// → Eigen sparse via setFromTriplets. Inverse of sparseToCOO. Throws (JS) on
// length mismatch or out-of-bounds indices.
static Eigen::SparseMatrix<double> cooToSparse(Napi::Env env, Napi::Object coo, const char* name) {
    if (!coo.Has("row") || !coo.Has("col") || !coo.Has("data") || !coo.Has("rows") || !coo.Has("cols"))
        throw Napi::Error::New(env, std::string(name) + ": expected {row,col,data,rows,cols}");
    const int rows = coo.Get("rows").As<Napi::Number>().Int32Value();
    const int cols = coo.Get("cols").As<Napi::Number>().Int32Value();
    auto row = coo.Get("row").As<Napi::Int32Array>();
    auto col = coo.Get("col").As<Napi::Int32Array>();
    auto data = coo.Get("data").As<Napi::Float64Array>();
    const size_t nnz = data.ElementLength();
    if (row.ElementLength() != nnz || col.ElementLength() != nnz)
        throw Napi::Error::New(env, std::string(name) + ": row/col/data length mismatch");
    std::vector<Eigen::Triplet<double>> T;
    T.reserve(nnz);
    for (size_t i = 0; i < nnz; i++) {
        const int r = row[i], c = col[i];
        if (r < 0 || r >= rows || c < 0 || c >= cols)
            throw Napi::Error::New(env, std::string(name) + ": index out of bounds");
        T.emplace_back(r, c, data[i]);
    }
    Eigen::SparseMatrix<double> M(rows, cols);
    M.setFromTriplets(T.begin(), T.end());
    return M;
}

// Convert a complex sparse matrix to COO with parallel real/imag value arrays.
// Layout matches `sparseToCOO` for compatibility with the JS-side flatten
// convention (CONTRIBUTING.md §11), but exposes complex values as two parallel
// Float64Array streams instead of one. Used for the connection-Laplacian
// "complex" output format.
static Napi::Object sparseComplexToCOO(Napi::Env env,
                                       const Eigen::SparseMatrix<std::complex<double>>& M) {
    int nnz = static_cast<int>(M.nonZeros());

    auto rowArr     = Napi::Int32Array::New(env, nnz);
    auto colArr     = Napi::Int32Array::New(env, nnz);
    auto realArr    = Napi::Float64Array::New(env, nnz);
    auto imagArr    = Napi::Float64Array::New(env, nnz);

    int32_t* rowData  = rowArr.Data();
    int32_t* colData  = colArr.Data();
    double*  realData = realArr.Data();
    double*  imagData = imagArr.Data();

    int k = 0;
    for (int outer = 0; outer < M.outerSize(); outer++) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(M, outer); it; ++it) {
            rowData[k]  = static_cast<int32_t>(it.row());
            colData[k]  = static_cast<int32_t>(it.col());
            realData[k] = it.value().real();
            imagData[k] = it.value().imag();
            k++;
        }
    }

    auto result = Napi::Object::New(env);
    result.Set("row",       rowArr);
    result.Set("col",       colArr);
    result.Set("realData",  realArr);
    result.Set("imagData",  imagArr);
    result.Set("rows",      Napi::Number::New(env, M.rows()));
    result.Set("cols",      Napi::Number::New(env, M.cols()));
    result.Set("nnz",       Napi::Number::New(env, nnz));
    return result;
}

// Convert diagonal of a sparse matrix to Float64Array (for mass matrix compactness)
static Napi::Float64Array sparseDiagonalToFloat64Array(Napi::Env env, const Eigen::SparseMatrix<double>& M) {
    int n = static_cast<int>(M.rows());
    auto arr = Napi::Float64Array::New(env, n);
    double* data = arr.Data();
    for (int i = 0; i < n; i++) data[i] = 0.0;
    for (int outer = 0; outer < M.outerSize(); outer++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, outer); it; ++it) {
            if (it.row() == it.col()) data[it.row()] = it.value();
        }
    }
    return arr;
}

// ─── The surface: geometry::SurfaceArrays ↔ a JS object ───────
//
// `geometry.surfaceArrays` makes the surface once (the intrinsic Delaunay triangulation is the expensive step) and returns a plain object; every surface
// function takes that object as its first argument and rebuilds SurfaceArrays from it, in both directions through the two functions below. Nothing is kept
// between calls: no handle, no cache.
//
//   S.positions      Float64Array [3 nV] row-major
//   S.faces          Int32Array   [3 nF] 0-based
//   S.triangulation  { options: { intrinsic, mollify, flipEps, mass, surface },
//                      record:  { mesh, mollify, epsilon, flipEps, flips, delaunay, mass, boundaryVertices, geometryCentral, fingerprint, gauge, gaugeId },
//                      mass, inputEdgeLength, edgeLength, signpost, vertexAngleSum, halfedgeVectorInVertex   Float64Array
//                      halfedgeNext, halfedgeVertex, halfedgeFace, vertexHalfedge, faceHalfedge             Float64Array of geometry-central's own 0-based ids
//                                                                                                           (-1 = none; exact integers, kept raw)
//                      boundaryLoops  number }
// The triangulation's arrays are opaque: pass the object on, do not edit it.

static const char* massKindName(geom::MassKind k) {
    switch (k) {
        case geom::MassKind::InputMixedVoronoi: return "input-mixed-voronoi";
        case geom::MassKind::Barycentric:       return "barycentric";
        case geom::MassKind::Galerkin:          return "galerkin";
    }
    return "input-mixed-voronoi";
}

static geom::MassKind parseMassKind(const std::string& s) {
    if (s == "input-mixed-voronoi") return geom::MassKind::InputMixedVoronoi;
    if (s == "barycentric")         return geom::MassKind::Barycentric;
    if (s == "galerkin")            return geom::MassKind::Galerkin;
    throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "the surface's mass is '" + s + "': use input-mixed-voronoi, barycentric or galerkin");
}

static Napi::Float64Array doublesOut(Napi::Env env, const std::vector<double>& v) {
    auto arr = Napi::Float64Array::New(env, v.size());
    if (!v.empty()) std::memcpy(arr.Data(), v.data(), v.size() * sizeof(double));
    return arr;
}

static Napi::Float64Array idsOut(Napi::Env env, const std::vector<int64_t>& v) {
    auto arr = Napi::Float64Array::New(env, v.size());
    for (size_t i = 0; i < v.size(); ++i) arr[i] = static_cast<double>(v[i]);
    return arr;
}

static Napi::Value surfaceField(const Napi::Object& o, const char* name) {
    if (!o.Has(name))
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            std::string("the surface has no field '") + name + "': make it with geometry.surfaceArrays(vertices, faces)");
    return o.Get(name);
}

static Napi::Object surfaceObjectField(const Napi::Object& o, const char* name) {
    Napi::Value v = surfaceField(o, name);
    if (!v.IsObject()) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, std::string("the surface field '") + name + "' must be an object");
    return v.As<Napi::Object>();
}

static std::vector<double> doublesField(const Napi::Object& o, const char* name) {
    Napi::Value v = surfaceField(o, name);
    if (!v.IsTypedArray() || v.As<Napi::TypedArray>().TypedArrayType() != napi_float64_array)
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, std::string("the surface field '") + name + "' must be a Float64Array");
    auto a = v.As<Napi::Float64Array>();
    return std::vector<double>(a.Data(), a.Data() + a.ElementLength());
}

static std::vector<int64_t> idsField(const Napi::Object& o, const char* name) {
    const std::vector<double> d = doublesField(o, name);
    std::vector<int64_t> out(d.size());
    for (size_t i = 0; i < d.size(); ++i) out[i] = static_cast<int64_t>(d[i]);
    return out;
}

static double numberField(const Napi::Object& o, const char* name) {
    Napi::Value v = surfaceField(o, name);
    if (v.IsBoolean()) return v.As<Napi::Boolean>().Value() ? 1.0 : 0.0;
    if (!v.IsNumber()) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, std::string("the surface field '") + name + "' must be a number");
    return v.As<Napi::Number>().DoubleValue();
}

static std::string stringField(const Napi::Object& o, const char* name) {
    Napi::Value v = surfaceField(o, name);
    if (!v.IsString()) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, std::string("the surface field '") + name + "' must be a string");
    return v.As<Napi::String>().Utf8Value();
}

static Napi::Object surfaceToObject(Napi::Env env, const geom::SurfaceArrays& s) {
    const geom::IntrinsicTriangulation& t = s.triangulation;
    auto opt = Napi::Object::New(env);
    opt.Set("intrinsic", Napi::Boolean::New(env, t.options.intrinsic));
    opt.Set("mollify",   Napi::Number::New(env, t.options.mollify));
    opt.Set("flipEps",   Napi::Number::New(env, t.options.flipEps));
    opt.Set("mass",      Napi::String::New(env, massKindName(t.options.mass)));
    opt.Set("surface",   Napi::String::New(env, t.options.surface));
    auto rec = Napi::Object::New(env);
    rec.Set("mesh",             Napi::String::New(env, t.record.mesh));
    rec.Set("mollify",          Napi::Number::New(env, t.record.mollify));
    rec.Set("epsilon",          Napi::Number::New(env, t.record.epsilon));
    rec.Set("flipEps",          Napi::Number::New(env, t.record.flipEps));
    rec.Set("flips",            Napi::Number::New(env, t.record.flips));
    rec.Set("delaunay",         Napi::Boolean::New(env, t.record.delaunay));
    rec.Set("mass",             Napi::String::New(env, t.record.mass));
    rec.Set("boundaryVertices", Napi::Number::New(env, t.record.boundaryVertices));
    rec.Set("geometryCentral",  Napi::String::New(env, t.record.geometryCentral));
    rec.Set("fingerprint",      Napi::String::New(env, t.record.fingerprint));
    rec.Set("gauge",            Napi::String::New(env, t.record.gauge));
    rec.Set("gaugeId",          Napi::String::New(env, t.record.gaugeId));
    auto tri = Napi::Object::New(env);
    tri.Set("options",                opt);
    tri.Set("record",                 rec);
    tri.Set("mass",                   doublesOut(env, t.mass));
    tri.Set("inputEdgeLength",        doublesOut(env, t.inputEdgeLength));
    tri.Set("halfedgeNext",           idsOut(env, t.halfedgeNext));
    tri.Set("halfedgeVertex",         idsOut(env, t.halfedgeVertex));
    tri.Set("halfedgeFace",           idsOut(env, t.halfedgeFace));
    tri.Set("vertexHalfedge",         idsOut(env, t.vertexHalfedge));
    tri.Set("faceHalfedge",           idsOut(env, t.faceHalfedge));
    tri.Set("boundaryLoops",          Napi::Number::New(env, static_cast<double>(t.boundaryLoops)));
    tri.Set("edgeLength",             doublesOut(env, t.edgeLength));
    tri.Set("signpost",               doublesOut(env, t.signpost));
    tri.Set("vertexAngleSum",         doublesOut(env, t.vertexAngleSum));
    tri.Set("halfedgeVectorInVertex", doublesOut(env, t.halfedgeVectorInVertex));

    auto out = Napi::Object::New(env);
    auto pos = Napi::Float64Array::New(env, static_cast<size_t>(s.positions.size()));
    for (int i = 0; i < s.positions.rows(); ++i)
        for (int j = 0; j < 3; ++j) pos[static_cast<size_t>(3 * i + j)] = s.positions(i, j);
    auto fac = Napi::Int32Array::New(env, static_cast<size_t>(s.faces.size()));
    for (int i = 0; i < s.faces.rows(); ++i)
        for (int j = 0; j < 3; ++j) fac[static_cast<size_t>(3 * i + j)] = s.faces(i, j);
    out.Set("positions",     pos);
    out.Set("faces",         fac);
    out.Set("triangulation", tri);
    return out;
}

static geom::SurfaceArrays objectToSurface(const Napi::Value& value) {
    if (!value.IsObject())
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "expected the surface object made by geometry.surfaceArrays(vertices, faces)");
    const Napi::Object o = value.As<Napi::Object>();
    geom::SurfaceArrays out;
    const std::vector<double> xyz = doublesField(o, "positions");
    Napi::Value fv = surfaceField(o, "faces");
    if (!fv.IsTypedArray() || fv.As<Napi::TypedArray>().TypedArrayType() != napi_int32_array)
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "the surface field 'faces' must be an Int32Array");
    auto fa = fv.As<Napi::Int32Array>();
    if (xyz.size() % 3 != 0 || fa.ElementLength() % 3 != 0)
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "the surface's positions and faces hold triples");
    out.positions = Eigen::Map<const geom::RowMat3>(xyz.data(), static_cast<Eigen::Index>(xyz.size() / 3), 3);
    out.faces     = Eigen::Map<const geom::FaceMat>(fa.Data(), static_cast<Eigen::Index>(fa.ElementLength() / 3), 3);

    const Napi::Object tr = surfaceObjectField(o, "triangulation");
    geom::IntrinsicTriangulation& t = out.triangulation;
    const Napi::Object opt = surfaceObjectField(tr, "options");
    t.options.intrinsic = numberField(opt, "intrinsic") != 0.0;
    t.options.mollify   = numberField(opt, "mollify");
    t.options.flipEps   = numberField(opt, "flipEps");
    t.options.mass      = parseMassKind(stringField(opt, "mass"));
    t.options.surface   = stringField(opt, "surface");
    const Napi::Object rec = surfaceObjectField(tr, "record");
    t.record.mesh             = stringField(rec, "mesh");
    t.record.mollify          = numberField(rec, "mollify");
    t.record.epsilon          = numberField(rec, "epsilon");
    t.record.flipEps          = numberField(rec, "flipEps");
    t.record.flips            = static_cast<int>(numberField(rec, "flips"));
    t.record.delaunay         = numberField(rec, "delaunay") != 0.0;
    t.record.mass             = stringField(rec, "mass");
    t.record.boundaryVertices = static_cast<int>(numberField(rec, "boundaryVertices"));
    t.record.geometryCentral  = stringField(rec, "geometryCentral");
    t.record.fingerprint      = stringField(rec, "fingerprint");
    t.record.gauge            = stringField(rec, "gauge");
    t.record.gaugeId          = stringField(rec, "gaugeId");
    t.mass                   = doublesField(tr, "mass");
    t.inputEdgeLength        = doublesField(tr, "inputEdgeLength");
    t.halfedgeNext           = idsField(tr, "halfedgeNext");
    t.halfedgeVertex         = idsField(tr, "halfedgeVertex");
    t.halfedgeFace           = idsField(tr, "halfedgeFace");
    t.vertexHalfedge         = idsField(tr, "vertexHalfedge");
    t.faceHalfedge           = idsField(tr, "faceHalfedge");
    t.boundaryLoops          = static_cast<int64_t>(numberField(tr, "boundaryLoops"));
    t.edgeLength             = doublesField(tr, "edgeLength");
    t.signpost               = doublesField(tr, "signpost");
    t.vertexAngleSum         = doublesField(tr, "vertexAngleSum");
    t.halfedgeVectorInVertex = doublesField(tr, "halfedgeVectorInVertex");
    return out;
}

// info[i] as the surface
static geom::SurfaceArrays surfaceArg(const Napi::CallbackInfo& info, size_t i = 0) {
    if (info.Length() <= i) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "expected the surface (from geometry.surfaceArrays) as the first argument");
    return objectToSurface(info[i]);
}

// ─── argument helpers ────────────────────────────────────────

static std::vector<int> intsArg(const Napi::Value& v, const char* what) {
    if (!v.IsTypedArray() || v.As<Napi::TypedArray>().TypedArrayType() != napi_int32_array)
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, std::string(what) + " must be an Int32Array");
    auto a = v.As<Napi::Int32Array>();
    return std::vector<int>(a.Data(), a.Data() + a.ElementLength());
}

static Eigen::VectorXd vectorArg(const Napi::Value& v, const char* what) {
    if (!v.IsTypedArray() || v.As<Napi::TypedArray>().TypedArrayType() != napi_float64_array)
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, std::string(what) + " must be a Float64Array");
    return toVectorXd(v.As<Napi::Float64Array>());
}

// Float64Array [rows*cols] row-major → MatrixXd
static Eigen::MatrixXd matrixArg(const Napi::Value& v, Eigen::Index rows, Eigen::Index cols, const char* what) {
    if (!v.IsTypedArray() || v.As<Napi::TypedArray>().TypedArrayType() != napi_float64_array)
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, std::string(what) + " must be a Float64Array");
    auto a = v.As<Napi::Float64Array>();
    if (static_cast<Eigen::Index>(a.ElementLength()) != rows * cols)
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, std::string(what) + " must have length " + std::to_string(rows * cols));
    Eigen::MatrixXd m(rows, cols);
    for (Eigen::Index r = 0; r < rows; ++r)
        for (Eigen::Index c = 0; c < cols; ++c) m(r, c) = a[static_cast<size_t>(r * cols + c)];
    return m;
}

static std::map<int, double> vertexValueMap(const Napi::Value& verts, const Napi::Value& values, const char* what) {
    const std::vector<int> idx = intsArg(verts, "the vertices");
    const Eigen::VectorXd val = vectorArg(values, "the values");
    if (static_cast<size_t>(val.size()) != idx.size())
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, std::string(what) + ": the values must match the vertices in length");
    std::map<int, double> out;
    for (size_t i = 0; i < idx.size(); ++i) out[idx[i]] = val[static_cast<Eigen::Index>(i)];
    return out;
}

static Napi::Value vector3Out(Napi::Env env, const Eigen::Vector3d& c) {
    auto out = Napi::Float64Array::New(env, 3);
    out[0] = c.x(); out[1] = c.y(); out[2] = c.z();
    return out;
}

static bool optionalNumber(const Napi::CallbackInfo& info, size_t i, double& out) {
    if (info.Length() > i && info[i].IsNumber()) { out = info[i].As<Napi::Number>().DoubleValue(); return true; }
    return false;
}

static Napi::Object logMapOut(Napi::Env env, const geom::transport::LogMapResult& r) {
    auto obj = Napi::Object::New(env);
    obj.Set("logCoords", matrixToFloat64Array(env, r.logCoords));
    obj.Set("sourceE1", vector3Out(env, r.sourceE1));
    obj.Set("sourceE2", vector3Out(env, r.sourceE2));
    return obj;
}

// ─── geometry.surfaceArrays(vertices, faces, options?) → surface ────
//
// options: { intrinsicDelaunay?: boolean (default true; false = the named option "embedded": the input as it is), mollify?: number (relative delta, default 1e-6),
//            flipEps?: number (default 1e-6), surface?: string (an id, recorded in the gauge identity) }. The operators then assemble on the intrinsic mollified
// Delaunay mesh (geometry-central's signpost triangulation, nxr/geometry/intrinsic_surface.h) with the mixed-Voronoi mass of the mollified input.

Napi::Value SurfaceArraysJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        if (info.Length() < 2 || !info[0].IsTypedArray() || !info[1].IsTypedArray())
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "geometry.surfaceArrays(vertices: Float64Array, faces: Int32Array, options?: { intrinsicDelaunay?: boolean, mollify?: number, flipEps?: number, surface?: string })");
        bool intrinsicDelaunay = true;
        geom::DelaunayOptions dopt;
        std::string surfaceId;
        if (info.Length() > 2 && info[2].IsObject()) {
            Napi::Object o = info[2].As<Napi::Object>();
            if (o.Has("intrinsicDelaunay")) intrinsicDelaunay = o.Get("intrinsicDelaunay").ToBoolean().Value();
            if (o.Has("mollify")) dopt.mollify = o.Get("mollify").ToNumber().DoubleValue();
            if (o.Has("flipEps")) dopt.flipEps = o.Get("flipEps").ToNumber().DoubleValue();
            if (o.Has("surface")) surfaceId = o.Get("surface").ToString().Utf8Value();
        }
        auto verticesArr = info[0].As<Napi::Float64Array>();
        auto facesArr = info[1].As<Napi::Int32Array>();
        int nV = static_cast<int>(verticesArr.ElementLength() / 3);
        int nF = static_cast<int>(facesArr.ElementLength() / 3);
        geom::IntrinsicOptions options = geom::intrinsicOptions(intrinsicDelaunay, dopt);
        options.surface = surfaceId;
        return surfaceToObject(env, geom::surfaceArrays(verticesArr.Data(), nV, facesArr.Data(), nF, options));
    });
}

// ─── operators.assembleManifoldOperators(surface, massVariant?) → { cotanLaplacian, mass, vertexDualAreas, vertexNormals, ... } ───

Napi::Value AssembleManifoldOperatorsJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        const ops::MassMatrixVariant variant = (info.Length() > 1 && info[1].IsString())
            ? ops::parseMassMatrixVariant(info[1].As<Napi::String>().Utf8Value()) : ops::MassMatrixVariant::MixedVoronoi;
        const auto o = ops::assembleManifoldOperators(surface, variant);

        auto result = Napi::Object::New(env);
        result.Set("cotanLaplacian",  sparseToCOO(env, o.cotanLaplacian));
        result.Set("mass",            sparseToCOO(env, o.mass));
        result.Set("vertexDualAreas", toFloat64Array(env, o.vertexDualAreas));
        result.Set("vertexNormals",   matrixToFloat64Array(env, o.vertexNormals));
        result.Set("totalArea",       Napi::Number::New(env, o.totalArea));
        result.Set("nV",              Napi::Number::New(env, surface.nV()));
        result.Set("nE",              Napi::Number::New(env, surface.nV() + surface.nF() - geom::eulerCharacteristic(surface)));   // χ = nV − nE + nF
        result.Set("nF",              Napi::Number::New(env, surface.nF()));
        return result;
    });
}

// ─── operators.assembleDECOperators(surface) → { d0, d1, hodge1 } ───

Napi::Value AssembleDECOperatorsJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const ops::DECOperators dec = ops::assembleDECOperators(surfaceArg(info));
        auto result = Napi::Object::New(env);
        result.Set("d0", sparseToCOO(env, dec.d0));
        result.Set("d1", sparseToCOO(env, dec.d1));
        result.Set("hodge1", sparseDiagonalToFloat64Array(env, dec.hodge1));
        return result;
    });
}

// ─── operators.assembleConnectionLaplacian(surface, options) → { K, baseDim, ... } ───
//
// Options object: { domain?, nSym?, regularization?, format? }
//   domain         : 'vertex' (default) | 'face' | 'edge'
//   nSym           : 1 (default) | 2 | 4 | …  (must be > 0)
//   regularization : number, default 1e-8
//   format         : 'real2N' (default) | 'complex'
//
// Returns:
//   {
//     K: { row, col, data,            rows, cols }   when format == 'real2N'
//        { row, col, realData, imagData, rows, cols } when format == 'complex'
//     baseDim, outputDim,
//     domain  : 'vertex' | 'face' | 'edge',
//     nSym    : number,
//     regularization : number,
//     format  : 'real2N' | 'complex'
//   }

Napi::Value AssembleConnectionLaplacianJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        cl::ConnectionLaplacianOptions opts;

        if (info.Length() > 1 && info[1].IsObject()) {
            Napi::Object o = info[1].As<Napi::Object>();
            if (o.Has("domain"))
                opts.domain = cl::parseConnectionDomain(o.Get("domain").As<Napi::String>().Utf8Value());
            if (o.Has("nSym"))
                opts.nSym = o.Get("nSym").As<Napi::Number>().Int32Value();
            if (o.Has("regularization"))
                opts.regularization = o.Get("regularization").As<Napi::Number>().DoubleValue();
            if (o.Has("format"))
                opts.format = cl::parseConnectionLaplacianFormat(o.Get("format").As<Napi::String>().Utf8Value());
        }
        const cl::ConnectionLaplacian c = cl::assembleConnectionLaplacian(surface, opts);

        const char* domainStr =
            c.domain == cl::ConnectionDomain::Vertex              ? "vertex" :
            c.domain == cl::ConnectionDomain::Face                ? "face"   :
            c.domain == cl::ConnectionDomain::EdgeCrouzeixRaviart ? "edge"   : "?";
        const char* formatStr =
            c.format == cl::ConnectionLaplacianFormat::Real2N ? "real2N" : "complex";

        auto result = Napi::Object::New(env);
        if (c.format == cl::ConnectionLaplacianFormat::Real2N) {
            result.Set("K", sparseToCOO(env, c.K_real));
        } else {
            result.Set("K", sparseComplexToCOO(env, c.K_complex));
        }
        result.Set("baseDim",        Napi::Number::New(env, c.baseDim));
        result.Set("outputDim",      Napi::Number::New(env, c.outputDim));
        result.Set("domain",         Napi::String::New(env, domainStr));
        result.Set("nSym",           Napi::Number::New(env, c.nSym));
        result.Set("regularization", Napi::Number::New(env, c.regularization));
        result.Set("format",         Napi::String::New(env, formatStr));
        return result;
    });
}

// ─── eigensolves: AsyncWorkers (these can take seconds) ──────
//
// Optional cancel and progress are SharedArrayBuffer-backed Int32Arrays. The worker thread polls / writes them while the JS main thread reads / writes from
// anywhere via Atomics.*. Layout convention (matches WASM and MEX):
//   cancelArr[0]   — non-zero ⇒ cancel requested
//   progressArr[0] — current iteration counter
//   progressArr[1] — high-water bound for iteration
//   progressArr[2] — residual × 1e6
// The JS caller keeps the arrays reachable while the returned Promise is pending (they are pinned with persistent references for the worker's lifetime).

static Napi::Object eigenResultToObject(Napi::Env env, const EigenResult& r, bool withNConverged = true) {
    auto obj = Napi::Object::New(env);
    // vMajor row-major: matches Zarr `[V, K]` schema and renderer expectations.
    obj.Set("eigenvectors", matrixToFloat64Array(env, r.eigenvectors));
    obj.Set("eigenvalues",  toFloat64Array(env, r.eigenvalues));
    obj.Set("k",            Napi::Number::New(env, r.k));
    if (withNConverged) obj.Set("nConverged", Napi::Number::New(env, r.nConverged));
    obj.Set("nV",           Napi::Number::New(env, r.eigenvectors.rows()));
    return obj;
}

// AsyncWorker for the eigensolve of a pre-assembled pencil (K, M): the caller's matrices are ingested on the JS thread (in EigenJS); this worker runs
// operators::eigen (the scale-aware shift ladder when no sigma is given, a single fixed shift otherwise).
class EigenWorker : public Napi::AsyncWorker {
public:
    EigenWorker(Napi::Env env, Eigen::SparseMatrix<double> A, Eigen::SparseMatrix<double> B,
                int k, bool hasSigma, double sigma, bool normalize, bool removeDC,
                Napi::Value cancelVal, int32_t* cancelPtr, Napi::Value progressVal, int32_t* progressPtr, std::size_t progressLen)
        : Napi::AsyncWorker(env), deferred(Napi::Promise::Deferred::New(env)),
          A_(std::move(A)), B_(std::move(B)), k_(k), hasSigma_(hasSigma),
          sigma_(sigma), normalize_(normalize), removeDC_(removeDC),
          cancelPtr_(cancelPtr), progressPtr_(progressPtr), progressLen_(progressLen) {
        // pin the SAB-backed Int32Arrays for the worker's lifetime so their backing memory stays live
        if (!cancelVal.IsUndefined() && !cancelVal.IsNull()) cancelRef_ = Napi::Persistent(cancelVal);
        if (!progressVal.IsUndefined() && !progressVal.IsNull()) progressRef_ = Napi::Persistent(progressVal);
    }

    void Execute() override {
        try {
            // std::atomic<int32_t> is layout/alignment-compatible with int32_t on every platform we ship — this reinterpret is sound in practice and the
            // canonical way to bridge a SAB cell into the C++ atomic-pointer API.
            CancellationToken cancel = cancelPtr_
                ? CancellationToken(reinterpret_cast<const std::atomic<int32_t>*>(cancelPtr_))
                : CancellationToken{};
            ProgressObserver progress;
            if (progressPtr_ && progressLen_ >= 3) {
                progress.iteration       = reinterpret_cast<std::atomic<int32_t>*>(progressPtr_ + 0);
                progress.totalIterations = reinterpret_cast<std::atomic<int32_t>*>(progressPtr_ + 1);
                progress.residualMicro   = reinterpret_cast<std::atomic<int32_t>*>(progressPtr_ + 2);
            }
            result_ = ops::eigen(A_, B_, k_, hasSigma_ ? sigma_ : ops::kShiftLadder, normalize_, removeDC_, cancel, progress);
        } catch (const nxr::core::Error& e) {
            // Preserve structured error info for OnError to attach to the JS Error.
            errorCode_ = std::string(nxr::core::errorCodeName(e.code()));
            errorHint_ = std::string(e.hint());
            SetError(e.what());
        } catch (const std::exception& e) {
            errorCode_ = "INTERNAL_ERROR";
            SetError(e.what());
        }
    }

    void OnOK() override { deferred.Resolve(eigenResultToObject(Env(), result_)); }

    void OnError(const Napi::Error& e) override {
        Napi::Env env = Env();
        // Attach structured fields to the Error so JS can switch on .code and surface .hint without parsing the message.
        Napi::Error err = Napi::Error::New(env, e.Message());
        if (!errorCode_.empty()) err.Set("code", Napi::String::New(env, errorCode_));
        if (!errorHint_.empty()) err.Set("hint", Napi::String::New(env, errorHint_));
        deferred.Reject(err.Value());
    }

    Napi::Promise GetPromise() { return deferred.Promise(); }

private:
    Napi::Promise::Deferred deferred;
    Napi::Reference<Napi::Value> cancelRef_;    // pin SAB-backed Int32Array
    Napi::Reference<Napi::Value> progressRef_;  // pin SAB-backed Int32Array
    Eigen::SparseMatrix<double> A_, B_;
    int k_;
    bool hasSigma_;
    double sigma_;
    bool normalize_, removeDC_;
    int32_t*    cancelPtr_;        // nullable; points into a SAB cell
    int32_t*    progressPtr_;      // nullable; points into a SAB cell
    std::size_t progressLen_;      // element count, for safety-bound check
    EigenResult result_;
    std::string errorCode_;
    std::string errorHint_;
};

// ─── operators.eigen(K, M, opts) → Promise<{eigenvectors,eigenvalues,k,nConverged,nV}> ───
// Solves a caller-provided symmetric pencil (K, M), K φ = λ M φ, for the k smallest eigenpairs (K, M as the COO objects `sparseToCOO` returns).
// opts = { k, sigma?, normalize? (default true), removeDC? (default false), cancel?: Int32Array, progress?: Int32Array }. sigma present → a single fixed shift;
// absent → the scale-aware shift ladder.
Napi::Value EigenJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    auto fail = [&](const std::string& code, const std::string& msg) -> Napi::Value {
        auto d = Napi::Promise::Deferred::New(env);
        Napi::Error err = Napi::Error::New(env, "[" + code + "] " + msg);
        err.Set("code", Napi::String::New(env, code));
        d.Reject(err.Value());
        return d.Promise();
    };
    try {
        if (info.Length() < 3 || !info[0].IsObject() || !info[1].IsObject() || !info[2].IsObject())
            return fail("INVALID_INPUT", "operators.eigen(K, M, opts): K, M, opts must be objects.");
        Eigen::SparseMatrix<double> A = cooToSparse(env, info[0].As<Napi::Object>(), "operators.eigen K");
        Eigen::SparseMatrix<double> B = cooToSparse(env, info[1].As<Napi::Object>(), "operators.eigen M");
        Napi::Object opts = info[2].As<Napi::Object>();
        if (A.rows() != A.cols() || B.rows() != B.cols() || A.rows() != B.rows())
            return fail("INVALID_INPUT", "operators.eigen: K and M must be square and the same size.");
        if (!opts.Has("k")) return fail("INVALID_INPUT", "operators.eigen: k is required.");
        const int n = static_cast<int>(A.rows());
        const int k = opts.Get("k").As<Napi::Number>().Int32Value();
        if (k < 1 || k > n - 1)
            return fail("INVALID_INPUT", "operators.eigen: k must be in [1, N-1].");
        const bool hasSigma  = opts.Has("sigma");
        const double sigma   = hasSigma ? opts.Get("sigma").As<Napi::Number>().DoubleValue() : 0.0;
        const bool normalize = opts.Has("normalize") ? opts.Get("normalize").As<Napi::Boolean>().Value() : true;
        const bool removeDC  = opts.Has("removeDC") ? opts.Get("removeDC").As<Napi::Boolean>().Value() : false;

        // Optional cancel + progress arrays. The caller is expected to back them with a SharedArrayBuffer so flips are visible cross-thread.
        int32_t*    cancelPtr   = nullptr;
        int32_t*    progressPtr = nullptr;
        std::size_t progressLen = 0;
        Napi::Value cancelVal   = env.Undefined();
        Napi::Value progressVal = env.Undefined();
        if (opts.Has("cancel") && opts.Get("cancel").IsTypedArray()) {
            auto arr = opts.Get("cancel").As<Napi::Int32Array>();
            if (arr.ElementLength() >= 1) { cancelPtr = arr.Data(); cancelVal = opts.Get("cancel"); }
        }
        if (opts.Has("progress") && opts.Get("progress").IsTypedArray()) {
            auto arr = opts.Get("progress").As<Napi::Int32Array>();
            progressPtr = arr.Data();
            progressLen = arr.ElementLength();
            progressVal = opts.Get("progress");
        }

        auto* worker = new EigenWorker(env, std::move(A), std::move(B), k, hasSigma, sigma, normalize, removeDC,
                                       cancelVal, cancelPtr, progressVal, progressPtr, progressLen);
        auto promise = worker->GetPromise();
        worker->Queue();
        return promise;
    } catch (const Napi::Error& e) {
        return fail("INVALID_INPUT", e.Message());
    } catch (const std::exception& e) {
        return fail("INTERNAL_ERROR", e.what());
    }
}

// AsyncWorker for the named-operator eigensolve. opts are parsed into an EigenOperatorSpec on the JS thread (in EigenOperatorJS); the surface is rebuilt from
// its arrays there too, and moved into the worker, which only runs the off-thread solve and marshals the result.
class EigenOperatorWorker : public Napi::AsyncWorker {
public:
    EigenOperatorWorker(Napi::Env env, geom::SurfaceArrays surface, ops::EigenOperatorSpec spec,
                        int k, double sigma, bool normalize, bool multiplets, bool dense, int blockSize)
        : Napi::AsyncWorker(env), deferred(Napi::Promise::Deferred::New(env)),
          surface_(std::move(surface)), spec_(spec), k_(k), sigma_(sigma),
          normalize_(normalize), multiplets_(multiplets), dense_(dense),
          blockSize_(blockSize) {}

    void Execute() override {
        try {
            result_ = ops::eigenOperator(surface_, spec_, k_, sigma_, normalize_, multiplets_, dense_);
        } catch (const nxr::core::Error& e) {
            errorCode_ = std::string(nxr::core::errorCodeName(e.code()));
            errorHint_ = std::string(e.hint());
            SetError(e.what());
        } catch (const std::exception& e) {
            errorCode_ = "INTERNAL_ERROR";
            SetError(e.what());
        }
    }

    void OnOK() override {
        Napi::Env env = Env();
        auto obj = Napi::Object::New(env);
        obj.Set("eigenvectors", matrixToFloat64Array(env, result_.eigenvectors));
        obj.Set("eigenvalues",  toFloat64Array(env, result_.eigenvalues));
        obj.Set("k",            Napi::Number::New(env, result_.k));
        obj.Set("nConverged",   Napi::Number::New(env, result_.nConverged));
        obj.Set("blockSize",    Napi::Number::New(env, blockSize_));
        if (result_.kind.size() > 0) {   // Hodge–Dirac: 0 exact · 1 harmonic · 2 coexact, per mode
            auto kind = Napi::Int32Array::New(env, result_.kind.size());
            for (Eigen::Index i = 0; i < result_.kind.size(); ++i) kind[static_cast<size_t>(i)] = result_.kind(i);
            obj.Set("kind", kind);
        }
        deferred.Resolve(obj);
    }

    void OnError(const Napi::Error& e) override {
        Napi::Env env = Env();
        Napi::Error err = Napi::Error::New(env, e.Message());
        if (!errorCode_.empty()) err.Set("code", Napi::String::New(env, errorCode_));
        if (!errorHint_.empty()) err.Set("hint", Napi::String::New(env, errorHint_));
        deferred.Reject(err.Value());
    }

    Napi::Promise GetPromise() { return deferred.Promise(); }

private:
    Napi::Promise::Deferred deferred;
    geom::SurfaceArrays surface_;
    ops::EigenOperatorSpec spec_;
    int k_; double sigma_; bool normalize_, multiplets_, dense_;
    int blockSize_;
    EigenResult result_;
    std::string errorCode_, errorHint_;
};

// ─── operators.eigenOperator(surface, opts) → Promise<{eigenvectors,eigenvalues,k,nConverged,blockSize}> ───
// Async. Mirrors the wasm function's opts parsing; the operator AND its natural generalized mass are assembled C++-side.
Napi::Value EigenOperatorJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    // Reject (not throw) on validation error so the caller's await/.catch is uniform.
    auto fail = [&](const std::string& codeName, const std::string& msg, const std::string& hint) -> Napi::Value {
        auto d = Napi::Promise::Deferred::New(env);
        Napi::Error err = Napi::Error::New(env, "[" + codeName + "] " + msg);
        err.Set("code", Napi::String::New(env, codeName));
        if (!hint.empty()) err.Set("hint", Napi::String::New(env, hint));
        d.Reject(err.Value());
        return d.Promise();
    };
    try {
        if (info.Length() < 2 || !info[1].IsObject())
            return fail("INVALID_INPUT", "operators.eigenOperator(surface, opts): opts must be an object.", "");
        Napi::Object opts = info[1].As<Napi::Object>();
        if (!opts.Has("operator"))
            return fail("INVALID_INPUT", "operators.eigenOperator: expected { operator, k, ... }.", "");

        ops::EigenOperatorSpec spec;
        int blockSize = 1;
        const std::string op = opts.Get("operator").As<Napi::String>().Utf8Value();
        if (op == "laplacian") {
            const std::string sub = opts.Has("subtype") ? opts.Get("subtype").As<Napi::String>().Utf8Value() : "";
            if      (sub == "cotan") spec.op = ops::EigenOperator::LaplacianCotan;
            else if (sub == "graph") spec.op = ops::EigenOperator::LaplacianGraph;
            else return fail("INVALID_INPUT", "operators.eigenOperator laplacian: subtype must be cotan|graph.", "");
        } else if (op == "cotan") {
            spec.op = ops::EigenOperator::LaplacianCotan;
        } else if (op == "graph") {
            spec.op = ops::EigenOperator::LaplacianGraph;
        } else if (op == "dirac") {
            spec.op = ops::EigenOperator::Dirac; blockSize = 4;
        } else if (op == "diracFace") {
            spec.op = ops::EigenOperator::DiracFace; blockSize = 4;
        } else if (op == "hodgeDirac") {
            spec.op = ops::EigenOperator::HodgeDirac;   // N = V+E+F; default sigma = k lowest +mu by the exact/coexact split
        } else if (op == "spinExtrinsic") {
            spec.op = ops::EigenOperator::SpinExtrinsic; blockSize = 4;
        } else if (op == "spinIntrinsicHY") {
            spec.op = ops::EigenOperator::SpinIntrinsicHY; blockSize = 4;
        } else if (op == "spinIntrinsicYe") {
            spec.op = ops::EigenOperator::SpinIntrinsicYe; blockSize = 4;
        } else {
            return fail("INVALID_INPUT", "operators.eigenOperator: operator must be laplacian|cotan|graph|dirac|diracFace|hodgeDirac|spinExtrinsic|spinIntrinsicHY|spinIntrinsicYe.", "");
        }
        if (opts.Has("graphMass")) spec.graphDegreeMass = opts.Get("graphMass").As<Napi::String>().Utf8Value() == "degree";
        if (opts.Has("star1Floor")) spec.star1Floor = opts.Get("star1Floor").As<Napi::Number>().DoubleValue();
        if (opts.Has("mollify"))    spec.mollify    = opts.Get("mollify").As<Napi::Number>().DoubleValue();
        if (opts.Has("bothSigns"))  spec.bothSigns  = opts.Get("bothSigns").As<Napi::Boolean>().Value();

        if (opts.Has("tau"))  spec.tau  = opts.Get("tau").As<Napi::Number>().DoubleValue();
        if (opts.Has("mass")) spec.mass = ops::parseMassMatrixVariant(opts.Get("mass").As<Napi::String>().Utf8Value());
        if (!opts.Has("k")) return fail("INVALID_INPUT", "operators.eigenOperator: k is required.", "");

        const int    k          = opts.Get("k").As<Napi::Number>().Int32Value();
        const double sigma      = opts.Has("sigma")      ? opts.Get("sigma").As<Napi::Number>().DoubleValue()     : ops::kShiftLadder;
        const bool   normalize  = opts.Has("normalize")  ? opts.Get("normalize").As<Napi::Boolean>().Value()      : true;
        const bool   multiplets = opts.Has("multiplets") ? opts.Get("multiplets").As<Napi::Boolean>().Value()     : false;
        const bool   dense      = opts.Has("dense")      ? opts.Get("dense").As<Napi::Boolean>().Value()          : false;

        auto* worker = new EigenOperatorWorker(env, surfaceArg(info), spec, k, sigma, normalize, multiplets, dense, blockSize);
        worker->Queue();
        return worker->GetPromise();
    } catch (const nxr::core::Error& e) {
        return fail(std::string(nxr::core::errorCodeName(e.code())), e.what(), std::string(e.hint()));
    } catch (const std::exception& e) {
        return fail("INTERNAL_ERROR", e.what(), "");
    }
}

// ─── operators.poisson(K, M, densityVertices, densityValues) ───
// K, M: the COO objects of the Laplacian and the mass. The Laplacian's factor is made inside, from K.

Napi::Value PoissonJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        if (info.Length() < 4 || !info[0].IsObject() || !info[1].IsObject())
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "operators.poisson(K, M, densityVertices: Int32Array, densityValues: Float64Array)");
        Eigen::SparseMatrix<double> K = cooToSparse(env, info[0].As<Napi::Object>(), "operators.poisson K");
        Eigen::SparseMatrix<double> M = cooToSparse(env, info[1].As<Napi::Object>(), "operators.poisson M");
        const std::map<int, double> densityMap = vertexValueMap(info[2], info[3], "operators.poisson");
        const std::unique_ptr<ops::CholeskyFactor> factor = ops::laplacianFactor(K);
        return toFloat64Array(env, ops::poisson(K, M, *factor, densityMap));
    });
}

// ─── geometry.heat(surface, sources, options?) ────────────
// sources: an Int32Array of vertices → a Float64Array [nV]; an Array of Int32Arrays (the batch form, one factorization for all) → an Array of Float64Array [nV].
// options: { tCoef?: number (default 1), mesh?: 'intrinsic' (default) | 'input' }.

Napi::Value HeatJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        double tCoef = 1.0;
        geom::HeatMesh mesh = geom::HeatMesh::Intrinsic;
        if (info.Length() > 2 && info[2].IsObject()) {
            Napi::Object o = info[2].As<Napi::Object>();
            if (o.Has("tCoef")) tCoef = o.Get("tCoef").As<Napi::Number>().DoubleValue();
            if (o.Has("mesh")) {
                const std::string m = o.Get("mesh").As<Napi::String>().Utf8Value();
                if (m == "input") mesh = geom::HeatMesh::Input;
                else if (m != "intrinsic") throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "heat: mesh must be 'intrinsic' or 'input'");
            }
        }
        if (info.Length() > 1 && info[1].IsArray()) {
            Napi::Array sets = info[1].As<Napi::Array>();
            std::vector<std::vector<int>> sourceSets;
            for (uint32_t i = 0; i < sets.Length(); ++i) sourceSets.push_back(intsArg(sets.Get(i), "each source set"));
            const Eigen::MatrixXd D = geom::heat(surface, sourceSets, tCoef, mesh);   // column j = the distances from set j
            auto out = Napi::Array::New(env, static_cast<size_t>(D.cols()));
            for (Eigen::Index j = 0; j < D.cols(); ++j) out.Set(static_cast<uint32_t>(j), toFloat64Array(env, D.col(j)));
            return out;
        }
        return toFloat64Array(env, geom::heat(surface, intsArg(info[1], "the sources"), tCoef, mesh));
    });
}

// ─── signal.hodge(surface, omega?) — returns all scalars + vectors ─
// omega: a 1-form on the OPERATOR (intrinsic triangulation) mesh's edges (a random one when omitted). The DEC and the two factorizations are made inside.

Napi::Value HodgeJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        const ops::DECOperators dec = ops::assembleDECOperators(surface);
        const std::unique_ptr<ops::CholeskyFactor> exact = ops::exactFormFactor(dec);
        const std::unique_ptr<ops::SquareSolver> coexact = ops::coexactFormSolver(dec);

        // Use provided omega or generate random
        Eigen::VectorXd omega;
        if (info.Length() > 1 && info[1].IsTypedArray()) {
            omega = toVectorXd(info[1].As<Napi::Float64Array>());
        } else {
            omega = syn::randomOmega(static_cast<int>(dec.d0.rows()));
        }

        nxr::signal::HodgeResult result = nxr::signal::hodge(surface, dec, *exact, *coexact, omega);

        auto obj = Napi::Object::New(env);
        // Scalar potentials
        obj.Set("alpha", toFloat64Array(env, result.exactPotential));
        obj.Set("beta", toFloat64Array(env, result.coExactPotentialV));
        obj.Set("combined", toFloat64Array(env, result.combinedPotential));
        // Input 1-form and its components
        obj.Set("omega", toFloat64Array(env, result.omega));
        obj.Set("dAlpha", toFloat64Array(env, result.dAlpha));
        obj.Set("deltaBeta", toFloat64Array(env, result.deltaBeta));
        obj.Set("gamma", toFloat64Array(env, result.gamma));
        // Face-centered vector fields [nF * 3]
        obj.Set("omegaVectors", matrixToFloat64Array(env, result.omegaVectors));
        obj.Set("dAlphaVectors", matrixToFloat64Array(env, result.dAlphaVectors));
        obj.Set("deltaBetaVectors", matrixToFloat64Array(env, result.deltaBetaVectors));
        obj.Set("gammaVectors", matrixToFloat64Array(env, result.gammaVectors));
        return obj;
    });
}

// ─── geometry.curvatures(surface) → { gaussian, mean, kMin, kMax, principalDir } ──

Napi::Value CurvaturesJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        geom::CurvatureResult result = geom::curvatures(surfaceArg(info));

        auto obj = Napi::Object::New(env);
        obj.Set("gaussian", toFloat64Array(env, result.gaussian));
        obj.Set("mean", toFloat64Array(env, result.mean));
        obj.Set("kMin", toFloat64Array(env, result.kMin));
        obj.Set("kMax", toFloat64Array(env, result.kMax));
        obj.Set("principalDir", matrixToFloat64Array(env, result.principalDirMax));
        return obj;
    });
}

// ─── geometry.normals(surface, type) → Float64Array [nV*3] ──

Napi::Value NormalsJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        int typeInt = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 0;
        geom::NormalType type = static_cast<geom::NormalType>(typeInt);
        return matrixToFloat64Array(env, geom::normals(surface, type));
    });
}

// ─── signal.whitney(surface, oneForm) → Float64Array [nF*3] ──
// oneForm: a 1-form on the OPERATOR (intrinsic triangulation) mesh's edges.

Napi::Value WhitneyJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        Eigen::VectorXd oneForm = vectorArg(info[1], "oneForm");
        return matrixToFloat64Array(env, nxr::signal::whitney(surface, ops::assembleDECOperators(surface), oneForm));
    });
}

// ─── operators.gradient(surface, vertexScalars) → Float64Array [nF*3] ──

Napi::Value GradientJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        return matrixToFloat64Array(env, ops::gradient(surface, vectorArg(info[1], "scalars")));
    });
}

// ─── signal.isoline(surface, scalars, numLevels, min, max) → { positions, count } ──

Napi::Value IsolineJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        Eigen::VectorXd scalars = vectorArg(info[1], "scalars");

        int numLevels = info.Length() > 2 ? info[2].As<Napi::Number>().Int32Value() : 20;
        double minVal = info.Length() > 3 ? info[3].As<Napi::Number>().DoubleValue() : 0.0;
        double maxVal = info.Length() > 4 ? info[4].As<Napi::Number>().DoubleValue() : 0.0;

        nxr::signal::IsolineResult result = nxr::signal::isoline(surface, scalars, numLevels, minVal, maxVal);

        auto obj = Napi::Object::New(env);
        obj.Set("positions", matrixToFloat64Array(env, result.positions));
        obj.Set("segmentCount", Napi::Number::New(env, result.segmentCount));
        return obj;
    });
}

// ─── operators.trivial(surface, singVerts, singValues) → { ... } ──

Napi::Value TrivialJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        const std::map<int, double> singMap = vertexValueMap(info[1], info[2], "operators.trivial");
        const ops::DECOperators dec = ops::assembleDECOperators(surface);
        const std::unique_ptr<ops::CholeskyFactor> exact = ops::exactFormFactor(dec);

        ops::connection::DirectionFieldResult result = ops::connection::trivial(surface, dec, *exact, singMap);

        auto obj = Napi::Object::New(env);
        obj.Set("connections", toFloat64Array(env, result.connections));
        obj.Set("directionVectors", matrixToFloat64Array(env, result.directionVectors));
        obj.Set("orthogonalVectors", matrixToFloat64Array(env, result.orthogonalVectors));
        obj.Set("eulerCharacteristic", Napi::Number::New(env, result.eulerCharacteristic));
        obj.Set("gaussBonnetSatisfied", Napi::Boolean::New(env, result.gaussBonnetSatisfied));
        return obj;
    });
}

// ─── signal.streamline(surface, faceField, numSeeds, stepCoef, maxSteps) → { positions, count } ──

Napi::Value StreamlineJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        Eigen::MatrixXd faceField = matrixArg(info[1], surface.nF(), 3, "faceField");

        int numSeeds = info.Length() > 2 ? info[2].As<Napi::Number>().Int32Value() : 15;
        double stepCoef = info.Length() > 3 ? info[3].As<Napi::Number>().DoubleValue() : 0.15;
        int maxSteps = info.Length() > 4 ? info[4].As<Napi::Number>().Int32Value() : 1000;

        nxr::signal::StreamlineResult result = nxr::signal::streamline(surface, faceField, numSeeds, stepCoef, maxSteps);

        auto obj = Napi::Object::New(env);
        obj.Set("positions", matrixToFloat64Array(env, result.positions));
        obj.Set("segmentCount", Napi::Number::New(env, result.segmentCount));
        return obj;
    });
}

// ─── synthesis.delta(nV, sourceVerts, sourceValues) → Float64Array [nV]: zero except at the sources ──

Napi::Value DeltaJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        if (info.Length() < 3 || !info[0].IsNumber())
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "synthesis.delta(nV: number, sourceVerts: Int32Array, sourceValues: Float64Array)");
        return toFloat64Array(env, syn::delta(info[0].As<Napi::Number>().Int32Value(), vertexValueMap(info[1], info[2], "synthesis.delta")));
    });
}

// An EigenResult from the object a solve returned: { eigenvectors: Float64Array [nV*k] row-major, eigenvalues: Float64Array [k] }.
static EigenResult eigenArg(const Napi::Value& v, const char* what) {
    if (!v.IsObject()) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, std::string(what) + ": expected { eigenvectors, eigenvalues } from an eigensolve");
    const Napi::Object o = v.As<Napi::Object>();
    if (!o.Has("eigenvectors") || !o.Has("eigenvalues"))
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, std::string(what) + ": expected { eigenvectors, eigenvalues } from an eigensolve");
    EigenResult r;
    r.eigenvalues = vectorArg(o.Get("eigenvalues"), "eigenvalues");
    const Eigen::Index k = r.eigenvalues.size();
    const Eigen::VectorXd flat = vectorArg(o.Get("eigenvectors"), "eigenvectors");
    if (k == 0 || flat.size() % k != 0) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, std::string(what) + ": eigenvectors must hold nV*k values");
    const Eigen::Index nV = flat.size() / k;
    r.eigenvectors.resize(nV, k);
    for (Eigen::Index i = 0; i < nV; ++i)
        for (Eigen::Index j = 0; j < k; ++j) r.eigenvectors(i, j) = flat[i * k + j];   // vMajor row-major in, column-major out
    r.k = static_cast<int>(k);
    r.nConverged = r.k;
    return r;
}

// ─── synthesis.heatDiffusion(M, eig, u0, timesteps, alpha?) ────
//
// Returns a [T*nV] Float32Array (row-major frame-major) plus T and nV. M: the mass as the COO object of `operators.assembleManifoldOperators`; eig: the result of an
// eigensolve (M-orthonormal modes); u0: the initial condition [nV] (`synthesis.delta`).

Napi::Value HeatDiffusionJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        if (info.Length() < 4 || !info[0].IsObject())
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "synthesis.heatDiffusion(M, eig, u0: Float64Array, timesteps: Float64Array, alpha?: number)");
        Eigen::SparseMatrix<double> M = cooToSparse(env, info[0].As<Napi::Object>(), "synthesis.heatDiffusion M");
        EigenResult eig = eigenArg(info[1], "synthesis.heatDiffusion");
        Eigen::VectorXd u0 = vectorArg(info[2], "u0");
        Eigen::VectorXd ts = vectorArg(info[3], "timesteps");
        const double alpha = info.Length() > 4 && info[4].IsNumber() ? info[4].As<Napi::Number>().DoubleValue() : 1.0;
        std::vector<double> timesteps(ts.data(), ts.data() + ts.size());

        Eigen::MatrixXf field = syn::heatDiffusion(M, eig, u0, timesteps, alpha);

        auto result = Napi::Object::New(env);
        result.Set("data", matrixToFloat32Array(env, field));
        result.Set("T", Napi::Number::New(env, static_cast<int>(field.rows())));
        result.Set("nV", Napi::Number::New(env, static_cast<int>(field.cols())));
        return result;
    });
}

// ─── synthesis.dampedWave(eig, {modeIndices, amplitudes, dampings, phases, timesteps}) ─

Napi::Value DampedWaveJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        EigenResult eig = eigenArg(info[0], "synthesis.dampedWave");
        auto opts = info[1].As<Napi::Object>();
        auto modeIdxArr  = opts.Get("modeIndices").As<Napi::Int32Array>();
        auto ampsArr     = opts.Get("amplitudes").As<Napi::Float64Array>();
        auto dampsArr    = opts.Get("dampings").As<Napi::Float64Array>();
        auto phasesArr   = opts.Get("phases").As<Napi::Float64Array>();
        auto tsArr       = opts.Get("timesteps").As<Napi::Float64Array>();

        std::vector<int> modeIndices(modeIdxArr.Data(), modeIdxArr.Data() + modeIdxArr.ElementLength());
        std::vector<double> amplitudes(ampsArr.Data(), ampsArr.Data() + ampsArr.ElementLength());
        std::vector<double> dampings(dampsArr.Data(), dampsArr.Data() + dampsArr.ElementLength());
        std::vector<double> phases(phasesArr.Data(), phasesArr.Data() + phasesArr.ElementLength());
        std::vector<double> timesteps(tsArr.Data(), tsArr.Data() + tsArr.ElementLength());

        Eigen::MatrixXf field = syn::dampedWave(eig, modeIndices, amplitudes, dampings, phases, timesteps);

        auto result = Napi::Object::New(env);
        result.Set("data", matrixToFloat32Array(env, field));
        result.Set("T", Napi::Number::New(env, static_cast<int>(field.rows())));
        result.Set("nV", Napi::Number::New(env, static_cast<int>(field.cols())));
        return result;
    });
}

// ─── Vector heat method ──────────────────────────────────────
//
// Each call builds the solver on the surface and drops it on return; the batch forms (an Array of source sets) share one solver and one factorization.

// geometry.parallel(surface, sourceVerts, sourceVectors, tCoef?): sourceVerts an Int32Array and sourceVectors a Float64Array [3n] (one set → Float64Array [nV*3]),
// or Arrays of them (the batch form → an Array of Float64Array [nV*3]).
Napi::Value ParallelJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        double tCoef = 1.0;
        optionalNumber(info, 3, tCoef);
        if (info.Length() > 2 && info[1].IsArray() != info[2].IsArray())
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "sourceVerts and sourceVectors are both Arrays (the batch form) or both typed arrays");
        if (info.Length() > 2 && info[1].IsArray()) {
            Napi::Array vs = info[1].As<Napi::Array>(), xs = info[2].As<Napi::Array>();
            if (vs.Length() != xs.Length()) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "sourceVectors holds one array per set of sourceVerts");
            std::vector<std::vector<int>> sets;
            std::vector<Eigen::MatrixXd> vecs;
            for (uint32_t i = 0; i < vs.Length(); ++i) {
                sets.push_back(intsArg(vs.Get(i), "each set of sourceVerts"));
                vecs.push_back(matrixArg(xs.Get(i), static_cast<Eigen::Index>(sets.back().size()), 3, "each set of sourceVectors"));
            }
            const auto out = geom::transport::parallel(surface, sets, vecs, tCoef);
            auto arr = Napi::Array::New(env, out.size());
            for (size_t i = 0; i < out.size(); ++i) arr.Set(static_cast<uint32_t>(i), matrixToFloat64Array(env, out[i]));
            return arr;
        }
        std::vector<int> verts = intsArg(info[1], "sourceVerts");
        Eigen::MatrixXd S = matrixArg(info[2], static_cast<Eigen::Index>(verts.size()), 3, "sourceVectors (Nx3, length 3*sources)");
        return matrixToFloat64Array(env, geom::transport::parallel(surface, verts, S, tCoef));
    });
}

Napi::Value ExtendScalarJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        std::vector<int> verts = intsArg(info[1], "sourceVerts");
        Eigen::VectorXd vals = vectorArg(info[2], "sourceValues");
        if (static_cast<size_t>(vals.size()) != verts.size())
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, "sourceVerts and sourceValues must have the same length");
        double tCoef = 1.0;
        optionalNumber(info, 3, tCoef);
        return toFloat64Array(env, geom::transport::extendScalar(surface, verts, vals, tCoef));
    });
}

// The flip-out geodesic (Sharp & Crane 2020) between two vertices, with
// each point's place on the mesh — parity with the WASM `tracePath`:
// { positions [N×3], nPoints, vertices Int32 [N×3], weights [N×3], length }.
Napi::Value TracePathJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        int a = info[1].As<Napi::Number>().Int32Value();
        int b = info[2].As<Napi::Number>().Int32Value();
        geom::query::GeodesicPath path = geom::query::tracePathSurface(surface, a, b);
        auto obj = Napi::Object::New(env);
        obj.Set("positions", matrixToFloat64Array(env, path.positions));
        obj.Set("nPoints",   Napi::Number::New(env, static_cast<double>(path.positions.rows())));
        auto verts = Napi::Int32Array::New(env, path.vertices.rows() * 3);
        for (int i = 0; i < path.vertices.rows(); i++)
            for (int k = 0; k < 3; k++) verts[i * 3 + k] = path.vertices(i, k);
        obj.Set("vertices",  verts);
        obj.Set("weights",   matrixToFloat64Array(env, path.weights));
        obj.Set("length",    Napi::Number::New(env, path.length));
        return obj;
    });
}

// The exact (MMP) geodesic distance between two vertices, mesh units —
// parity with the WASM `exactDistance`.
Napi::Value ExactDistanceJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        int a = info[1].As<Napi::Number>().Int32Value();
        int b = info[2].As<Napi::Number>().Int32Value();
        return Napi::Number::New(env, geom::query::exactDistance(surface, a, b));
    });
}

// geometry.logMap(surface, sourceVertex, strategy?, tCoef?): a number → one { logCoords, sourceE1, sourceE2 }; an Int32Array of sources (the batch form, one solver) → an Array of them.
Napi::Value LogMapJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        double strategyD = static_cast<double>(geom::transport::LogMapStrategy::AffineLocal);
        optionalNumber(info, 2, strategyD);
        auto strategy = static_cast<geom::transport::LogMapStrategy>(static_cast<int>(strategyD));
        double tCoef = 1.0;
        optionalNumber(info, 3, tCoef);
        if (info.Length() > 1 && info[1].IsTypedArray()) {
            const auto results = geom::transport::logMap(surface, intsArg(info[1], "sourceVertices"), strategy, tCoef);
            auto arr = Napi::Array::New(env, results.size());
            for (size_t i = 0; i < results.size(); ++i) arr.Set(static_cast<uint32_t>(i), logMapOut(env, results[i]));
            return arr;
        }
        return logMapOut(env, geom::transport::logMap(surface, info[1].As<Napi::Number>().Int32Value(), strategy, tCoef));
    });
}

Napi::Value FindCenterJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        std::vector<int> verts = intsArg(info[1], "sourceVerts");
        int p = info.Length() > 2 && info[2].IsNumber() ? info[2].As<Napi::Number>().Int32Value() : 2;
        double tCoef = 1.0;
        optionalNumber(info, 3, tCoef);
        return vector3Out(env, geom::transport::findCenter(surface, verts, p, tCoef));
    });
}

// ─── geometry.signedHeat(surface, curve, isLoop?, levelSet?, tCoef?) ─────────────────────
// curve: an Int32Array of vertices → a Float64Array [nV]; an Array of Int32Arrays (the batch form) → an Array of Float64Array [nV].

Napi::Value SignedHeatJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        bool isLoop = info.Length() > 2 && info[2].IsBoolean() ? info[2].As<Napi::Boolean>().Value() : true;
        int  lsInt  = info.Length() > 3 && info[3].IsNumber()
            ? info[3].As<Napi::Number>().Int32Value()
            : static_cast<int>(geom::SignedHeatLevelSet::ZeroSet);
        auto ls = static_cast<geom::SignedHeatLevelSet>(lsInt);
        double tCoef = 1.0;
        optionalNumber(info, 4, tCoef);
        if (info.Length() > 1 && info[1].IsArray()) {
            Napi::Array cs = info[1].As<Napi::Array>();
            std::vector<std::vector<int>> curves;
            for (uint32_t i = 0; i < cs.Length(); ++i) curves.push_back(intsArg(cs.Get(i), "each curve"));
            const Eigen::MatrixXd D = geom::signedHeat(surface, curves, isLoop, ls, tCoef);
            auto out = Napi::Array::New(env, static_cast<size_t>(D.cols()));
            for (Eigen::Index j = 0; j < D.cols(); ++j) out.Set(static_cast<uint32_t>(j), toFloat64Array(env, D.col(j)));
            return out;
        }
        return toFloat64Array(env, geom::signedHeat(surface, intsArg(info[1], "the curve"), isLoop, ls, tCoef));
    });
}

// ─── Smooth direction fields ────────────────────────────────

Napi::Value SmoothFaceJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        int  nSym             = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 4;
        bool alignToCurvature = info.Length() > 2 ? info[2].As<Napi::Boolean>().Value()     : false;
        return matrixToFloat64Array(env, geom::connection::smoothFace(surface, nSym, alignToCurvature));
    });
}

Napi::Value SmoothVertexJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        const geom::SurfaceArrays surface = surfaceArg(info);
        int  nSym             = info.Length() > 1 ? info[1].As<Napi::Number>().Int32Value() : 2;
        bool alignToCurvature = info.Length() > 2 ? info[2].As<Napi::Boolean>().Value()     : false;

        geom::connection::SmoothVertexFieldResult r = geom::connection::smoothVertex(surface, nSym, alignToCurvature);
        auto obj = Napi::Object::New(env);
        obj.Set("vertexVectors",  matrixToFloat64Array(env, r.vertexVectors));
        obj.Set("vertexFieldRaw", toFloat64Array(env, r.vertexFieldRaw));
        obj.Set("nSym", Napi::Number::New(env, r.nSym));
        return obj;
    });
}



// ─── geometry.frames(surface) → { e1, e2, normals } (per-face tangent frames) ───
// Each field is a row-major [nF×3] Float64Array.
Napi::Value FramesJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        geom::FaceFrames f = geom::frames(surfaceArg(info));
        auto obj = Napi::Object::New(env);
        obj.Set("e1",      matrixToFloat64Array(env, f.e1));
        obj.Set("e2",      matrixToFloat64Array(env, f.e2));
        obj.Set("normals", matrixToFloat64Array(env, f.normals));
        return obj;
    });
}

// ─── geometry.vertexFrames(surface) → { e1, e2, normals }, each [nV×3] ───
// The per-VERTEX tangent frame = geometry-central's vertexTangentBasis (e1 = basisX,
// e2 = n × e1, n = vertex normal): the gauge the VERTEX connection Laplacian and its
// eigenvectors are written in. A connection coordinate z = a + ib at vertex i is the
// tangent vector a·e1_i + b·e2_i. Parity with MEX 'geometry.vertexFrames' and WASM.
Napi::Value VertexFramesJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        geom::VertexFrames f = geom::vertexFrames(surfaceArg(info));
        auto obj = Napi::Object::New(env);
        obj.Set("e1",      matrixToFloat64Array(env, f.e1));
        obj.Set("e2",      matrixToFloat64Array(env, f.e2));
        obj.Set("normals", matrixToFloat64Array(env, f.normals));
        return obj;
    });
}

// ─── operators.operatorById(id) → operator-registry metadata ───
// Mirrors operatorByIdJS (wasm). Field names match the MEX/WASM struct exactly.
Napi::Value OperatorByIdJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        if (info.Length() < 1 || !info[0].IsString())
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators.operatorById(id): id must be a string.");
        const std::string id = info[0].As<Napi::String>().Utf8Value();
        using namespace nxr::operators::registry;
        const OperatorVariant* v = operatorById(id);
        if (!v) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "unknown operator id: " + id);
        auto o = Napi::Object::New(env);
        auto S = [&](const char* k, const std::string& val) { o.Set(k, Napi::String::New(env, val)); };
        S("id",           v->id);
        S("label",        v->label);
        S("bundle",       toString(v->bundle));
        S("holonomy",     toString(v->holonomy));
        S("order",        toString(v->order));
        S("role",         toString(v->role));
        S("field_type",   toString(v->field_type));
        S("domain",       toString(v->domain));
        S("singular",     toString(v->singular));
        S("gauge",        toString(v->gauge));
        S("coupling",     toString(v->coupling));
        S("natural_mass", v->natural_mass);
        o.Set("graded",   Napi::Boolean::New(env, v->graded));
        S("tau_presets",  v->tau_presets);
        S("status",       toString(v->status));
        S("notes",        v->notes);
        S("squares_to",   v->square.present &&  v->square.isSquaresTo ? v->square.target : std::string());
        S("square_of",    v->square.present && !v->square.isSquaresTo ? v->square.target : std::string());
        S("relation",     v->square.present ? std::string(toString(v->square.relation)) : std::string());
        S("input_field",  v->input_field);
        S("output_field", v->output_field);
        return o;
    });
}

// ─── operators.fieldById(id) → field-registry metadata ───
Napi::Value FieldByIdJS(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return nxrSyncCall(env, [&]() -> Napi::Value {
        if (info.Length() < 1 || !info[0].IsString())
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "operators.fieldById(id): id must be a string.");
        const std::string id = info[0].As<Napi::String>().Utf8Value();
        using namespace nxr::operators::registry;
        const FieldVariant* v = fieldById(id);
        if (!v) throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "unknown field id: " + id);
        const FieldDescriptor& d = v->descriptor;
        auto o = Napi::Object::New(env);
        auto S = [&](const char* k, const std::string& val) { o.Set(k, Napi::String::New(env, val)); };
        S("id",             v->id);
        S("label",          v->label);
        S("domain",         toString(d.domain));
        S("bundle",         toString(d.bundle));
        S("field_type",     toString(d.field_type));
        S("n_form",         toString(d.n_form));
        S("representation", toString(d.representation));
        S("gauge",          toString(d.gauge));
        o.Set("nSym",       Napi::Number::New(env, d.nSym));
        S("notes",          v->notes);
        return o;
    });
}

// ─── Old names: refused, never aliased ───────────────────────
//
// An old export (`heat`, `solve`, `createContext`, ...) is still exported, and throws an error naming the new name; the former context functions went with the
// surface class (the surface is data now).

struct Renamed { const char* oldName; const char* newName; };
static const Renamed kRenamed[] = {
    {"assembleManifoldOperators", "operators.assembleManifoldOperators"}, {"assembleDECOperators", "operators.assembleDECOperators"},
    {"assembleConnectionLaplacian", "operators.assembleConnectionLaplacian"}, {"solve", "operators.eigen"}, {"eigsPencil", "operators.eigen"},
    {"eigs", "operators.eigenOperator"}, {"poisson", "operators.poisson"}, {"heat", "geometry.heat"}, {"tracePath", "geometry.tracePath"},
    {"exactDistance", "geometry.exactDistance"}, {"hodge", "signal.hodge"}, {"curvatures", "geometry.curvatures"}, {"normals", "geometry.normals"},
    {"whitney", "signal.whitney"}, {"gradient", "operators.gradient"}, {"isoline", "signal.isoline"}, {"trivial", "operators.trivial"},
    {"streamline", "signal.streamline"}, {"heatDiffusion", "synthesis.heatDiffusion"}, {"dampedWave", "synthesis.dampedWave"},
    {"parallel", "geometry.parallel"}, {"extendScalar", "geometry.extendScalar"}, {"logMap", "geometry.logMap"}, {"findCenter", "geometry.findCenter"},
    {"signedHeat", "geometry.signedHeat"}, {"smoothFace", "geometry.smoothFace"}, {"smoothVertex", "geometry.smoothVertex"},
    {"frames", "geometry.frames"}, {"vertexFrames", "geometry.vertexFrames"},
    {"operatorInfo", "operators.operatorById"}, {"fieldInfo", "operators.fieldById"},
};
static const char* const kRemovedContext[] = {"createContext", "operators"};

static Napi::Value RefusedJS(const Napi::CallbackInfo& info) {
    const std::string* message = static_cast<const std::string*>(info.Data());
    Napi::Error err = Napi::Error::New(info.Env(), *message);
    err.Set("code", Napi::String::New(info.Env(), "RENAMED"));
    err.ThrowAsJavaScriptException();
    return info.Env().Null();
}

// ─── Module Init ─────────────────────────────────────────────

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    // the surface as data
    exports.Set("geometry.surfaceArrays", Napi::Function::New(env, SurfaceArraysJS));
    // operators
    exports.Set("operators.assembleManifoldOperators", Napi::Function::New(env, AssembleManifoldOperatorsJS));
    exports.Set("operators.assembleDECOperators", Napi::Function::New(env, AssembleDECOperatorsJS));
    exports.Set("operators.assembleConnectionLaplacian", Napi::Function::New(env, AssembleConnectionLaplacianJS));
    exports.Set("operators.eigen", Napi::Function::New(env, EigenJS));
    exports.Set("operators.eigenOperator", Napi::Function::New(env, EigenOperatorJS));
    exports.Set("operators.poisson", Napi::Function::New(env, PoissonJS));
    exports.Set("operators.gradient", Napi::Function::New(env, GradientJS));
    exports.Set("operators.trivial", Napi::Function::New(env, TrivialJS));
    exports.Set("operators.operatorById", Napi::Function::New(env, OperatorByIdJS));
    exports.Set("operators.fieldById", Napi::Function::New(env, FieldByIdJS));
    // geometry
    exports.Set("geometry.heat", Napi::Function::New(env, HeatJS));
    exports.Set("geometry.tracePath", Napi::Function::New(env, TracePathJS));
    exports.Set("geometry.exactDistance", Napi::Function::New(env, ExactDistanceJS));
    exports.Set("geometry.curvatures", Napi::Function::New(env, CurvaturesJS));
    exports.Set("geometry.normals", Napi::Function::New(env, NormalsJS));
    exports.Set("geometry.frames", Napi::Function::New(env, FramesJS));
    exports.Set("geometry.vertexFrames", Napi::Function::New(env, VertexFramesJS));
    // vector heat method
    exports.Set("geometry.parallel",    Napi::Function::New(env, ParallelJS));
    exports.Set("geometry.extendScalar", Napi::Function::New(env, ExtendScalarJS));
    exports.Set("geometry.logMap",       Napi::Function::New(env, LogMapJS));
    exports.Set("geometry.findCenter",   Napi::Function::New(env, FindCenterJS));
    // signed heat method
    exports.Set("geometry.signedHeat",     Napi::Function::New(env, SignedHeatJS));
    // smooth direction fields
    exports.Set("geometry.smoothFace",   Napi::Function::New(env, SmoothFaceJS));
    exports.Set("geometry.smoothVertex", Napi::Function::New(env, SmoothVertexJS));
    // signal
    exports.Set("signal.hodge", Napi::Function::New(env, HodgeJS));
    exports.Set("signal.whitney", Napi::Function::New(env, WhitneyJS));
    exports.Set("signal.isoline", Napi::Function::New(env, IsolineJS));
    exports.Set("signal.streamline", Napi::Function::New(env, StreamlineJS));
    // synthesis
    exports.Set("synthesis.delta", Napi::Function::New(env, DeltaJS));
    exports.Set("synthesis.heatDiffusion", Napi::Function::New(env, HeatDiffusionJS));
    exports.Set("synthesis.dampedWave", Napi::Function::New(env, DampedWaveJS));

    // the old names, refused with the new (their messages live as long as the module)
    static std::vector<std::string> messages;
    messages.clear();
    messages.reserve(sizeof(kRenamed) / sizeof(kRenamed[0]) + sizeof(kRemovedContext) / sizeof(kRemovedContext[0]));
    for (const Renamed& r : kRenamed) {
        messages.push_back(std::string("'") + r.oldName + "' was renamed '" + r.newName + "'");
        exports.Set(r.oldName, Napi::Function::New(env, RefusedJS, r.oldName, &messages.back()));
    }
    for (const char* name : kRemovedContext) {
        messages.push_back(std::string("'") + name + "' was removed with the surface class: the surface is now geometry.surfaceArrays(vertices, faces) data, passed as the first argument to each function");
        exports.Set(name, Napi::Function::New(env, RefusedJS, name, &messages.back()));
    }
    return exports;
}

NODE_API_MODULE(nxr_compute_addon, Init)
