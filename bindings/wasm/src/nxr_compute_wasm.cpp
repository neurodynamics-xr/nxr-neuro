/**
 * nxr_compute_wasm.cpp — Embind bindings exposing nxr-compute to JavaScript via WebAssembly.
 *
 * Mirrors the N-API addon's surface — the same compute functions, the same data shapes — built around Embind's idioms instead of N-API. Designed for browser
 * apps (three.js, plain JS, frameworks) that consume nxr-compute as a portable compute backend.
 *
 * The functions are the library's, by toolbox name (`geometry.heat` calls `nxr::geometry::heat`). They take arrays and return arrays, and nothing is kept
 * between calls (no class, no handle, no cache). A SURFACE is data: the object made once by `geometry.surfaceArrays`, passed as the first argument to every
 * surface function, which rebuilds the library's SurfaceArrays from it. A factorization a function needs is made inside it.
 *
 * Conventions:
 *   • Inputs:  JS typed arrays cross the boundary as `emscripten::val`,
 *              copied into Eigen / std::vector via convertJSArrayToNumberVector.
 *   • Outputs: returned as fresh JS typed arrays (.slice() of a heap view)
 *              so they're safe to keep across subsequent compute calls.
 *   • Sparse: returned as JS objects { row, col, data, rows, cols } in COO.
 *   • Structs (HodgeResult, EigenResult, …): returned as JS objects.
 */

#include <emscripten/bind.h>
#include <emscripten/val.h>

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
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using emscripten::val;

namespace {

// ── Conversion helpers: Eigen → JS typed arrays ──────────────
//
// Pattern: build a heap view via typed_memory_view, then .slice() so
// JS owns the copy. The C++ source can safely go out of scope after.

template <typename T>
val toJsArrayCopy(const T* data, std::size_t n) {
    val view = val(emscripten::typed_memory_view(n, data));
    return view.call<val>("slice");
}

val eigenVectorToVal(const Eigen::VectorXd& v) {
    return toJsArrayCopy(v.data(), static_cast<std::size_t>(v.size()));
}

/** Eigen MatrixXd → row-major flat, suitable for V×3 / F×3 attributes
 *  where each row is one (x, y, z) triple — directly consumable as a
 *  three.js BufferAttribute. */
val eigenMatrixToVal(const Eigen::MatrixXd& m) {
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        rowMajor = m;
    return toJsArrayCopy(rowMajor.data(),
        static_cast<std::size_t>(rowMajor.rows()) * rowMajor.cols());
}

// Per nxr-compute.h's hard layout rule (vMajor for eigenvectors), all
// V×K matrices flatten row-major into JS just like V×3 / F×3.
// Use eigenMatrixToVal above. The previous column-major helper
// was removed in Phase A — the Zarr schema and renderer both
// consume vMajor (U[v*K + k]).

val eigenMatrixFloat32ToVal(const Eigen::MatrixXf& m) {
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        rowMajor = m;
    return toJsArrayCopy(rowMajor.data(),
        static_cast<std::size_t>(rowMajor.rows()) * rowMajor.cols());
}

val sparseToVal(const Eigen::SparseMatrix<double>& M) {
    int nnz = static_cast<int>(M.nonZeros());
    std::vector<int32_t> rows(nnz), cols(nnz);
    std::vector<double>  vals(nnz);
    int k = 0;
    for (int outer = 0; outer < M.outerSize(); outer++) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(M, outer); it; ++it) {
            rows[k] = static_cast<int32_t>(it.row());
            cols[k] = static_cast<int32_t>(it.col());
            vals[k] = it.value();
            k++;
        }
    }
    val obj = val::object();
    obj.set("row",  toJsArrayCopy(rows.data(), nnz));
    obj.set("col",  toJsArrayCopy(cols.data(), nnz));
    obj.set("data", toJsArrayCopy(vals.data(), nnz));
    obj.set("rows", static_cast<int>(M.rows()));
    obj.set("cols", static_cast<int>(M.cols()));
    obj.set("nnz",  nnz);
    return obj;
}

// Complex sparse → COO with parallel real/imag arrays.
// Used by the connection-Laplacian "complex" output format.
val sparseComplexToVal(const Eigen::SparseMatrix<std::complex<double>>& M) {
    int nnz = static_cast<int>(M.nonZeros());
    std::vector<int32_t> rows(nnz), cols(nnz);
    std::vector<double>  re(nnz), im(nnz);
    int k = 0;
    for (int outer = 0; outer < M.outerSize(); outer++) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(M, outer); it; ++it) {
            rows[k] = static_cast<int32_t>(it.row());
            cols[k] = static_cast<int32_t>(it.col());
            re[k]   = it.value().real();
            im[k]   = it.value().imag();
            k++;
        }
    }
    val obj = val::object();
    obj.set("row",       toJsArrayCopy(rows.data(), nnz));
    obj.set("col",       toJsArrayCopy(cols.data(), nnz));
    obj.set("realData",  toJsArrayCopy(re.data(), nnz));
    obj.set("imagData",  toJsArrayCopy(im.data(), nnz));
    obj.set("rows",      static_cast<int>(M.rows()));
    obj.set("cols",      static_cast<int>(M.cols()));
    obj.set("nnz",       nnz);
    return obj;
}

// ── Conversion helpers: JS typed arrays → Eigen ──────────────

Eigen::VectorXd valToEigenVector(const val& jsArr) {
    auto vec = emscripten::convertJSArrayToNumberVector<double>(jsArr);
    Eigen::VectorXd out(vec.size());
    std::memcpy(out.data(), vec.data(), vec.size() * sizeof(double));
    return out;
}

std::vector<int> valToInt32Vector(const val& jsArr) {
    auto vec = emscripten::convertJSArrayToNumberVector<int>(jsArr);
    return vec;
}

std::vector<double> valToDoubleVector(const val& jsArr) {
    return emscripten::convertJSArrayToNumberVector<double>(jsArr);
}

// Re-throw an nxr-compute Error as a std::runtime_error whose message carries
// the "[CODE] msg | hint: ..." shape JS consumers parse. Centralised so every binding method that
// surfaces nxr::core::Error produces an identical JS .message.
[[noreturn]] void rethrowAsJsError(const nxr::core::Error& e) {
    std::string msg = "[";
    msg += nxr::core::errorCodeName(e.code());
    msg += "] ";
    msg += e.what();
    if (!e.hint().empty()) {
        msg += " | hint: ";
        msg += std::string(e.hint());
    }
    throw std::runtime_error(msg);
}

} // namespace

// ── The surface: geometry::SurfaceArrays ↔ a JS object ───────
//
// `geometry.surfaceArrays` makes the surface once (the intrinsic Delaunay triangulation is the expensive step) and returns a plain object; every surface
// function takes that object as its first argument and rebuilds SurfaceArrays from it, in both directions through the two functions below. Nothing is kept
// between calls: no class, no handle, no cache.
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

namespace {

namespace geom = nxr::geometry;
namespace ops = nxr::operators;
namespace cl = nxr::operators::laplacian::connection;
namespace syn = nxr::synthesis;
using nxr::operators::EigenResult;

// Run a body and re-throw an nxr::core::Error as the "[CODE] msg | hint: ..." JS error.
template <typename Fn>
auto guarded(Fn&& fn) -> decltype(fn()) {
    try {
        return fn();
    } catch (const nxr::core::Error& e) {
        rethrowAsJsError(e);
    }
}

[[noreturn]] void invalidInput(const std::string& msg) {
    throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput, msg);
}

const char* massKindName(geom::MassKind k) {
    switch (k) {
        case geom::MassKind::InputMixedVoronoi: return "input-mixed-voronoi";
        case geom::MassKind::Barycentric:       return "barycentric";
        case geom::MassKind::Galerkin:          return "galerkin";
    }
    return "input-mixed-voronoi";
}

geom::MassKind parseMassKind(const std::string& s) {
    if (s == "input-mixed-voronoi") return geom::MassKind::InputMixedVoronoi;
    if (s == "barycentric")         return geom::MassKind::Barycentric;
    if (s == "galerkin")            return geom::MassKind::Galerkin;
    invalidInput("the surface's mass is '" + s + "': use input-mixed-voronoi, barycentric or galerkin");
}

val doublesOut(const std::vector<double>& v) { return toJsArrayCopy(v.data(), v.size()); }

val idsOut(const std::vector<int64_t>& v) {
    std::vector<double> d(v.begin(), v.end());
    return toJsArrayCopy(d.data(), d.size());
}

val surfaceField(const val& o, const char* name) {
    if (!o.hasOwnProperty(name))
        invalidInput(std::string("the surface has no field '") + name + "': make it with geometry.surfaceArrays(vertices, faces)");
    return o[name];
}

std::vector<double> doublesField(const val& o, const char* name) { return emscripten::convertJSArrayToNumberVector<double>(surfaceField(o, name)); }

std::vector<int64_t> idsField(const val& o, const char* name) {
    const std::vector<double> d = doublesField(o, name);
    return std::vector<int64_t>(d.begin(), d.end());
}

double numberField(const val& o, const char* name) {
    val v = surfaceField(o, name);
    if (v.isTrue()) return 1.0;
    if (v.isFalse()) return 0.0;
    return v.as<double>();
}

std::string stringField(const val& o, const char* name) { return surfaceField(o, name).as<std::string>(); }

val surfaceToVal(const geom::SurfaceArrays& s) {
    const geom::IntrinsicTriangulation& t = s.triangulation;
    val opt = val::object();
    opt.set("intrinsic", t.options.intrinsic);
    opt.set("mollify",   t.options.mollify);
    opt.set("flipEps",   t.options.flipEps);
    opt.set("mass",      std::string(massKindName(t.options.mass)));
    opt.set("surface",   t.options.surface);
    val rec = val::object();
    rec.set("mesh",             t.record.mesh);
    rec.set("mollify",          t.record.mollify);
    rec.set("epsilon",          t.record.epsilon);
    rec.set("flipEps",          t.record.flipEps);
    rec.set("flips",            t.record.flips);
    rec.set("delaunay",         t.record.delaunay);
    rec.set("mass",             t.record.mass);
    rec.set("boundaryVertices", t.record.boundaryVertices);
    rec.set("geometryCentral",  t.record.geometryCentral);
    rec.set("fingerprint",      t.record.fingerprint);
    rec.set("gauge",            t.record.gauge);
    rec.set("gaugeId",          t.record.gaugeId);
    val tri = val::object();
    tri.set("options",                opt);
    tri.set("record",                 rec);
    tri.set("mass",                   doublesOut(t.mass));
    tri.set("inputEdgeLength",        doublesOut(t.inputEdgeLength));
    tri.set("halfedgeNext",           idsOut(t.halfedgeNext));
    tri.set("halfedgeVertex",         idsOut(t.halfedgeVertex));
    tri.set("halfedgeFace",           idsOut(t.halfedgeFace));
    tri.set("vertexHalfedge",         idsOut(t.vertexHalfedge));
    tri.set("faceHalfedge",           idsOut(t.faceHalfedge));
    tri.set("boundaryLoops",          static_cast<double>(t.boundaryLoops));
    tri.set("edgeLength",             doublesOut(t.edgeLength));
    tri.set("signpost",               doublesOut(t.signpost));
    tri.set("vertexAngleSum",         doublesOut(t.vertexAngleSum));
    tri.set("halfedgeVectorInVertex", doublesOut(t.halfedgeVectorInVertex));

    geom::RowMat3 pos = s.positions;   // row-major [nV x 3]
    geom::FaceMat fac = s.faces;
    val out = val::object();
    out.set("positions",     toJsArrayCopy(pos.data(), static_cast<std::size_t>(pos.size())));
    out.set("faces",         toJsArrayCopy(fac.data(), static_cast<std::size_t>(fac.size())));
    out.set("triangulation", tri);
    return out;
}

geom::SurfaceArrays valToSurface(const val& o) {
    if (o.isUndefined() || o.isNull() || o.typeOf().as<std::string>() != "object") invalidInput("expected the surface object made by geometry.surfaceArrays(vertices, faces)");
    geom::SurfaceArrays out;
    const std::vector<double> xyz = doublesField(o, "positions");
    const std::vector<int> fac = emscripten::convertJSArrayToNumberVector<int>(surfaceField(o, "faces"));
    if (xyz.size() % 3 != 0 || fac.size() % 3 != 0) invalidInput("the surface's positions and faces hold triples");
    out.positions = Eigen::Map<const geom::RowMat3>(xyz.data(), static_cast<Eigen::Index>(xyz.size() / 3), 3);
    out.faces.resize(static_cast<Eigen::Index>(fac.size() / 3), 3);
    for (std::size_t i = 0; i < fac.size(); ++i) out.faces(static_cast<Eigen::Index>(i / 3), static_cast<Eigen::Index>(i % 3)) = static_cast<std::int32_t>(fac[i]);

    const val tr = surfaceField(o, "triangulation");
    geom::IntrinsicTriangulation& t = out.triangulation;
    const val opt = surfaceField(tr, "options");
    t.options.intrinsic = numberField(opt, "intrinsic") != 0.0;
    t.options.mollify   = numberField(opt, "mollify");
    t.options.flipEps   = numberField(opt, "flipEps");
    t.options.mass      = parseMassKind(stringField(opt, "mass"));
    t.options.surface   = stringField(opt, "surface");
    const val rec = surfaceField(tr, "record");
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

// A COO object {row, col, data, rows, cols} (as `sparseToVal` returns) → Eigen sparse
Eigen::SparseMatrix<double> valToSparse(const val& coo, const char* name) {
    if (!coo.hasOwnProperty("row") || !coo.hasOwnProperty("col") || !coo.hasOwnProperty("data") || !coo.hasOwnProperty("rows") || !coo.hasOwnProperty("cols"))
        invalidInput(std::string(name) + ": expected {row, col, data, rows, cols}");
    const int rows = coo["rows"].as<int>(), cols = coo["cols"].as<int>();
    const auto r = emscripten::convertJSArrayToNumberVector<int32_t>(coo["row"]);
    const auto c = emscripten::convertJSArrayToNumberVector<int32_t>(coo["col"]);
    const auto d = emscripten::convertJSArrayToNumberVector<double>(coo["data"]);
    if (r.size() != c.size() || r.size() != d.size()) invalidInput(std::string(name) + ": row/col/data length mismatch");
    std::vector<Eigen::Triplet<double>> trip;
    trip.reserve(r.size());
    for (std::size_t i = 0; i < r.size(); ++i) {
        if (r[i] < 0 || r[i] >= rows || c[i] < 0 || c[i] >= cols) invalidInput(std::string(name) + ": index out of bounds");
        trip.emplace_back(r[i], c[i], d[i]);
    }
    Eigen::SparseMatrix<double> M(rows, cols);
    M.setFromTriplets(trip.begin(), trip.end());
    M.makeCompressed();
    return M;
}

// JS row-major flat [rows*cols] → MatrixXd
Eigen::MatrixXd valToMatrix(const val& arr, Eigen::Index rows, Eigen::Index cols, const char* what) {
    const std::vector<double> v = emscripten::convertJSArrayToNumberVector<double>(arr);
    if (static_cast<Eigen::Index>(v.size()) != rows * cols) invalidInput(std::string(what) + " must have length " + std::to_string(rows * cols));
    Eigen::MatrixXd m(rows, cols);
    for (Eigen::Index r = 0; r < rows; ++r)
        for (Eigen::Index c = 0; c < cols; ++c) m(r, c) = v[static_cast<std::size_t>(r * cols + c)];
    return m;
}

std::map<int, double> vertexValueMap(const val& verts, const val& values, const char* what) {
    const auto v = emscripten::convertJSArrayToNumberVector<int>(verts);
    const auto x = emscripten::convertJSArrayToNumberVector<double>(values);
    if (v.size() != x.size()) invalidInput(std::string(what) + ": the values must match the vertices in length");
    std::map<int, double> out;
    for (std::size_t i = 0; i < v.size(); ++i) out[v[i]] = x[i];
    return out;
}

val vertexVal(const Eigen::Vector3d& c) {
    double xyz[3] = {c.x(), c.y(), c.z()};
    return toJsArrayCopy(xyz, 3);
}

val logMapVal(const geom::transport::LogMapResult& r) {
    val obj = val::object();
    obj.set("logCoords", eigenMatrixToVal(r.logCoords));
    obj.set("sourceE1", vertexVal(r.sourceE1));
    obj.set("sourceE2", vertexVal(r.sourceE2));
    return obj;
}

val meshOpsToVal(const ops::ManifoldOperators& o, const geom::SurfaceArrays& surface) {
    val v = val::object();
    v.set("cotanLaplacian",  sparseToVal(o.cotanLaplacian));
    v.set("mass",            sparseToVal(o.mass));
    v.set("massVariant",     std::string(
        o.massVariant == ops::MassMatrixVariant::MixedVoronoi ? "mixedVoronoi" :
        o.massVariant == ops::MassMatrixVariant::Barycentric  ? "barycentric"  :
        o.massVariant == ops::MassMatrixVariant::Galerkin     ? "galerkin" : "?"));
    v.set("vertexDualAreas", eigenVectorToVal(o.vertexDualAreas));
    v.set("vertexNormals",   eigenMatrixToVal(o.vertexNormals));
    v.set("totalArea",       o.totalArea);
    v.set("nV",              surface.nV());
    v.set("nE",              surface.nV() + surface.nF() - geom::eulerCharacteristic(surface));   // χ = nV − nE + nF
    v.set("nF",              surface.nF());
    return v;
}

val eigenResultToVal(const EigenResult& r) {
    val o = val::object();
    // vMajor row-major: U[v*K + k]. Matches the cortical-flow Zarr schema (manifold/eigenmodes/eigenvectors stored as [V, K]) and the GPU spectral-synthesis access pattern.
    o.set("eigenvectors", eigenMatrixToVal(r.eigenvectors));
    o.set("eigenvalues",  eigenVectorToVal(r.eigenvalues));
    o.set("k",            r.k);
    o.set("nConverged",   r.nConverged);
    if (r.kind.size() > 0) {   // Hodge–Dirac: 0 exact · 1 harmonic · 2 coexact, per mode
        std::vector<int> kd(r.kind.data(), r.kind.data() + r.kind.size());
        o.set("kind", emscripten::val::array(kd));
    }
    return o;
}

EigenResult valToEigenResult(const val& s) {
    EigenResult r;
    auto uVec = emscripten::convertJSArrayToNumberVector<double>(s["eigenvectors"]);
    auto lVec = emscripten::convertJSArrayToNumberVector<double>(s["eigenvalues"]);
    int K = static_cast<int>(lVec.size());
    int V = K > 0 ? static_cast<int>(uVec.size()) / K : 0;
    // Round-trip layout: JS hands back vMajor (V×K row-major). Map the buffer as a row-major matrix and let Eigen's assignment transpose into its own storage.
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> rowMajor(uVec.data(), V, K);
    r.eigenvectors = rowMajor;
    r.eigenvalues.resize(K);
    if (K > 0) std::memcpy(r.eigenvalues.data(), lVec.data(), K * sizeof(double));
    r.k = K;
    r.nConverged = K;
    return r;
}

bool isMissing(const val& v) { return v.isUndefined() || v.isNull(); }

} // namespace

// ── geometry.surfaceArrays(vertices, faces, options) → surface ────
//
// options: { intrinsicDelaunay?: boolean (default true; false = the named option "embedded": the input as it is), mollify?: number (relative delta, default 1e-6),
//            flipEps?: number (default 1e-6), surface?: string (an id, recorded in the gauge identity) } or null.
val surfaceArraysJS(val verticesArr, val facesArr, val options) {
    return guarded([&] {
        auto verts = emscripten::convertJSArrayToNumberVector<double>(verticesArr);
        auto faces = emscripten::convertJSArrayToNumberVector<int>(facesArr);
        if (verts.size() % 3 != 0) throw std::invalid_argument("vertices length must be a multiple of 3");
        if (faces.size() % 3 != 0) throw std::invalid_argument("faces length must be a multiple of 3");
        std::vector<int32_t> faces32(faces.begin(), faces.end());
        bool intrinsicDelaunay = true;
        geom::DelaunayOptions dopt;
        std::string surfaceId;
        if (!isMissing(options)) {
            if (options.hasOwnProperty("intrinsicDelaunay")) intrinsicDelaunay = options["intrinsicDelaunay"].as<bool>();
            if (options.hasOwnProperty("mollify")) dopt.mollify = options["mollify"].as<double>();
            if (options.hasOwnProperty("flipEps")) dopt.flipEps = options["flipEps"].as<double>();
            if (options.hasOwnProperty("surface")) surfaceId = options["surface"].as<std::string>();
        }
        geom::IntrinsicOptions o = geom::intrinsicOptions(intrinsicDelaunay, dopt);
        o.surface = surfaceId;
        return surfaceToVal(geom::surfaceArrays(verts.data(), static_cast<int>(verts.size() / 3), faces32.data(), static_cast<int>(faces32.size() / 3), o));
    });
}

// ── operators ────────────────────────────────────────────────

// `variantName` ∈ "mixedVoronoi" (default, also for "") | "galerkin" | "barycentric" ("lumped" is retired).
val assembleManifoldOperatorsJS(val surfaceVal, std::string variantName) {
    return guarded([&] {
        const geom::SurfaceArrays surface = valToSurface(surfaceVal);
        const ops::MassMatrixVariant variant = variantName.empty() ? ops::MassMatrixVariant::MixedVoronoi : ops::parseMassMatrixVariant(variantName);
        return meshOpsToVal(ops::assembleManifoldOperators(surface, variant), surface);
    });
}

val assembleDECOperatorsJS(val surfaceVal) {
    return guarded([&] {
        const ops::DECOperators dec = ops::assembleDECOperators(valToSurface(surfaceVal));
        val o = val::object();
        o.set("d0",            sparseToVal(dec.d0));
        o.set("d1",            sparseToVal(dec.d1));
        o.set("hodge0",        sparseToVal(dec.hodge0));
        o.set("hodge1",        sparseToVal(dec.hodge1));
        o.set("hodge2",        sparseToVal(dec.hodge2));
        o.set("hodge1Inverse", sparseToVal(dec.hodge1Inverse));
        return o;
    });
}

// Connection Laplacian on the chosen domain (vertex / face / edge).
// `opts` is a JS object: { domain?, nSym?, regularization?, format? }.
// Defaults: { domain: 'vertex', nSym: 1, regularization: 1e-8, format: 'real2N' }.
val assembleConnectionLaplacianJS(val surfaceVal, val opts) {
    return guarded([&] {
        const geom::SurfaceArrays surface = valToSurface(surfaceVal);
        cl::ConnectionLaplacianOptions o;
        if (!isMissing(opts)) {
            val domainV = opts["domain"];
            val nSymV   = opts["nSym"];
            val regV    = opts["regularization"];
            val fmtV    = opts["format"];
            if (!domainV.isUndefined()) o.domain = cl::parseConnectionDomain(domainV.as<std::string>());
            if (!nSymV.isUndefined())   o.nSym = nSymV.as<int>();
            if (!regV.isUndefined())    o.regularization = regV.as<double>();
            if (!fmtV.isUndefined())    o.format = cl::parseConnectionLaplacianFormat(fmtV.as<std::string>());
        }
        const cl::ConnectionLaplacian c = cl::assembleConnectionLaplacian(surface, o);

        const char* domainStr =
            c.domain == cl::ConnectionDomain::Vertex              ? "vertex" :
            c.domain == cl::ConnectionDomain::Face                ? "face"   :
            c.domain == cl::ConnectionDomain::EdgeCrouzeixRaviart ? "edge"   : "?";
        const char* formatStr = c.format == cl::ConnectionLaplacianFormat::Real2N ? "real2N" : "complex";

        val out = val::object();
        if (c.format == cl::ConnectionLaplacianFormat::Real2N) out.set("K", sparseToVal(c.K_real));
        else                                                     out.set("K", sparseComplexToVal(c.K_complex));
        out.set("baseDim",        c.baseDim);
        out.set("outputDim",      c.outputDim);
        out.set("domain",         std::string(domainStr));
        out.set("nSym",           c.nSym);
        out.set("regularization", c.regularization);
        out.set("format",         std::string(formatStr));
        return out;
    });
}

/**
 * operators.eigen: generalized eigenproblem K φ = λ M φ from JS-supplied COO triplets (K and M assembled by `operators.assembleManifoldOperators`, or any other
 * sparse Laplacians: graph Laplacians, FEM assemblies, custom regularized matrices).
 *
 * Args (all coordinate-format inputs are 0-based):
 *   K_rows, K_cols  Int32Array (or array of ints), length = K_nnz
 *   K_vals          Float64Array (or array of numbers), length = K_nnz
 *   M_rows, M_cols, M_vals  same shape as K
 *   n               common matrix size for K and M (square)
 *   k, sigma        eigensolve params (sigma: an explicit shift; NaN = the scale-aware shift ladder, kShiftLadder)
 *   normalize       M-orthonormalize the eigenvectors; removeDC  strip the null space
 *   cancelAddr      wasm heap pointer to a single int32 (0 = none). Allocate via `Module._malloc(4)` and write via `Module.HEAP32[ptr >> 2] = 1` to cancel.
 *                   With SHARED_MEMORY, JS can use `Atomics.store(new Int32Array(Module.HEAPU8.buffer), ptr >> 2, 1)` from a different thread.
 *   progressAddr, progressLen   wasm heap pointer to a 3×int32 array (0 = none). Layout: [iteration, totalIterations, residual×1e6].
 *
 * Returns: { eigenvectors, eigenvalues, k, nConverged } in vMajor row-major layout (Φ[v, k] at offset v*K + k).
 * Errors land in JS as Error objects whose .message starts with "[CODE] " (the nxr::core::ErrorCode name, e.g. CANCELLED).
 * Triplets with duplicate (row, col) pairs are summed when the sparse matrix is built (Eigen's setFromTriplets default).
 */
val eigenJS(
    val K_rows, val K_cols, val K_vals,
    val M_rows, val M_cols, val M_vals,
    int n,
    int k, double sigma,
    std::intptr_t cancelAddr,
    std::intptr_t progressAddr,
    int progressLen,
    bool normalize,
    bool removeDC)
{
    auto Kr = emscripten::convertJSArrayToNumberVector<int32_t>(K_rows);
    auto Kc = emscripten::convertJSArrayToNumberVector<int32_t>(K_cols);
    auto Kv = emscripten::convertJSArrayToNumberVector<double>(K_vals);
    auto Mr = emscripten::convertJSArrayToNumberVector<int32_t>(M_rows);
    auto Mc = emscripten::convertJSArrayToNumberVector<int32_t>(M_cols);
    auto Mv = emscripten::convertJSArrayToNumberVector<double>(M_vals);

    if (Kr.size() != Kc.size() || Kr.size() != Kv.size()) throw std::invalid_argument("operators.eigen: K row/col/val length mismatch");
    if (Mr.size() != Mc.size() || Mr.size() != Mv.size()) throw std::invalid_argument("operators.eigen: M row/col/val length mismatch");
    if (n <= 0) throw std::invalid_argument("operators.eigen: n must be > 0");

    Eigen::SparseMatrix<double> K(n, n), M(n, n);
    {
        std::vector<Eigen::Triplet<double>> Kt;
        Kt.reserve(Kr.size());
        for (std::size_t i = 0; i < Kr.size(); ++i) Kt.emplace_back(Kr[i], Kc[i], Kv[i]);
        K.setFromTriplets(Kt.begin(), Kt.end());
        K.makeCompressed();
    }
    {
        std::vector<Eigen::Triplet<double>> Mt;
        Mt.reserve(Mr.size());
        for (std::size_t i = 0; i < Mr.size(); ++i) Mt.emplace_back(Mr[i], Mc[i], Mv[i]);
        M.setFromTriplets(Mt.begin(), Mt.end());
        M.makeCompressed();
    }

    nxr::core::CancellationToken cancel = cancelAddr
        ? nxr::core::CancellationToken(reinterpret_cast<const std::atomic<int32_t>*>(cancelAddr))
        : nxr::core::CancellationToken{};
    nxr::core::ProgressObserver progress;
    if (progressAddr && progressLen >= 3) {
        auto* base = reinterpret_cast<std::atomic<int32_t>*>(progressAddr);
        progress.iteration       = &base[0];
        progress.totalIterations = &base[1];
        progress.residualMicro   = &base[2];
    }

    return guarded([&] { return eigenResultToVal(ops::eigen(K, M, k, sigma, normalize, removeDC, cancel, progress)); });
}

// operators.eigenOperator: named-operator eigensolve — assembles the operator AND its natural generalized mass C++-side (no JS-side ⊗I₄, one boundary crossing)
// and eigensolves via the shared operators::eigenOperator (single source of truth).
// opts: { operator: 'laplacian'|'cotan'|'graph'|'dirac'|'diracFace'|'hodgeDirac'|'spinExtrinsic'|'spinIntrinsicHY'|'spinIntrinsicYe',
//         subtype?: 'cotan'|'graph',           // when operator === 'laplacian'
//         tau?: number,                        // dirac / diracFace blend ∈ [0,1]
//         graphMass?: 'identity'|'degree', star1Floor?, mollify?, bothSigns?   // graph / hodgeDirac
//         mass?: 'mixedVoronoi'|'galerkin'|'barycentric',   // vertex-domain mass (default mixedVoronoi)
//         k: number, sigma?: number (absent = the shift ladder, kShiftLadder),
//         normalize?: boolean (= true),
//         multiplets?: boolean (= false),      // Dirac: exact 4-fold ℍ-reconstruction
//         dense?: boolean (= false) }          // exact small-mesh verification
// Returns { eigenvectors (vMajor, n·k), eigenvalues, k, nConverged, blockSize }.
val eigenOperatorJS(val surfaceVal, val opts) {
    if (isMissing(opts) || opts["operator"].isUndefined()) throw std::runtime_error("[INVALID_INPUT] operators.eigenOperator: expected { operator, k, ... }.");
    return guarded([&] {
        const geom::SurfaceArrays surface = valToSurface(surfaceVal);
        ops::EigenOperatorSpec spec;
        const std::string op = opts["operator"].as<std::string>();
        int blockSize = 1;
        if (op == "laplacian") {
            const std::string sub = opts["subtype"].isUndefined() ? "" : opts["subtype"].as<std::string>();
            if      (sub == "cotan") spec.op = ops::EigenOperator::LaplacianCotan;
            else if (sub == "graph") spec.op = ops::EigenOperator::LaplacianGraph;
            else invalidInput("operators.eigenOperator laplacian: subtype must be cotan|graph.");
        } else if (op == "cotan") {
            spec.op = ops::EigenOperator::LaplacianCotan;
        } else if (op == "graph") {
            spec.op = ops::EigenOperator::LaplacianGraph;
        } else if (op == "dirac") {
            spec.op = ops::EigenOperator::Dirac; blockSize = 4;
        } else if (op == "diracFace") {
            spec.op = ops::EigenOperator::DiracFace; blockSize = 4;
        } else if (op == "hodgeDirac") {
            spec.op = ops::EigenOperator::HodgeDirac;
        } else if (op == "spinExtrinsic") {
            spec.op = ops::EigenOperator::SpinExtrinsic; blockSize = 4;
        } else if (op == "spinIntrinsicHY") {
            spec.op = ops::EigenOperator::SpinIntrinsicHY; blockSize = 4;
        } else if (op == "spinIntrinsicYe") {
            spec.op = ops::EigenOperator::SpinIntrinsicYe; blockSize = 4;
        } else {
            invalidInput("operators.eigenOperator: operator must be laplacian|cotan|graph|dirac|diracFace|hodgeDirac|spinExtrinsic|spinIntrinsicHY|spinIntrinsicYe.");
        }
        if (!opts["graphMass"].isUndefined()) spec.graphDegreeMass = opts["graphMass"].as<std::string>() == "degree";
        if (!opts["star1Floor"].isUndefined()) spec.star1Floor = opts["star1Floor"].as<double>();
        if (!opts["mollify"].isUndefined())    spec.mollify    = opts["mollify"].as<double>();
        if (!opts["bothSigns"].isUndefined())  spec.bothSigns  = opts["bothSigns"].as<bool>();
        if (!opts["tau"].isUndefined())  spec.tau  = opts["tau"].as<double>();
        if (!opts["mass"].isUndefined()) spec.mass = ops::parseMassMatrixVariant(opts["mass"].as<std::string>());
        if (opts["k"].isUndefined()) invalidInput("operators.eigenOperator: k is required.");

        const int    k          = opts["k"].as<int>();
        const double sigma      = opts["sigma"].isUndefined()      ? ops::kShiftLadder : opts["sigma"].as<double>();
        const bool   normalize  = opts["normalize"].isUndefined()  ? true  : opts["normalize"].as<bool>();
        const bool   multiplets = opts["multiplets"].isUndefined() ? false : opts["multiplets"].as<bool>();
        const bool   dense      = opts["dense"].isUndefined()      ? false : opts["dense"].as<bool>();

        EigenResult r = ops::eigenOperator(surface, spec, k, sigma, normalize, multiplets, dense);
        val out = eigenResultToVal(r);
        out.set("blockSize", blockSize);
        return out;
    });
}

// operators.normalize(U (vMajor row-major, rows x cols), rows, cols, M COO) → U normalized (M-orthonormal), vMajor row-major
val normalizeJS(val UJsArr, int rows, int cols, val Mcoo) {
    return guarded([&] {
        Eigen::MatrixXd U = valToMatrix(UJsArr, rows, cols, "eigenvectors");
        Eigen::SparseMatrix<double> M = valToSparse(Mcoo, "operators.normalize M");
        return eigenMatrixToVal(ops::normalize(U, M));
    });
}

val removeDCJS(val eigStruct) {
    return guarded([&] { return eigenResultToVal(ops::removeDC(valToEigenResult(eigStruct))); });
}

// operators.poisson(K, M, sourceVerts, sourceValues): K, M as COO objects; the Laplacian's factor is made inside, from K.
val poissonJS(val Kcoo, val Mcoo, val sourceVertsArr, val sourceValuesArr) {
    return guarded([&] {
        Eigen::SparseMatrix<double> K = valToSparse(Kcoo, "operators.poisson K");
        Eigen::SparseMatrix<double> M = valToSparse(Mcoo, "operators.poisson M");
        const std::map<int, double> srcMap = vertexValueMap(sourceVertsArr, sourceValuesArr, "operators.poisson");
        const std::unique_ptr<ops::CholeskyFactor> factor = ops::laplacianFactor(K);
        return eigenVectorToVal(ops::poisson(K, M, *factor, srcMap));
    });
}

val gradientJS(val surfaceVal, val scalarArr) {
    return guarded([&] { return eigenMatrixToVal(ops::gradient(valToSurface(surfaceVal), valToEigenVector(scalarArr))); });
}

// operators.trivial(surface, singVerts, singValues): the trivial-connection direction field with prescribed singularities (their indices sum to χ).
val trivialJS(val surfaceVal, val singVertsArr, val singValuesArr) {
    return guarded([&] {
        const geom::SurfaceArrays surface = valToSurface(surfaceVal);
        const std::map<int, double> singMap = vertexValueMap(singVertsArr, singValuesArr, "operators.trivial");
        const ops::DECOperators dec = ops::assembleDECOperators(surface);
        const std::unique_ptr<ops::CholeskyFactor> exact = ops::exactFormFactor(dec);
        ops::connection::DirectionFieldResult r = ops::connection::trivial(surface, dec, *exact, singMap);
        val o = val::object();
        o.set("connections",          eigenVectorToVal(r.connections));
        o.set("directionVectors",     eigenMatrixToVal(r.directionVectors));
        o.set("orthogonalVectors",    eigenMatrixToVal(r.orthogonalVectors));
        o.set("eulerCharacteristic",  r.eulerCharacteristic);
        o.set("gaussBonnetSatisfied", r.gaussBonnetSatisfied);
        return o;
    });
}

// ── signal ───────────────────────────────────────────────────

// signal.hodge(surface, omega): omega is a 1-form on the OPERATOR (intrinsic triangulation) mesh's edges; the DEC and the two factorizations are made inside.
val hodgeJS(val surfaceVal, val omegaArr) {
    return guarded([&] {
        const geom::SurfaceArrays surface = valToSurface(surfaceVal);
        Eigen::VectorXd omega = valToEigenVector(omegaArr);
        const ops::DECOperators dec = ops::assembleDECOperators(surface);
        const std::unique_ptr<ops::CholeskyFactor> exact = ops::exactFormFactor(dec);
        const std::unique_ptr<ops::SquareSolver> coexact = ops::coexactFormSolver(dec);
        nxr::signal::HodgeResult h = nxr::signal::hodge(surface, dec, *exact, *coexact, omega);
        val o = val::object();
        o.set("exactPotential",     eigenVectorToVal(h.exactPotential));
        o.set("coExactPotentialV",  eigenVectorToVal(h.coExactPotentialV));
        o.set("combinedPotential",  eigenVectorToVal(h.combinedPotential));
        o.set("dAlpha",             eigenVectorToVal(h.dAlpha));
        o.set("deltaBeta",          eigenVectorToVal(h.deltaBeta));
        o.set("gamma",              eigenVectorToVal(h.gamma));
        o.set("omegaVectors",       eigenMatrixToVal(h.omegaVectors));
        o.set("dAlphaVectors",      eigenMatrixToVal(h.dAlphaVectors));
        o.set("deltaBetaVectors",   eigenMatrixToVal(h.deltaBetaVectors));
        o.set("gammaVectors",       eigenMatrixToVal(h.gammaVectors));
        return o;
    });
}

val isolineJS(val surfaceVal, val scalarsArr, int numLevels, double minVal, double maxVal) {
    return guarded([&] {
        nxr::signal::IsolineResult r = nxr::signal::isoline(valToSurface(surfaceVal), valToEigenVector(scalarsArr), numLevels, minVal, maxVal);
        val out = val::object();
        out.set("positions",    eigenMatrixToVal(r.positions));
        out.set("segmentCount", r.segmentCount);
        return out;
    });
}

val streamlineJS(val surfaceVal, val faceFieldArr, int numSeeds, double stepCoef, int maxSteps) {
    return guarded([&] {
        const geom::SurfaceArrays surface = valToSurface(surfaceVal);
        Eigen::MatrixXd faceField = valToMatrix(faceFieldArr, surface.nF(), 3, "faceField");
        nxr::signal::StreamlineResult r = nxr::signal::streamline(surface, faceField, numSeeds, stepCoef, maxSteps);
        val o = val::object();
        o.set("positions",    eigenMatrixToVal(r.positions));
        o.set("segmentCount", r.segmentCount);
        return o;
    });
}

// signal.whitney(surface, oneForm): a 1-form on the OPERATOR (intrinsic triangulation) mesh's edges to face vectors [nF*3].
val whitneyJS(val surfaceVal, val oneFormArr) {
    return guarded([&] {
        const geom::SurfaceArrays surface = valToSurface(surfaceVal);
        return eigenMatrixToVal(nxr::signal::whitney(surface, ops::assembleDECOperators(surface), valToEigenVector(oneFormArr)));
    });
}

// ── geometry ─────────────────────────────────────────────────

val curvaturesJS(val surfaceVal) {
    return guarded([&] {
        geom::CurvatureResult c = geom::curvatures(valToSurface(surfaceVal));
        val o = val::object();
        o.set("gaussian",     eigenVectorToVal(c.gaussian));
        o.set("mean",         eigenVectorToVal(c.mean));
        o.set("kMin",         eigenVectorToVal(c.kMin));
        o.set("kMax",         eigenVectorToVal(c.kMax));
        o.set("principalDir", eigenMatrixToVal(c.principalDirMax));
        return o;
    });
}

val bffJS(val surfaceVal) {
    return guarded([&] { return eigenMatrixToVal(geom::parametrization::bff(valToSurface(surfaceVal))); });
}

val framesJS(val surfaceVal) {
    return guarded([&] {
        auto f = geom::frames(valToSurface(surfaceVal));
        val obj = val::object();
        obj.set("e1",      eigenMatrixToVal(f.e1));
        obj.set("e2",      eigenMatrixToVal(f.e2));
        obj.set("normals", eigenMatrixToVal(f.normals));
        return obj;
    });
}

/** Per-VERTEX tangent frames — the vertex connection Laplacian's gauge (geometry-central vertexTangentBasis). Parity with node/MEX 'geometry.vertexFrames'. */
val vertexFramesJS(val surfaceVal) {
    return guarded([&] {
        auto f = geom::vertexFrames(valToSurface(surfaceVal));
        val obj = val::object();
        obj.set("e1",      eigenMatrixToVal(f.e1));
        obj.set("e2",      eigenMatrixToVal(f.e2));
        obj.set("normals", eigenMatrixToVal(f.normals));
        return obj;
    });
}

val normalsJS(val surfaceVal, int type) {
    return guarded([&] { return eigenMatrixToVal(geom::normals(valToSurface(surfaceVal), static_cast<geom::NormalType>(type))); });
}

// geometry.heat(surface, sources, tCoef, mesh): sources a vertex array (→ Float64Array [nV]) or an Array of vertex arrays (the batch form, one factorization for
// all → an Array of Float64Array [nV]); mesh = 'intrinsic' | 'input'.
val heatJS(val surfaceVal, val sources, double tCoef, std::string meshName) {
    return guarded([&] {
        const geom::SurfaceArrays surface = valToSurface(surfaceVal);
        geom::HeatMesh mesh = geom::HeatMesh::Intrinsic;
        if (meshName == "input") mesh = geom::HeatMesh::Input;
        else if (!meshName.empty() && meshName != "intrinsic") invalidInput("heat: mesh must be 'intrinsic' or 'input'");
        if (sources.isArray()) {
            std::vector<std::vector<int>> sets;
            const int n = sources["length"].as<int>();
            for (int i = 0; i < n; ++i) sets.push_back(emscripten::convertJSArrayToNumberVector<int>(sources[i]));
            const Eigen::MatrixXd D = geom::heat(surface, sets, tCoef, mesh);   // column j = the distances from set j
            val out = val::array();
            for (Eigen::Index j = 0; j < D.cols(); ++j) out.call<void>("push", eigenVectorToVal(D.col(j)));
            return out;
        }
        return eigenVectorToVal(geom::heat(surface, emscripten::convertJSArrayToNumberVector<int>(sources), tCoef, mesh));
    });
}

val tracePathJS(val surfaceVal, int vStart, int vEnd) {
    return guarded([&] {
        geom::query::GeodesicPath path = geom::query::tracePathSurface(valToSurface(surfaceVal), vStart, vEnd);
        val out = val::object();
        out.set("positions", eigenMatrixToVal(path.positions));
        out.set("nPoints",   static_cast<int>(path.positions.rows()));
        // where each point lies on the mesh: up to three vertices and their weights
        std::vector<int> verts(path.vertices.rows() * 3);
        for (int i = 0; i < path.vertices.rows(); i++)
            for (int k = 0; k < 3; k++) verts[i * 3 + k] = path.vertices(i, k);
        out.set("vertices",  val(emscripten::typed_memory_view(verts.size(), verts.data())).call<val>("slice"));
        out.set("weights",   eigenMatrixToVal(path.weights));
        out.set("length",    path.length);
        return out;
    });
}

// the exact (MMP) geodesic distance between two vertices, mesh units
double exactDistanceJS(val surfaceVal, int vStart, int vEnd) {
    return guarded([&] { return geom::query::exactDistance(valToSurface(surfaceVal), vStart, vEnd); });
}

// ── geometry: vector heat method ─────────────────────────────
//
// Each call builds the solver on the surface and drops it on return; the batch forms (Arrays of source sets / sources) share one solver and one factorization.

// geometry.parallel(surface, sourceVerts, sourceVectors, tCoef): one set (vertex array, Float64Array [3n] → Float64Array [nV*3]) or Arrays of them (→ an Array of them).
val parallelJS(val surfaceVal, val sourceVerts, val sourceVectors, double tCoef) {
    return guarded([&] {
        const geom::SurfaceArrays surface = valToSurface(surfaceVal);
        if (sourceVerts.isArray() != sourceVectors.isArray()) invalidInput("sourceVerts and sourceVectors are both Arrays (the batch form) or both typed arrays");
        if (sourceVerts.isArray()) {
            const int n = sourceVerts["length"].as<int>();
            if (sourceVectors["length"].as<int>() != n) invalidInput("sourceVectors holds one array per set of sourceVerts");
            std::vector<std::vector<int>> sets;
            std::vector<Eigen::MatrixXd> vecs;
            for (int i = 0; i < n; ++i) {
                sets.push_back(emscripten::convertJSArrayToNumberVector<int>(sourceVerts[i]));
                vecs.push_back(valToMatrix(sourceVectors[i], static_cast<Eigen::Index>(sets.back().size()), 3, "each set of sourceVectors"));
            }
            const auto out = geom::transport::parallel(surface, sets, vecs, tCoef);
            val arr = val::array();
            for (const auto& m : out) arr.call<void>("push", eigenMatrixToVal(m));
            return arr;
        }
        auto verts = emscripten::convertJSArrayToNumberVector<int>(sourceVerts);
        Eigen::MatrixXd S = valToMatrix(sourceVectors, static_cast<Eigen::Index>(verts.size()), 3, "sourceVectors (Nx3, length 3*sources)");
        return eigenMatrixToVal(geom::transport::parallel(surface, verts, S, tCoef));
    });
}

val extendScalarJS(val surfaceVal, val sourceVertsArr, val sourceValuesArr, double tCoef) {
    return guarded([&] {
        const geom::SurfaceArrays surface = valToSurface(surfaceVal);
        auto verts = emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr);
        auto vals  = emscripten::convertJSArrayToNumberVector<double>(sourceValuesArr);
        if (verts.size() != vals.size()) invalidInput("sourceVerts and sourceValues must have the same length");
        Eigen::VectorXd vv(vals.size());
        for (std::size_t i = 0; i < vals.size(); i++) vv[i] = vals[i];
        return eigenVectorToVal(geom::transport::extendScalar(surface, verts, vv, tCoef));
    });
}

// geometry.logMap(surface, sourceVertex, strategy, tCoef): a number → one { logCoords, sourceE1, sourceE2 }; an array of sources (the batch form, one solver) → an Array of them.
val logMapJS(val surfaceVal, val sourceVertex, int strategy, double tCoef) {
    return guarded([&] {
        const geom::SurfaceArrays surface = valToSurface(surfaceVal);
        auto s = static_cast<geom::transport::LogMapStrategy>(strategy);
        if (sourceVertex.isNumber()) return logMapVal(geom::transport::logMap(surface, sourceVertex.as<int>(), s, tCoef));
        const auto results = geom::transport::logMap(surface, emscripten::convertJSArrayToNumberVector<int>(sourceVertex), s, tCoef);
        val arr = val::array();
        for (const auto& r : results) arr.call<void>("push", logMapVal(r));
        return arr;
    });
}

val findCenterJS(val surfaceVal, val sourceVertsArr, int p, double tCoef) {
    return guarded([&] {
        return vertexVal(geom::transport::findCenter(valToSurface(surfaceVal), emscripten::convertJSArrayToNumberVector<int>(sourceVertsArr), p, tCoef));
    });
}

// geometry.signedHeat(surface, curve, isLoop, levelSet, tCoef): curve a vertex array (→ Float64Array [nV]) or an Array of them (the batch form → an Array of Float64Array [nV]).
val signedHeatJS(val surfaceVal, val curve, bool isLoop, int levelSet, double tCoef) {
    return guarded([&] {
        const geom::SurfaceArrays surface = valToSurface(surfaceVal);
        auto ls = static_cast<geom::SignedHeatLevelSet>(levelSet);
        if (curve.isArray()) {
            std::vector<std::vector<int>> curves;
            const int n = curve["length"].as<int>();
            for (int i = 0; i < n; ++i) curves.push_back(emscripten::convertJSArrayToNumberVector<int>(curve[i]));
            const Eigen::MatrixXd D = geom::signedHeat(surface, curves, isLoop, ls, tCoef);
            val out = val::array();
            for (Eigen::Index j = 0; j < D.cols(); ++j) out.call<void>("push", eigenVectorToVal(D.col(j)));
            return out;
        }
        return eigenVectorToVal(geom::signedHeat(surface, emscripten::convertJSArrayToNumberVector<int>(curve), isLoop, ls, tCoef));
    });
}

// ── geometry: smooth direction fields ────

val smoothFaceJS(val surfaceVal, int nSym, bool alignToCurvature) {
    return guarded([&] { return eigenMatrixToVal(geom::connection::smoothFace(valToSurface(surfaceVal), nSym, alignToCurvature)); });
}

val smoothVertexJS(val surfaceVal, int nSym, bool alignToCurvature) {
    return guarded([&] {
        geom::connection::SmoothVertexFieldResult r = geom::connection::smoothVertex(valToSurface(surfaceVal), nSym, alignToCurvature);
        val obj = val::object();
        obj.set("vertexVectors",  eigenMatrixToVal(r.vertexVectors));
        obj.set("vertexFieldRaw", eigenVectorToVal(r.vertexFieldRaw));
        obj.set("nSym",           r.nSym);
        return obj;
    });
}

// ── synthesis: field generators ──────────────────────────────

// synthesis.delta(nV, sourceVerts, sourceValues) → Float64Array [nV]: zero except at the sources.
val deltaJS(int nV, val sourceVertsArr, val sourceValuesArr) {
    return guarded([&] { return eigenVectorToVal(syn::delta(nV, vertexValueMap(sourceVertsArr, sourceValuesArr, "synthesis.delta"))); });
}

// synthesis.heatDiffusion(M (COO), eig, u0, timesteps, alpha) → { data: Float32Array [T*nV], T, nV }; eig: the result of an eigensolve (M-orthonormal modes).
val heatDiffusionJS(val Mcoo, val eigStruct, val u0Arr, val timestepsArr, double alpha) {
    return guarded([&] {
        Eigen::SparseMatrix<double> M = valToSparse(Mcoo, "synthesis.heatDiffusion M");
        EigenResult eig = valToEigenResult(eigStruct);
        Eigen::VectorXd u0 = valToEigenVector(u0Arr);
        std::vector<double> ts = valToDoubleVector(timestepsArr);
        Eigen::MatrixXf field = syn::heatDiffusion(M, eig, u0, ts, alpha);
        val out = val::object();
        out.set("data", eigenMatrixFloat32ToVal(field));
        out.set("T",  static_cast<int>(field.rows()));
        out.set("nV", static_cast<int>(field.cols()));
        return out;
    });
}

val dampedWaveJS(val eigStruct, val modeIndicesArr, val amplitudesArr, val dampingsArr, val phasesArr, val timestepsArr) {
    return guarded([&] {
        EigenResult eig = valToEigenResult(eigStruct);
        auto mi  = valToInt32Vector(modeIndicesArr);
        auto am  = valToDoubleVector(amplitudesArr);
        auto da  = valToDoubleVector(dampingsArr);
        auto ph  = valToDoubleVector(phasesArr);
        auto ts  = valToDoubleVector(timestepsArr);
        Eigen::MatrixXf field = syn::dampedWave(eig, mi, am, da, ph, ts);
        val out = val::object();
        out.set("data", eigenMatrixFloat32ToVal(field));
        out.set("T",  static_cast<int>(field.rows()));
        out.set("nV", static_cast<int>(field.cols()));
        return out;
    });
}

// synthesis.randomDecomposed1Form(surface, alphaStrength, betaStrength, gammaStrength, seed) → a 1-form on the OPERATOR (intrinsic triangulation) mesh's edges
val randomDecomposed1FormJS(val surfaceVal, double alphaStrength, double betaStrength, double gammaStrength, int seed) {
    return guarded([&] {
        const geom::SurfaceArrays surface = valToSurface(surfaceVal);
        const ops::DECOperators dec = ops::assembleDECOperators(surface);
        return eigenVectorToVal(syn::randomDecomposed1Form(dec, surface.nV(), static_cast<int>(dec.d0.rows()), surface.nF(),
                                                           alphaStrength, betaStrength, gammaStrength, static_cast<unsigned int>(seed)));
    });
}

// ── Free functions ───────────────────────────────────────────

std::string getVersion() {
    return "nxr-compute 0.1.0";
}

// ── Operator registry metadata ───────────────────────────────
//
// Mirrors the MEX 'operators.operatorById' command (bindings/mex/src/nxr_compute_mex.cpp).
// Returns a plain JS object with all OperatorVariant metadata strings. Field
// names match the MEX struct fields exactly so downstream consumers work
// identically against either binding.
emscripten::val operatorByIdJS(std::string id) {
    using namespace nxr::operators::registry;
    const OperatorVariant* v = operatorById(id);
    if (!v) {
        try {
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                                   "unknown operator id: " + id);
        } catch (const nxr::core::Error& e) { rethrowAsJsError(e); }
    }
    emscripten::val o = emscripten::val::object();
    o.set("id",           v->id);
    o.set("label",        v->label);
    o.set("bundle",       std::string(toString(v->bundle)));
    o.set("holonomy",     std::string(toString(v->holonomy)));
    o.set("order",        std::string(toString(v->order)));
    o.set("role",         std::string(toString(v->role)));
    o.set("field_type",   std::string(toString(v->field_type)));
    o.set("domain",       std::string(toString(v->domain)));
    o.set("singular",     std::string(toString(v->singular)));
    o.set("gauge",        std::string(toString(v->gauge)));
    o.set("coupling",     std::string(toString(v->coupling)));
    o.set("natural_mass", v->natural_mass);
    o.set("graded",       v->graded);
    o.set("tau_presets",  v->tau_presets);
    o.set("status",       std::string(toString(v->status)));
    o.set("notes",        v->notes);
    o.set("squares_to",   v->square.present &&  v->square.isSquaresTo ? v->square.target : std::string());
    o.set("square_of",    v->square.present && !v->square.isSquaresTo ? v->square.target : std::string());
    o.set("relation",     v->square.present ? std::string(toString(v->square.relation)) : std::string());
    o.set("input_field",  v->input_field);
    o.set("output_field", v->output_field);
    return o;
}

// ── Field registry metadata ──────────────────────────────────
//
// Mirrors the MEX 'operators.fieldById' command (bindings/mex/src/nxr_compute_mex.cpp).
// Returns a plain JS object with all FieldVariant metadata strings. Field
// names match the MEX struct fields exactly so downstream consumers work
// identically against either binding.
emscripten::val fieldByIdJS(std::string id) {
    using namespace nxr::operators::registry;
    const FieldVariant* v = fieldById(id);
    if (!v) {
        try {
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                                   "unknown field id: " + id);
        } catch (const nxr::core::Error& e) { rethrowAsJsError(e); }
    }
    const FieldDescriptor& d = v->descriptor;
    emscripten::val o = emscripten::val::object();
    o.set("id",             v->id);
    o.set("label",          v->label);
    o.set("domain",         std::string(toString(d.domain)));
    o.set("bundle",         std::string(toString(d.bundle)));
    o.set("field_type",     std::string(toString(d.field_type)));
    o.set("n_form",         std::string(toString(d.n_form)));
    o.set("representation", std::string(toString(d.representation)));
    o.set("gauge",          std::string(toString(d.gauge)));
    o.set("nSym",           d.nSym);
    o.set("notes",          v->notes);
    return o;
}

// ── Embind module bindings ───────────────────────────────────
//
// Free functions named `<toolbox>.<function>` (one name per thing across the C++ function, the node and the bindings). Embind takes every argument (no
// optional ones): the JS wrapper (js/wrap.mjs) fills the defaults. An old name (`createContext`, `solve`, `heat`, ...) is refused there with the new name.

EMSCRIPTEN_BINDINGS(nxr_compute_wasm) {
    // the surface as data
    emscripten::function("geometry.surfaceArrays", &surfaceArraysJS);
    // operators
    emscripten::function("operators.assembleManifoldOperators", &assembleManifoldOperatorsJS);
    emscripten::function("operators.assembleDECOperators", &assembleDECOperatorsJS);
    emscripten::function("operators.assembleConnectionLaplacian", &assembleConnectionLaplacianJS);
    emscripten::function("operators.eigen", &eigenJS);
    emscripten::function("operators.eigenOperator", &eigenOperatorJS);
    emscripten::function("operators.normalize", &normalizeJS);
    emscripten::function("operators.removeDC", &removeDCJS);
    emscripten::function("operators.poisson", &poissonJS);
    emscripten::function("operators.gradient", &gradientJS);
    emscripten::function("operators.trivial", &trivialJS);
    emscripten::function("operators.operatorById", &operatorByIdJS);
    emscripten::function("operators.fieldById", &fieldByIdJS);
    // signal
    emscripten::function("signal.hodge", &hodgeJS);
    emscripten::function("signal.isoline", &isolineJS);
    emscripten::function("signal.streamline", &streamlineJS);
    emscripten::function("signal.whitney", &whitneyJS);
    // geometry
    emscripten::function("geometry.curvatures", &curvaturesJS);
    emscripten::function("geometry.bff", &bffJS);
    emscripten::function("geometry.frames", &framesJS);
    emscripten::function("geometry.vertexFrames", &vertexFramesJS);
    emscripten::function("geometry.normals", &normalsJS);
    emscripten::function("geometry.heat", &heatJS);
    emscripten::function("geometry.tracePath", &tracePathJS);
    emscripten::function("geometry.exactDistance", &exactDistanceJS);
    emscripten::function("geometry.parallel", &parallelJS);
    emscripten::function("geometry.extendScalar", &extendScalarJS);
    emscripten::function("geometry.logMap", &logMapJS);
    emscripten::function("geometry.findCenter", &findCenterJS);
    emscripten::function("geometry.signedHeat", &signedHeatJS);
    emscripten::function("geometry.smoothFace", &smoothFaceJS);
    emscripten::function("geometry.smoothVertex", &smoothVertexJS);
    // synthesis
    emscripten::function("synthesis.delta", &deltaJS);
    emscripten::function("synthesis.heatDiffusion", &heatDiffusionJS);
    emscripten::function("synthesis.dampedWave", &dampedWaveJS);
    emscripten::function("synthesis.randomDecomposed1Form", &randomDecomposed1FormJS);

    emscripten::function("version", &getVersion);
}
