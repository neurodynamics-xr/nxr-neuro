/**
 * nxr_compute_mex.cpp — MATLAB MEX dispatcher around the nxr-compute library.
 *
 * Single mexFunction with command dispatch (gpjs / libigl-matlab style). One artifact (nxr_compute.mexw64) consumed by Brainstorm, SPM, or any MATLAB
 * analysis pipeline.
 *
 * THE COMMANDS ARE THE LIBRARY'S FUNCTIONS, by their toolbox name: `<toolbox>.<function>` (one name per thing across the C++ function, the node and
 * the bindings). Every command takes arrays and returns arrays; nothing is kept between calls (no context, no cache). A SURFACE is data: the struct
 * made once by `geometry.surfaceArrays` (positions, faces and the intrinsic Delaunay triangulation, the expensive step), passed as the first
 * argument to every surface command, which rebuilds the library's SurfaceArrays from it. A factorization a command needs is made inside it.
 *
 * Usage from MATLAB:
 *   S      = nxr_compute('geometry.surfaceArrays', V, F);               % once: the surface as data
 *   ops    = nxr_compute('operators.assembleManifoldOperators', S);
 *   eig    = nxr_compute('operators.eigen', ops.cotanLaplacian, ops.mass, k, [], true, true);   % K, M, k, sigma, normalize, removeDC
 *   d      = nxr_compute('geometry.heat', S, sourceVertices);           % a cell array of source sets gives one column per set
 *
 * V is Vx3 double; F is Fx3 (double / int32 / uint32) with 1-based indices (MATLAB convention). All sparse outputs are CSC double.
 *
 * An OLD command string (`heat`, `solve`, `create`, ...) is refused with an error naming the new name (no alias); the former context commands
 * (create, destroy, topology, geometry, gauge, bundle, operators, embedded, intrinsic, extrinsic, facets) are refused with the reason.
 *
 * Cancellation: long-running commands (operators.eigen) honour Ctrl-C in MATLAB via libut's utIsInterruptPending — the cancellation token polls it
 * from inside nxr-compute's solver and throws Error(Cancelled), which surfaces as MException 'nxr:cancelled'.
 */

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
#include "nxr/operators/registry/field_registry.h"
#include "nxr/operators/registry/operator_registry.h"
#include "first_slice.h"
#include "marshal.h"
#include "mex.h"

#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// libut is MATLAB's internal utilities lib. utIsInterruptPending()
// is the canonical Ctrl-C bridge used by libigl-matlab, gptoolbox,
// and others; it's been stable across MATLAB releases for 20+ years
// despite being undocumented. Linkage is automatic on Windows
// (libut.lib ships next to libmex.lib in the matlabroot/extern dir).
extern "C" bool utIsInterruptPending();

using namespace nxr::geometry::mex;

namespace {

namespace geom = nxr::geometry;
namespace ops = nxr::operators;
namespace cl = nxr::operators::laplacian::connection;

// Build a CancellationToken that fires when MATLAB user hits Ctrl-C.
// Wrapped in std::function so the same nxr::core::CancellationToken type
// covers both this and SAB-backed JS flags.
nxr::core::CancellationToken makeCtrlCToken() {
    return nxr::core::CancellationToken([]() {
        return utIsInterruptPending();
    });
}

// Lowercase first letter of an error-code name for MATLAB identifier
// convention: 'nxr:invalidInput' rather than 'nxr:INVALID_INPUT'.
// MATLAB's MException identifier rules require a lowercase first
// letter on each colon-segment.
std::string toMatlabIdentifier(nxr::core::ErrorCode code) {
    std::string name(nxr::core::errorCodeName(code));
    // Convert UPPER_SNAKE → camelCase: lowercase the first letter
    // and remove underscores, capitalizing the next char.
    std::string out;
    out.reserve(name.size());
    bool nextUpper = false;
    bool first = true;
    for (char c : name) {
        if (c == '_') { nextUpper = true; continue; }
        if (first) {
            out.push_back(static_cast<char>(std::tolower(c)));
            first = false;
        } else if (nextUpper) {
            out.push_back(c);  // already uppercase
            nextUpper = false;
        } else {
            out.push_back(static_cast<char>(std::tolower(c)));
        }
    }
    return out;
}

// ── argument helpers ─────────────────────────────────────────

// The number of arguments after the command string must be in [lo, hi].
void argCount(const char* cmd, int nrhs, int lo, int hi, const char* usage) {
    const int n = nrhs - 1;
    if (n < lo || n > hi)
        throw std::invalid_argument(std::string("nxr_compute('") + cmd + "', ...) takes " + (lo == hi ? std::to_string(lo) : std::to_string(lo) + " to " + std::to_string(hi)) +
                                    " arguments: " + usage);
}

bool flagArg(const mxArray* a) { return getIntArg(a) != 0; }

// Index sets: a numeric vector is one set; a cell array holds one numeric vector per set (the batch form). 1-based in, 0-based out.
bool isBatch(const mxArray* a) { return mxIsCell(a); }

std::vector<std::vector<int>> indexSets(const mxArray* a) {
    std::vector<std::vector<int>> out;
    const std::size_t n = mxGetNumberOfElements(a);
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i) out.push_back(mxToVertexIndices(mxGetCell(a, i)));
    return out;
}

// A vertex → value map from the 1-based vertices and their values (singularity indices, source densities).
std::map<int, double> vertexValueMap(const mxArray* verts, const mxArray* values, const char* what) {
    const std::vector<int> idx = mxToVertexIndices(verts);
    const Eigen::VectorXd val = mxToEigenVector(values);
    if (static_cast<std::size_t>(val.size()) != idx.size())
        throw std::invalid_argument(std::string(what) + ": the values must match the vertices in length");
    std::map<int, double> out;
    for (std::size_t i = 0; i < idx.size(); ++i) out[idx[i]] = val[static_cast<Eigen::Index>(i)];
    return out;
}

cl::ConnectionLaplacianOptions connectionOptions(const mxArray* opts, bool withDomain, cl::ConnectionLaplacianOptions o) {
    if (!opts || mxIsEmpty(opts)) return o;
    if (!mxIsStruct(opts)) throw std::invalid_argument("opts must be a struct");
    const mxArray* f;
    if (withDomain && (f = mxGetField(opts, 0, "domain"))) o.domain = cl::parseConnectionDomain(getStringArg(f));
    if ((f = mxGetField(opts, 0, "nSym")))                 o.nSym = getIntArg(f);
    if ((f = mxGetField(opts, 0, "regularization")))       o.regularization = getDoubleArg(f);
    if ((f = mxGetField(opts, 0, "format")))               o.format = cl::parseConnectionLaplacianFormat(getStringArg(f));
    return o;
}

mxArray* vector3ToMx(const Eigen::Vector3d& c) {
    mxArray* out = mxCreateDoubleMatrix(1, 3, mxREAL);
    double* op = mxGetPr(out);
    op[0] = c.x(); op[1] = c.y(); op[2] = c.z();
    return out;
}

// Validate a triangle mesh BEFORE handing it to geometry-central. geometry-central's
// ManifoldSurfaceMesh constructor crashes the host process (segfault) on malformed
// connectivity -- out-of-range or 0-based-underflowing face indices, degenerate faces,
// non-manifold edges, or inconsistent winding. We reject those here with a clean,
// catchable nxr:invalidInput error instead, so a bad mesh cannot take down MATLAB.
// Faces are 0-based at this point (mxToFaceBuffer already converted from MATLAB 1-based).
void validateMesh(int nV, int nF, const std::vector<std::int32_t>& faces) {
    if (nV <= 0 || nF <= 0)
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "geometry.surfaceArrays: empty mesh (nV=" + std::to_string(nV) + ", nF=" + std::to_string(nF) + ").");

    auto keyOf = [](std::int32_t a, std::int32_t b) -> std::uint64_t {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32)
             |  static_cast<std::uint64_t>(static_cast<std::uint32_t>(b));
    };
    std::unordered_map<std::uint64_t,int> undEdge;   // undirected edge -> incident-face count
    std::unordered_map<std::uint64_t,int> dirEdge;   // directed half-edge -> count

    for (int i = 0; i < nF; ++i) {
        std::int32_t a = faces[3*i], b = faces[3*i+1], c = faces[3*i+2];
        // (1) index range [0, nV)
        for (int j = 0; j < 3; ++j) {
            std::int32_t idx = faces[3*i + j];
            if (idx < 0 || idx >= nV)
                throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                    "geometry.surfaceArrays: face " + std::to_string(i) + " references vertex " +
                    std::to_string(idx) + " (valid range [0, " + std::to_string(nV-1) + "]).",
                    "Face indices must be within the vertex range (MATLAB passes 1-based; the binding converts to 0-based).");
        }
        // (2) degenerate triangle (a vertex repeats)
        if (a == b || b == c || c == a)
            throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                "geometry.surfaceArrays: face " + std::to_string(i) + " is degenerate (a vertex repeats within the triangle).");
        // (3) edge-manifold + (4) consistent orientation
        const std::int32_t tri[3][2] = {{a,b},{b,c},{c,a}};
        for (const auto& e : tri) {
            std::int32_t u = e[0], v = e[1];
            std::int32_t lo = (u < v) ? u : v, hi = (u < v) ? v : u;
            if (++undEdge[keyOf(lo,hi)] > 2)
                throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                    "geometry.surfaceArrays: non-manifold mesh -- an edge is shared by more than two faces.",
                    "geometry-central's ManifoldSurfaceMesh cannot represent a non-manifold edge.");
            if (++dirEdge[keyOf(u,v)] > 1)
                throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
                    "geometry.surfaceArrays: inconsistent face orientation -- a directed edge appears in two faces with the same winding.",
                    "Faces must be consistently oriented.");
        }
    }
}

// ── geometry.surfaceArrays(V, F[, opts]) → surface struct ─────
//
// DEFAULT intrinsic (geometry-central's intrinsic mollified Delaunay triangulation, nxr/geometry/intrinsic_surface.h); opts.intrinsicDelaunay = false is
// the named option "embedded". opts.mollify (relative delta, 1e-6) and opts.flipEps (1e-6) are the construction's recorded options; opts.surface is an id
// recorded in the gauge identity.

void cmdSurfaceArrays(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.surfaceArrays", nrhs, 2, 3, "V, F[, opts]");
    int nV = 0, nF = 0;
    auto verts = mxToVertexBuffer(prhs[1], nV);
    auto faces = mxToFaceBuffer(prhs[2], nF);
    validateMesh(nV, nF, faces);   // guard geometry-central against malformed meshes (segfault -> clean MATLAB error)

    bool intrinsicDelaunay = true;
    geom::DelaunayOptions dopt;
    std::string surfaceId;
    if (nrhs >= 4 && !mxIsEmpty(prhs[3])) {
        if (!mxIsStruct(prhs[3]))
            throw std::invalid_argument("nxr_compute('geometry.surfaceArrays', V, F, opts): opts must be a struct");
        const mxArray* f = mxGetField(prhs[3], 0, "intrinsicDelaunay");
        if (f && !mxIsEmpty(f)) intrinsicDelaunay = flagArg(f);
        const mxArray* fm = mxGetField(prhs[3], 0, "mollify");
        if (fm && !mxIsEmpty(fm)) dopt.mollify = mxGetScalar(fm);
        const mxArray* ff = mxGetField(prhs[3], 0, "flipEps");
        if (ff && !mxIsEmpty(ff)) dopt.flipEps = mxGetScalar(ff);
        const mxArray* fs = mxGetField(prhs[3], 0, "surface");
        if (fs && !mxIsEmpty(fs)) surfaceId = getStringArg(fs);
    }
    geom::IntrinsicOptions options = geom::intrinsicOptions(intrinsicDelaunay, dopt);
    options.surface = surfaceId;
    plhs[0] = surfaceToStruct(geom::surfaceArrays(verts.data(), nV, faces.data(), nF, options));
}

// ── operators.assembleManifoldOperators(surface[, massVariant]) → struct ─

void cmdAssembleManifoldOperators(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("operators.assembleManifoldOperators", nrhs, 1, 2, "surface[, massVariant]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    const ops::MassMatrixVariant variant = (nrhs >= 3) ? ops::parseMassMatrixVariant(getStringArg(prhs[2])) : ops::MassMatrixVariant::MixedVoronoi;
    auto o = ops::assembleManifoldOperators(surface, variant);
    const int nV = surface.nV(), nF = surface.nF();
    const int nE = nV + nF - geom::eulerCharacteristic(surface);   // χ = nV − nE + nF
    plhs[0] = meshOperatorsToStruct(o, nV, nE, nF);
}

// ── operators.eigen(K, M, k[, sigma[, normalize[, removeDC]]]) → struct ─

void cmdEigen(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("operators.eigen", nrhs, 3, 6, "K, M, k[, sigma[, normalize[, removeDC]]]  (sigma = [] for the shift ladder)");
    auto K = mxToEigenSparse(prhs[1]);
    auto M = mxToEigenSparse(prhs[2]);
    int k = getIntArg(prhs[3]);
    double sigma = (nrhs >= 5 && !mxIsEmpty(prhs[4])) ? getDoubleArg(prhs[4]) : ops::kShiftLadder;
    bool normalize = (nrhs >= 6) ? flagArg(prhs[5]) : false;
    bool removeDC  = (nrhs >= 7) ? flagArg(prhs[6]) : false;

    // Ctrl-C polling lives entirely in the token; nxr-compute doesn't know about MATLAB.
    auto result = ops::eigen(K, M, k, sigma, normalize, removeDC, makeCtrlCToken());
    plhs[0] = eigenResultToStruct(result);
}

// ── operators.normalize(U, M) → U ────────────────────────────

void cmdNormalize(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("operators.normalize", nrhs, 2, 2, "U, M");
    auto U = mxToEigenMatrix(prhs[1]);
    auto M = mxToEigenSparse(prhs[2]);
    plhs[0] = eigenMatrixToMx(ops::normalize(U, M));
}

// ── operators.removeDC(eigStruct) → eigStruct ────────────────

void cmdRemoveDC(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("operators.removeDC", nrhs, 1, 1, "eig");
    auto eig = mxToEigenResult(prhs[1]);
    plhs[0] = eigenResultToStruct(ops::removeDC(eig));
}

// ── geometry.fixDelaunay(V, F) → [V2, F2, nFlips]  (Delaunay edge-flip repair) ──
//
//   plhs[0] = V (passthrough via mxDuplicateArray — vertices unchanged)
//   plhs[1] = F2 (Fx3 double, 1-based)  — only when nlhs >= 2
//   plhs[2] = nFlips (scalar double)    — only when nlhs >= 3

void cmdFixDelaunay(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.fixDelaunay", nrhs, 2, 2, "V, F");
    int nV = 0, nF = 0;
    std::vector<double>        Vbuf = mxToVertexBuffer(prhs[1], nV);   // row-major xyz
    std::vector<std::int32_t>  Fbuf = mxToFaceBuffer(prhs[2], nF);     // 0-based

    auto r = geom::fixDelaunay(Vbuf.data(), nV, Fbuf.data(), nF);

    plhs[0] = mxDuplicateArray(prhs[1]);                         // V2 == V (passthrough)
    if (nlhs >= 2) {
        Eigen::MatrixXd Fd = r.faces.cast<double>().array() + 1.0;  // 0-based → 1-based
        plhs[1] = eigenMatrixToMx(Fd);                               // nF×3, column-major
    }
    if (nlhs >= 3) plhs[2] = scalarToMx(static_cast<double>(r.flips));
}

// ── Heat method ──────────────────────────────────────────────

geom::HeatMesh parseHeatMesh(const std::string& s) {
    if (s == "intrinsic") return geom::HeatMesh::Intrinsic;
    if (s == "input")     return geom::HeatMesh::Input;
    throw std::invalid_argument("mesh must be 'intrinsic' or 'input'");
}

// geometry.heat(surface, sources[, tCoef[, mesh]]): sources is a vertex vector (→ [nV x 1]) or a cell array of vertex vectors (→ [nV x nSets], one column per set).
void cmdHeat(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.heat", nrhs, 2, 4, "surface, sources[, tCoef[, mesh]]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    const double tCoef = (nrhs >= 4) ? getDoubleArg(prhs[3]) : 1.0;
    const geom::HeatMesh mesh = (nrhs >= 5) ? parseHeatMesh(getStringArg(prhs[4])) : geom::HeatMesh::Intrinsic;
    if (isBatch(prhs[2])) plhs[0] = eigenMatrixToMx(geom::heat(surface, indexSets(prhs[2]), tCoef, mesh));
    else plhs[0] = eigenVectorToMx(geom::heat(surface, mxToVertexIndices(prhs[2]), tCoef, mesh));
}

// geometry.signedHeat(surface, curve, isLoop[, levelSet[, tCoef]]): curve is a vertex vector (→ [nV x 1]) or a cell array of curves (→ [nV x nCurves]).
void cmdSignedHeat(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.signedHeat", nrhs, 3, 5, "surface, curve, isLoop[, levelSet[, tCoef]]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    const bool isLoop = flagArg(prhs[3]);
    const auto ls = static_cast<geom::SignedHeatLevelSet>((nrhs >= 5) ? getIntArg(prhs[4]) : static_cast<int>(geom::SignedHeatLevelSet::ZeroSet));
    const double tCoef = (nrhs >= 6) ? getDoubleArg(prhs[5]) : 1.0;
    if (isBatch(prhs[2])) plhs[0] = eigenMatrixToMx(geom::signedHeat(surface, indexSets(prhs[2]), isLoop, ls, tCoef));
    else plhs[0] = eigenVectorToMx(geom::signedHeat(surface, mxToVertexIndices(prhs[2]), isLoop, ls, tCoef));
}

// ── Vector heat method (Sharp, Soliman, Crane 2019) ─────────

Eigen::MatrixXd sourceVectors(const mxArray* a, std::size_t n) {
    Eigen::MatrixXd v = mxToEigenMatrix(a);
    if (v.cols() != 3 || static_cast<std::size_t>(v.rows()) != n)
        throw std::invalid_argument("sourceVectors must be Nx3 with N = numel(sourceVerts)");
    return v;
}

// geometry.parallel(surface, sourceVerts, sourceVectors[, tCoef]): sourceVerts and sourceVectors are one set (a vector, an Nx3 matrix → [nV x 3]) or cell
// arrays of sets (→ a cell array of [nV x 3]).
void cmdParallel(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.parallel", nrhs, 3, 4, "surface, sourceVerts, sourceVectors[, tCoef]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    const double tCoef = (nrhs >= 5) ? getDoubleArg(prhs[4]) : 1.0;
    if (isBatch(prhs[2]) != isBatch(prhs[3]))
        throw std::invalid_argument("sourceVerts and sourceVectors are both cell arrays (the batch form) or both plain arrays");
    if (isBatch(prhs[2])) {
        const auto sets = indexSets(prhs[2]);
        if (mxGetNumberOfElements(prhs[3]) != sets.size())
            throw std::invalid_argument("sourceVectors holds one matrix per set of sourceVerts");
        std::vector<Eigen::MatrixXd> vecs;
        for (std::size_t i = 0; i < sets.size(); ++i) vecs.push_back(sourceVectors(mxGetCell(prhs[3], i), sets[i].size()));
        const auto out = geom::transport::parallel(surface, sets, vecs, tCoef);
        mxArray* cells = mxCreateCellMatrix(1, out.size());
        for (std::size_t i = 0; i < out.size(); ++i) mxSetCell(cells, i, eigenMatrixToMx(out[i]));
        plhs[0] = cells;
        return;
    }
    const auto srcIdx = mxToVertexIndices(prhs[2]);   // 1-based → 0-based
    plhs[0] = eigenMatrixToMx(geom::transport::parallel(surface, srcIdx, sourceVectors(prhs[3], srcIdx.size()), tCoef));
}

void cmdExtendScalar(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.extendScalar", nrhs, 3, 4, "surface, sourceVerts, sourceValues[, tCoef]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    auto srcIdx = mxToVertexIndices(prhs[2]);
    auto srcVal = mxToEigenVector(prhs[3]);
    if (static_cast<std::size_t>(srcVal.size()) != srcIdx.size())
        throw std::invalid_argument("sourceValues must have the same length as sourceVerts");
    const double tCoef = (nrhs >= 5) ? getDoubleArg(prhs[4]) : 1.0;
    plhs[0] = eigenVectorToMx(geom::transport::extendScalar(surface, srcIdx, srcVal, tCoef));
}

// geometry.logMap(surface, sourceVertices[, strategy[, tCoef]]) → a 1 x nSources struct array (logCoords, sourceE1, sourceE2); one source gives a scalar struct.
void cmdLogMap(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.logMap", nrhs, 2, 4, "surface, sourceVertices[, strategy[, tCoef]]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    const auto sources = mxToVertexIndices(prhs[2]);
    const auto strategy = static_cast<geom::transport::LogMapStrategy>(
        (nrhs >= 4) ? getIntArg(prhs[3]) : static_cast<int>(geom::transport::LogMapStrategy::AffineLocal));
    const double tCoef = (nrhs >= 5) ? getDoubleArg(prhs[4]) : 1.0;
    const auto results = geom::transport::logMap(surface, sources, strategy, tCoef);
    const char* fields[] = {"logCoords", "sourceE1", "sourceE2"};
    mxArray* s = mxCreateStructMatrix(1, results.size(), 3, fields);
    for (std::size_t i = 0; i < results.size(); ++i) {
        mxSetField(s, i, "logCoords", eigenMatrixToMx(results[i].logCoords));
        mxSetField(s, i, "sourceE1", vector3ToMx(results[i].sourceE1));
        mxSetField(s, i, "sourceE2", vector3ToMx(results[i].sourceE2));
    }
    plhs[0] = s;
}

void cmdFindCenter(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.findCenter", nrhs, 2, 4, "surface, sourceVerts[, p[, tCoef]]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    auto srcIdx = mxToVertexIndices(prhs[2]);
    int p = (nrhs >= 4) ? getIntArg(prhs[3]) : 2;
    const double tCoef = (nrhs >= 5) ? getDoubleArg(prhs[4]) : 1.0;
    plhs[0] = vector3ToMx(geom::transport::findCenter(surface, srcIdx, p, tCoef));
}

// ── Smooth direction fields (Knöppel-Crane) ─────────────────

void cmdSmoothFace(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.smoothFace", nrhs, 1, 3, "surface[, nSym[, alignToCurvature]]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    int  nSym  = (nrhs >= 3) ? getIntArg(prhs[2]) : 4;
    bool align = (nrhs >= 4) ? flagArg(prhs[3]) : false;
    plhs[0] = eigenMatrixToMx(geom::connection::smoothFace(surface, nSym, align));   // F×3
}

void cmdSmoothVertex(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.smoothVertex", nrhs, 1, 3, "surface[, nSym[, alignToCurvature]]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    int  nSym  = (nrhs >= 3) ? getIntArg(prhs[2]) : 2;
    bool align = (nrhs >= 4) ? flagArg(prhs[3]) : false;
    auto r = geom::connection::smoothVertex(surface, nSym, align);
    const char* fields[] = {"vertexVectors", "vertexFieldRaw", "nSym"};
    mxArray* s = mxCreateStructMatrix(1, 1, 3, fields);
    mxSetField(s, 0, "vertexVectors",  eigenMatrixToMx(r.vertexVectors));
    mxSetField(s, 0, "vertexFieldRaw", eigenVectorToMx(r.vertexFieldRaw));
    mxSetField(s, 0, "nSym",           mxCreateDoubleScalar(r.nSym));
    plhs[0] = s;
}

// ── Stripe patterns (Knöppel-Crane SIGGRAPH 2015) ───────────
//
// Returns positions as a (2*segCount)x3 matrix of endpoint pairs.
// MATLAB users render with line() / patch() over consecutive pairs.

mxArray* stripePatternResultToStruct(const geom::parametrization::stripes::StripePatternResult& r) {
    return positionsSegmentsToStruct(r.positions, r.segmentCount);
}

void cmdStripes(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.compute", nrhs, 3, 4, "surface, vertexFieldRaw, frequency[, connect]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    Eigen::VectorXd raw = mxToEigenVector(prhs[2]);
    double freq = getDoubleArg(prhs[3]);
    bool connect = (nrhs >= 5) ? flagArg(prhs[4]) : true;
    plhs[0] = stripePatternResultToStruct(geom::parametrization::stripes::compute(surface, raw, freq, connect));
}

void cmdStripesFreq(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.computeFreq", nrhs, 3, 4, "surface, vertexFieldRaw, frequencies[, connect]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    Eigen::VectorXd raw   = mxToEigenVector(prhs[2]);
    Eigen::VectorXd freqs = mxToEigenVector(prhs[3]);
    bool connect = (nrhs >= 5) ? flagArg(prhs[4]) : true;
    plhs[0] = stripePatternResultToStruct(geom::parametrization::stripes::computeFreq(surface, raw, freqs, connect));
}

// ── operators: assembly ──────────────────────────────────────

void cmdAssembleDECOperators(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("operators.assembleDECOperators", nrhs, 1, 1, "surface");
    plhs[0] = decOperatorsToStruct(ops::assembleDECOperators(structToSurface(prhs[1])));
}

void cmdAssembleConnectionLaplacian(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("operators.assembleConnectionLaplacian", nrhs, 1, 2, "surface[, opts]  (opts: domain, nSym, regularization, format)");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    cl::ConnectionLaplacianOptions o;   // value-init → vertex / nSym=1 / 1e-8 / Real2N
    o = connectionOptions((nrhs >= 3) ? prhs[2] : nullptr, true, o);
    plhs[0] = connectionLaplacianToStruct(cl::assembleConnectionLaplacian(surface, o));
}

// operators.assembleTrivialConnectionLaplacian(surface, singVerts, singValues[, opts]): the singularity indices sum to the Euler characteristic.
// Complex format by default (the primary MATLAB consumer calls eigs on K_complex).
void cmdAssembleTrivialConnectionLaplacian(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("operators.assembleTrivialConnectionLaplacian", nrhs, 3, 4,
             "surface, singVerts, singValues[, opts]\n"
             "  singVerts : 1-based vertex indices of singularities\n"
             "  singValues: corresponding singularity indices (sum must equal Euler characteristic)\n"
             "  opts      : optional struct with fields nSym (default 1), regularization (default 1e-8), format ('complex'|'real2N')");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    const std::map<int, double> sing = vertexValueMap(prhs[2], prhs[3], "operators.assembleTrivialConnectionLaplacian");
    cl::ConnectionLaplacianOptions o;
    o.domain = cl::ConnectionDomain::Vertex;
    o.format = cl::ConnectionLaplacianFormat::Complex;
    o = connectionOptions((nrhs >= 5) ? prhs[4] : nullptr, false, o);
    const ops::DECOperators dec = ops::assembleDECOperators(surface);   // the operator mesh (intrinsic triangulation), as before the surface struct
    const std::unique_ptr<ops::CholeskyFactor> exact = ops::exactFormFactor(dec);
    plhs[0] = connectionLaplacianToStruct(cl::assembleTrivialConnectionLaplacian(surface, sing, dec, *exact, o));
}

// ── geometry: frames, normals, curvature ─────────────────────

void cmdFrames(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.frames", nrhs, 1, 1, "surface");
    plhs[0] = faceFramesToStruct(geom::frames(structToSurface(prhs[1])));
}

void cmdVertexFrames(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.vertexFrames", nrhs, 1, 1, "surface");
    plhs[0] = vertexFramesToStruct(geom::vertexFrames(structToSurface(prhs[1])));
}

geom::NormalType parseNormalType(const std::string& s) {
    using NT = geom::NormalType;
    if (s == "angle")  return NT::AngleWeighted;
    if (s == "area")   return NT::AreaWeighted;
    if (s == "equal")  return NT::EqualWeighted;
    if (s == "sphere") return NT::SphereInscribed;
    if (s == "mean")   return NT::MeanCurvature;
    if (s == "gauss")  return NT::GaussCurvature;
    throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
        "unknown normal type \"" + s + "\" (use angle|area|equal|sphere|mean|gauss)");
}

void cmdNormals(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.normals", nrhs, 1, 2, "surface[, type]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    auto type = geom::NormalType::AngleWeighted;
    if (nrhs >= 3) type = parseNormalType(getStringArg(prhs[2]));
    plhs[0] = eigenMatrixToMx(geom::normals(surface, type));
}

void cmdCurvatures(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.curvatures", nrhs, 1, 1, "surface");
    plhs[0] = curvatureResultToStruct(geom::curvatures(structToSurface(prhs[1])));
}

void cmdBff(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.bff", nrhs, 1, 1, "surface");
    plhs[0] = eigenMatrixToMx(geom::parametrization::bff(structToSurface(prhs[1])));   // [V, 2]
}

void cmdTracePath(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("geometry.tracePath", nrhs, 3, 3, "surface, vStart, vEnd");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    int a = getIntArg(prhs[2]) - 1;   // 1-based → 0-based
    int b = getIntArg(prhs[3]) - 1;
    plhs[0] = eigenMatrixToMx(geom::query::tracePath(surface, a, b));
}

// ── operators: solvers ───────────────────────────────────────

// operators.poisson(K, M, sourceVerts, sourceValues): the Laplacian's factor is made inside, from K.
void cmdPoisson(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("operators.poisson", nrhs, 4, 4, "K, M, sourceVerts, sourceValues");
    auto K = mxToEigenSparse(prhs[1]);
    auto M = mxToEigenSparse(prhs[2]);
    const std::map<int, double> density = vertexValueMap(prhs[3], prhs[4], "operators.poisson");
    const std::unique_ptr<ops::CholeskyFactor> factor = ops::laplacianFactor(K);
    plhs[0] = eigenVectorToMx(ops::poisson(K, M, *factor, density));
}

// signal.hodge(surface, omega): omega is a 1-form on the OPERATOR (intrinsic triangulation) mesh's edges; the DEC and the two factorizations are made inside.
void cmdHodge(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("signal.hodge", nrhs, 2, 2, "surface, omega");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    Eigen::VectorXd omega = mxToEigenVector(prhs[2]);
    const ops::DECOperators dec = ops::assembleDECOperators(surface);
    const std::unique_ptr<ops::CholeskyFactor> exact = ops::exactFormFactor(dec);
    const std::unique_ptr<ops::SquareSolver> coexact = ops::coexactFormSolver(dec);
    plhs[0] = hodgeResultToStruct(nxr::signal::hodge(surface, dec, *exact, *coexact, omega));
}

void cmdIsoline(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("signal.isoline", nrhs, 2, 5, "surface, scalarField[, numLevels[, minVal[, maxVal]]]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    Eigen::VectorXd field = mxToEigenVector(prhs[2]);
    int    numLevels = (nrhs >= 4) ? getIntArg(prhs[3])    : 20;
    double minV      = (nrhs >= 5) ? getDoubleArg(prhs[4]) : 0.0;
    double maxV      = (nrhs >= 6) ? getDoubleArg(prhs[5]) : 0.0;
    auto r = nxr::signal::isoline(surface, field, numLevels, minV, maxV);
    plhs[0] = positionsSegmentsToStruct(r.positions, r.segmentCount);
}

// operators.trivial(surface, singVerts, singValues): a smooth direction field with the prescribed singularities (trivial connections).
void cmdTrivial(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("operators.trivial", nrhs, 3, 3, "surface, singVerts, singValues");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    const std::map<int, double> sing = vertexValueMap(prhs[2], prhs[3], "operators.trivial");
    const ops::DECOperators dec = ops::assembleDECOperators(surface);
    const std::unique_ptr<ops::CholeskyFactor> exact = ops::exactFormFactor(dec);
    plhs[0] = directionFieldResultToStruct(ops::connection::trivial(surface, dec, *exact, sing));
}

void cmdStreamline(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("signal.streamline", nrhs, 2, 5, "surface, faceField[, numSeeds[, stepCoef[, maxSteps]]]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    Eigen::MatrixXd faceField = mxToEigenMatrix(prhs[2]);   // [nF, 3]
    int    numSeeds = (nrhs >= 4) ? getIntArg(prhs[3])    : 15;
    double stepCoef = (nrhs >= 5) ? getDoubleArg(prhs[4]) : 0.15;
    int    maxSteps = (nrhs >= 6) ? getIntArg(prhs[5])    : 1000;
    auto r = nxr::signal::streamline(surface, faceField, numSeeds, stepCoef, maxSteps);
    plhs[0] = positionsSegmentsToStruct(r.positions, r.segmentCount);
}

// signal.whitney(surface, oneForm): a 1-form on the OPERATOR (intrinsic triangulation) mesh's edges to face vectors.
void cmdWhitney(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("signal.whitney", nrhs, 2, 2, "surface, oneForm");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    Eigen::VectorXd oneForm = mxToEigenVector(prhs[2]);
    plhs[0] = eigenMatrixToMx(nxr::signal::whitney(surface, ops::assembleDECOperators(surface), oneForm));
}

void cmdGradient(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("operators.gradient", nrhs, 2, 2, "surface, scalarField");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    plhs[0] = eigenMatrixToMx(ops::gradient(surface, mxToEigenVector(prhs[2])));
}

// ── synthesis: field generators ──────────────────────────────

// synthesis.delta(nV, sourceVerts, sourceValues) → [nV x 1]: zero except at the sources.
void cmdDelta(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("synthesis.delta", nrhs, 3, 3, "nV, sourceVerts, sourceValues");
    plhs[0] = eigenVectorToMx(nxr::synthesis::delta(getIntArg(prhs[1]), vertexValueMap(prhs[2], prhs[3], "synthesis.delta")));
}

// synthesis.heatDiffusion(M, eig, u0, timesteps[, alpha]) → [T x nV]
void cmdHeatDiffusion(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("synthesis.heatDiffusion", nrhs, 4, 5, "M, eig, u0, timesteps[, alpha]");
    auto M = mxToEigenSparse(prhs[1]);
    auto eig = mxToEigenResult(prhs[2]);
    Eigen::VectorXd u0 = mxToEigenVector(prhs[3]);
    Eigen::VectorXd ts = mxToEigenVector(prhs[4]);
    double alpha = (nrhs >= 6) ? getDoubleArg(prhs[5]) : 1.0;
    std::vector<double> timesteps(ts.data(), ts.data() + ts.size());
    plhs[0] = eigenMatrixXfToMx(nxr::synthesis::heatDiffusion(M, eig, u0, timesteps, alpha));
}

// synthesis.dampedWave(eig, modeIndices, amplitudes, dampings, phases, timesteps) → [T x nV]
void cmdDampedWave(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("synthesis.dampedWave", nrhs, 6, 6, "eig, modeIndices, amplitudes, dampings, phases, timesteps");
    auto eig = mxToEigenResult(prhs[1]);
    auto modeIdx = mxToVertexIndices(prhs[2]);   // 1-based → 0-based mode columns
    auto toVec = [](const mxArray* a) {
        auto v = mxToEigenVector(a);
        return std::vector<double>(v.data(), v.data() + v.size());
    };
    plhs[0] = eigenMatrixXfToMx(nxr::synthesis::dampedWave(eig, modeIdx, toVec(prhs[3]), toVec(prhs[4]), toVec(prhs[5]), toVec(prhs[6])));
}

// synthesis.randomDecomposed1Form(surface, alphaStr, betaStr, gammaStr[, seed]) → a 1-form on the OPERATOR (intrinsic triangulation) mesh's edges
void cmdRandomDecomposed1Form(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("synthesis.randomDecomposed1Form", nrhs, 4, 5, "surface, alphaStr, betaStr, gammaStr[, seed]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    const ops::DECOperators dec = ops::assembleDECOperators(surface);
    double aS = getDoubleArg(prhs[2]);
    double bS = getDoubleArg(prhs[3]);
    double gS = getDoubleArg(prhs[4]);
    unsigned int seed = (nrhs >= 6) ? static_cast<unsigned int>(getIntArg(prhs[5])) : 42u;
    plhs[0] = eigenVectorToMx(nxr::synthesis::randomDecomposed1Form(
        dec, surface.nV(), static_cast<int>(dec.d0.rows()), surface.nF(), aS, bS, gS, seed));
}

// ── Covariant differential: frameTransport / liftToWorld / liftToFrame ──
//
// Thin wrappers around nxr::operators::differential::* that handle 1-based
// MATLAB index conversion and [nV×3] matrix marshalling via the existing
// mxToEigenMatrix / eigenMatrixToMx helpers from marshal.h.

// nxr_compute('operators.frameTransport', surface, i, j) -> [3×3]   (i,j are 1-based)
void cmdFrameTransport(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("operators.frameTransport", nrhs, 3, 3, "surface, i, j");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    int i = static_cast<int>(mxGetScalar(prhs[2])) - 1;   // 1-based -> 0-based
    int j = static_cast<int>(mxGetScalar(prhs[3])) - 1;
    plhs[0] = eigenMatrixToMx(ops::differential::frameTransport(surface, i, j));
}

// nxr_compute('operators.liftToWorld', surface, Lloc[nV×3]) -> [nV×3]
void cmdLiftToWorld(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("operators.liftToWorld", nrhs, 2, 2, "surface, field[nV x 3]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    plhs[0] = eigenMatrixToMx(ops::differential::liftToWorld(surface, mxToEigenMatrix(prhs[2])));
}

// nxr_compute('operators.liftToFrame', surface, Lworld[nV×3]) -> [nV×3]
void cmdLiftToFrame(int, mxArray** plhs, int nrhs, const mxArray** prhs) {
    argCount("operators.liftToFrame", nrhs, 2, 2, "surface, field[nV x 3]");
    const geom::SurfaceArrays surface = structToSurface(prhs[1]);
    plhs[0] = eigenMatrixToMx(ops::differential::liftToFrame(surface, mxToEigenMatrix(prhs[2])));
}

// ── version() → string ───────────────────────────────────────

void cmdVersion(int /*nlhs*/, mxArray** plhs,
                int /*nrhs*/, const mxArray** /*prhs*/) {
    plhs[0] = mxCreateString("nxr-compute 0.1.0");
}

// ── operators.operatorById(id) → struct of controlled-vocab metadata strings ─
//
// nxr_compute('operators.operatorById', 'laplaceBeltrami')  →  struct with fields:
//   id, label, bundle, holonomy, order, role, field_type, domain, singular,
//   gauge, coupling, natural_mass, graded, tau_presets, status, notes,
//   squares_to, square_of, relation
//
// All string fields. 'graded' is returned as a double scalar (1 or 0) for
// MATLAB compatibility.  squares_to / square_of / relation are empty strings
// when no cross-link is present.

void cmdOperatorById(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    using namespace nxr::operators::registry;
    if (nrhs < 2 || !mxIsChar(prhs[1]))
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "operatorById requires an id string.");
    char buf[128];
    mxGetString(prhs[1], buf, sizeof(buf));
    const OperatorVariant* v = operatorById(buf);
    if (!v)
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            std::string("operatorById: unknown operator id: ") + buf);

    const char* fields[] = {
        "id", "label", "bundle", "holonomy", "order", "role",
        "field_type", "domain", "singular", "gauge", "coupling",
        "natural_mass", "graded", "tau_presets", "status", "notes",
        "squares_to", "square_of", "relation",
        "input_field", "output_field"
    };
    mxArray* s = mxCreateStructMatrix(1, 1,
        static_cast<int>(sizeof(fields) / sizeof(*fields)), fields);

    mxSetField(s, 0, "id",           mxCreateString(v->id.c_str()));
    mxSetField(s, 0, "label",        mxCreateString(v->label.c_str()));
    mxSetField(s, 0, "bundle",       mxCreateString(toString(v->bundle)));
    mxSetField(s, 0, "holonomy",     mxCreateString(toString(v->holonomy)));
    mxSetField(s, 0, "order",        mxCreateString(toString(v->order)));
    mxSetField(s, 0, "role",         mxCreateString(toString(v->role)));
    mxSetField(s, 0, "field_type",   mxCreateString(toString(v->field_type)));
    mxSetField(s, 0, "domain",       mxCreateString(toString(v->domain)));
    mxSetField(s, 0, "singular",     mxCreateString(toString(v->singular)));
    mxSetField(s, 0, "gauge",        mxCreateString(toString(v->gauge)));
    mxSetField(s, 0, "coupling",     mxCreateString(toString(v->coupling)));
    mxSetField(s, 0, "natural_mass", mxCreateString(v->natural_mass.c_str()));
    mxSetField(s, 0, "graded",       mxCreateDoubleScalar(v->graded ? 1.0 : 0.0));
    mxSetField(s, 0, "tau_presets",  mxCreateString(v->tau_presets.c_str()));
    mxSetField(s, 0, "status",       mxCreateString(toString(v->status)));
    mxSetField(s, 0, "notes",        mxCreateString(v->notes.c_str()));
    mxSetField(s, 0, "squares_to",   mxCreateString(
        (v->square.present &&  v->square.isSquaresTo) ? v->square.target.c_str() : ""));
    mxSetField(s, 0, "square_of",    mxCreateString(
        (v->square.present && !v->square.isSquaresTo) ? v->square.target.c_str() : ""));
    mxSetField(s, 0, "relation",     mxCreateString(
        v->square.present ? toString(v->square.relation) : ""));
    mxSetField(s, 0, "input_field",  mxCreateString(v->input_field.c_str()));
    mxSetField(s, 0, "output_field", mxCreateString(v->output_field.c_str()));
    plhs[0] = s;
}

// ── operators.fieldById(id) → struct of controlled-vocab metadata strings ───
//
// nxr_compute('operators.fieldById', 'tangentVertex')  →  struct with fields:
//   id, label, domain, bundle, field_type, n_form, representation, gauge,
//   nSym, notes
//
// All string fields except nSym (returned as double scalar for MATLAB compat).
// Returns nxr:invalidInput on unknown id.

void cmdFieldById(int /*nlhs*/, mxArray** plhs, int nrhs, const mxArray** prhs) {
    using namespace nxr::operators::registry;
    if (nrhs < 2 || !mxIsChar(prhs[1]))
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            "fieldById requires an id string.");
    char buf[128];
    mxGetString(prhs[1], buf, sizeof(buf));
    const FieldVariant* v = fieldById(buf);
    if (!v)
        throw nxr::core::Error(nxr::core::ErrorCode::InvalidInput,
            std::string("fieldById: unknown field id: ") + buf);

    const char* fields[] = {
        "id", "label", "domain", "bundle", "field_type", "n_form",
        "representation", "gauge", "nSym", "notes"
    };
    mxArray* s = mxCreateStructMatrix(1, 1,
        static_cast<int>(sizeof(fields) / sizeof(*fields)), fields);

    const FieldDescriptor& d = v->descriptor;
    mxSetField(s, 0, "id",             mxCreateString(v->id.c_str()));
    mxSetField(s, 0, "label",          mxCreateString(v->label.c_str()));
    mxSetField(s, 0, "domain",         mxCreateString(toString(d.domain)));
    mxSetField(s, 0, "bundle",         mxCreateString(toString(d.bundle)));
    mxSetField(s, 0, "field_type",     mxCreateString(toString(d.field_type)));
    mxSetField(s, 0, "n_form",         mxCreateString(toString(d.n_form)));
    mxSetField(s, 0, "representation", mxCreateString(toString(d.representation)));
    mxSetField(s, 0, "gauge",          mxCreateString(toString(d.gauge)));
    mxSetField(s, 0, "nSym",          mxCreateDoubleScalar((double)d.nSym));
    mxSetField(s, 0, "notes",          mxCreateString(v->notes.c_str()));
    plhs[0] = s;
}

} // namespace

// ── mexFunction entry point ──────────────────────────────────

void mexFunction(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
    if (nrhs < 1 || !mxIsChar(prhs[0])) {
        mexErrMsgIdAndTxt("nxr:nrhs",
            "First argument must be a command string. Try nxr_compute('version').");
    }

    std::string cmd;
    try {
        cmd = getStringArg(prhs[0]);
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("nxr:badCommand", "%s", e.what());
        return;
    }

    try {
        // the first slice (the Python wheel's functions): "compute.*" and "io.*" commands, called by the MATLAB package +nxr/+ops and +nxr/+io
        if (nxr::mex::dispatchFirstSlice(cmd, nlhs, plhs, nrhs, prhs)) return;
#ifdef NXR_MEX_WITH_PIPELINES
        if (nxr::mex::dispatchPipeline(cmd, nlhs, plhs, nrhs, prhs)) return;   // "run.*": nxr.run, nxr.list, nxr.read
#endif

        // an old command string is refused with its new name (no alias); a removed context command with the reason
        std::string refusal;
        if (nxr::mex::refusedCommand(cmd, refusal)) throw nxr::mex::SliceFailure{"nxr:renamed", refusal};

        if      (cmd == "geometry.surfaceArrays")                      cmdSurfaceArrays(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.assembleManifoldOperators")         cmdAssembleManifoldOperators(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.eigen")                             cmdEigen(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.normalize")                         cmdNormalize(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.fixDelaunay")                        cmdFixDelaunay(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.removeDC")                          cmdRemoveDC(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.parallel")                           cmdParallel(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.extendScalar")                       cmdExtendScalar(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.logMap")                             cmdLogMap(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.findCenter")                         cmdFindCenter(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.signedHeat")                         cmdSignedHeat(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.smoothFace")                         cmdSmoothFace(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.smoothVertex")                       cmdSmoothVertex(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.compute")                            cmdStripes(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.computeFreq")                        cmdStripesFreq(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.assembleDECOperators")              cmdAssembleDECOperators(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.assembleConnectionLaplacian")       cmdAssembleConnectionLaplacian(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.assembleTrivialConnectionLaplacian") cmdAssembleTrivialConnectionLaplacian(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.frames")                             cmdFrames(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.vertexFrames")                       cmdVertexFrames(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.normals")                            cmdNormals(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.poisson")                           cmdPoisson(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.heat")                               cmdHeat(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.tracePath")                          cmdTracePath(nlhs, plhs, nrhs, prhs);
        else if (cmd == "signal.hodge")                                cmdHodge(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.curvatures")                         cmdCurvatures(nlhs, plhs, nrhs, prhs);
        else if (cmd == "geometry.bff")                                cmdBff(nlhs, plhs, nrhs, prhs);
        else if (cmd == "signal.isoline")                              cmdIsoline(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.trivial")                           cmdTrivial(nlhs, plhs, nrhs, prhs);
        else if (cmd == "signal.streamline")                           cmdStreamline(nlhs, plhs, nrhs, prhs);
        else if (cmd == "signal.whitney")                              cmdWhitney(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.gradient")                          cmdGradient(nlhs, plhs, nrhs, prhs);
        else if (cmd == "synthesis.delta")                             cmdDelta(nlhs, plhs, nrhs, prhs);
        else if (cmd == "synthesis.heatDiffusion")                     cmdHeatDiffusion(nlhs, plhs, nrhs, prhs);
        else if (cmd == "synthesis.dampedWave")                        cmdDampedWave(nlhs, plhs, nrhs, prhs);
        else if (cmd == "synthesis.randomDecomposed1Form")             cmdRandomDecomposed1Form(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.frameTransport")                    cmdFrameTransport(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.liftToWorld")                       cmdLiftToWorld(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.liftToFrame")                       cmdLiftToFrame(nlhs, plhs, nrhs, prhs);
        else if (cmd == "version")                                     cmdVersion(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.operatorById")                      cmdOperatorById(nlhs, plhs, nrhs, prhs);
        else if (cmd == "operators.fieldById")                         cmdFieldById(nlhs, plhs, nrhs, prhs);
        else {
            mexErrMsgIdAndTxt("nxr:unknownCommand",
                "Unknown command: \"%s\". Available: geometry.surfaceArrays (the surface as data, passed to every surface command), "
                "geometry.{fixDelaunay, parallel, extendScalar, logMap, findCenter, signedHeat, heat, smoothFace, smoothVertex, compute, computeFreq, "
                "frames, vertexFrames, normals, curvatures, bff, tracePath}, "
                "operators.{assembleManifoldOperators, assembleDECOperators, assembleConnectionLaplacian, assembleTrivialConnectionLaplacian, eigen, "
                "normalize, removeDC, poisson, trivial, gradient, frameTransport, liftToWorld, liftToFrame, operatorById, fieldById}, "
                "signal.{hodge, isoline, streamline, whitney}, synthesis.{delta, heatDiffusion, dampedWave, randomDecomposed1Form}, version; "
                "and the commands named compute.* and io.* (use nxr.compute and nxr.io).",
                cmd.c_str());
        }
    } catch (const nxr::mex::SliceFailure& e) {
        mexErrMsgIdAndTxt(e.id.c_str(), "%s", e.text.c_str());
    } catch (const nxr::core::Error& e) {
        // Route structured nxr-compute errors to MException identifiers MATLAB
        // can pattern-match (e.g. ME.identifier == "nxr:cancelled").
        std::string id = "nxr:" + toMatlabIdentifier(e.code());
        if (e.hint().empty()) {
            mexErrMsgIdAndTxt(id.c_str(), "nxr_compute('%s'): %s", cmd.c_str(), e.what());
        } else {
            std::string hint(e.hint());
            mexErrMsgIdAndTxt(id.c_str(), "nxr_compute('%s'): %s\n  hint: %s",
                              cmd.c_str(), e.what(), hint.c_str());
        }
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("nxr:error", "nxr_compute('%s') failed: %s", cmd.c_str(), e.what());
    }
}
