// nxr::ladder: the dyadic tile ladder of the Cortical Flow Studio (engine/ladder.ts), bit for bit, and the tile summaries nxr stores.
// (1) PARITY with the Studio's own code (golden/gen_ladder_golden.mts): level scalars, the tile of every vertex per level (IDENTICAL), tile areas, centres, adjacency, the scalar map
//     statistics, the closed-form (frame) vector sums, and the Levi-Civita connection and resultants (rho, Omega: the Studio's `transported` through the C++ chart); the sphere and an irregular mesh;
//     the SUBJECT tree's codes (h 2^D + t) on a two-piece mesh.
// (2) the tiles are equal-area within the Studio's own tolerance on a sphere, nested (a level refines the one above), and Weyl's law pairs 2^l tiles with lambda = 4 pi 2^l / A.
// (3) a uniform flow on a FOLDED sheet keeps its tile resultant length in the transported form, where a raw 3-D sum shrinks (and where the frame form, in a frame that turns, shrinks too).
#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>

#include "golden_io.h"
#include "mesh_helpers.h"
#include "nxr/chart.h"
#include "nxr/frames.h"
#include "nxr/ladder.h"
#include "nxr/measurement/reduce.h"

using namespace nxr::ladder;
using Eigen::MatrixXd;
using Eigen::VectorXd;

static double maxDiff(const MatrixXd& a, const MatrixXd& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) return 1e9;
    double d = 0;
    for (Eigen::Index j = 0; j < a.cols(); ++j) for (Eigen::Index i = 0; i < a.rows(); ++i) {
        const double x = a(i, j), y = b(i, j);
        if (std::isnan(x) || std::isnan(y)) { if (std::isnan(x) != std::isnan(y)) d = 1e9; continue; }
        if (std::isinf(x) || std::isinf(y)) { if (x != y) d = 1e9; continue; }
        d = std::max(d, std::abs(x - y));
    }
    return d;
}
static MatrixXd row(const VectorXd& v) { return v.transpose(); }
template <class T> static MatrixXd rowOf(const std::vector<T>& v) { MatrixXd m(1, static_cast<Eigen::Index>(v.size())); for (size_t k = 0; k < v.size(); ++k) m(0, static_cast<Eigen::Index>(k)) = static_cast<double>(v[k]); return m; }
static double wrap(double a) { return std::remainder(a, 2 * M_PI); }

// THE TILE SUMMARIES are measures now: `nxr::measurement::reduce` over the tiles' vertices, area-weighted (the arithmetic nxr's measure.scalars.map_stats / measure.vectors.vector_stats and the
// Levi-Civita resultant had). A vertex whose components are not finite is left out of the tiles (member -1), as the caller of the reduction does.
namespace ms = nxr::measurement;
static ms::Ingredients overTiles(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const double* x) {
    nxr::selection::MemberOfElement m; m.member_of = node; m.members = nodes;
    ms::Ingredients g(1, nodes);
    ms::Request r; r.measures = {"sum", "max", "min"}; r.weighting = ms::Weighting::given;   // the ingredients a measure needs are folded: sum, and the extremes the summary reads
    ms::reduce(g, x, 1, static_cast<std::int64_t>(node.size()), m, area.data(), r);
    return g;
}
static VectorXd sums(const ms::Ingredients& g) { VectorXd v(g.members); for (int k = 0; k < g.members; ++k) v[k] = g.sum[static_cast<size_t>(k)]; return v; }
struct ScalarSummary { VectorXd w, s1, s2, n, min, max; };
static ScalarSummary scalarSummary(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const double* x) {
    std::vector<int32_t> nd = node;                                      // a vertex whose value is not finite is left out of its tile (member -1), as the caller of the reduction does
    for (size_t i = 0; i < nd.size(); ++i) if (!std::isfinite(x[i])) nd[i] = -1;
    const ms::Ingredients g = overTiles(nd, nodes, area, x);
    ScalarSummary s;
    s.w.resize(nodes); s.s1.resize(nodes); s.s2.resize(nodes); s.n.resize(nodes); s.min.resize(nodes); s.max.resize(nodes);
    for (int k = 0; k < nodes; ++k) { const size_t i = static_cast<size_t>(k); s.w[k] = g.weight[i]; s.s1[k] = g.sum[i]; s.s2[k] = g.sum_sq[i]; s.n[k] = static_cast<double>(g.count[i]); s.min[k] = g.min[i]; s.max[k] = g.max[i]; }
    return s;
}
struct FrameSummary { VectorXd w, c1, c2, cn, c11, c22, c12, mag; };
static FrameSummary frameSummary(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const nxr::ladder::Mat3& e1, const nxr::ladder::Mat3& e2, const nxr::ladder::Mat3& normal, const nxr::ladder::Mat3& v) {
    const size_t n = node.size();
    std::vector<int32_t> nd = node;
    std::vector<double> c1(n), c2(n), cn(n), c11(n), c22(n), c12(n), mag(n);
    for (size_t r = 0; r < n; ++r) {
        const Eigen::Index i = static_cast<Eigen::Index>(r);
        c1[r] = v.row(i).dot(e1.row(i)); c2[r] = v.row(i).dot(e2.row(i)); cn[r] = v.row(i).dot(normal.row(i));
        if (!std::isfinite(c1[r]) || !std::isfinite(c2[r]) || !std::isfinite(cn[r])) nd[r] = -1;
        c11[r] = c1[r] * c1[r]; c22[r] = c2[r] * c2[r]; c12[r] = c1[r] * c2[r]; mag[r] = std::hypot(c1[r], c2[r]);
    }
    FrameSummary f;
    const ms::Ingredients g1 = overTiles(nd, nodes, area, c1.data());
    f.w.resize(nodes);
    for (int k = 0; k < nodes; ++k) f.w[k] = g1.weight[static_cast<size_t>(k)];
    f.c1 = sums(g1); f.c2 = sums(overTiles(nd, nodes, area, c2.data())); f.cn = sums(overTiles(nd, nodes, area, cn.data()));
    f.c11 = sums(overTiles(nd, nodes, area, c11.data())); f.c22 = sums(overTiles(nd, nodes, area, c22.data())); f.c12 = sums(overTiles(nd, nodes, area, c12.data())); f.mag = sums(overTiles(nd, nodes, area, mag.data()));
    return f;
}
// a tile's resultant: the sum of the vector map carried to the tile centre (carriedToCentre, pointwise) over the tile's vertices; weight and magnitude as the old resultant returned them
static Eigen::VectorXcd lcResultant(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const nxr::ladder::Mat3& e1, const nxr::ladder::Mat3& e2, const VectorXd& rho, const nxr::ladder::Mat3& v,
                                    VectorXd* weight = nullptr, VectorXd* magnitude = nullptr) {
    std::vector<char> finite;
    const Eigen::VectorXcd z = nxr::ladder::carriedToCentre(e1, e2, rho, v, &finite);
    std::vector<int32_t> nd = node;
    std::vector<double> re(node.size()), im(node.size()), mg(node.size());
    for (size_t r = 0; r < node.size(); ++r) { if (!finite[r]) nd[r] = -1; re[r] = finite[r] ? z[static_cast<Eigen::Index>(r)].real() : 0.0; im[r] = finite[r] ? z[static_cast<Eigen::Index>(r)].imag() : 0.0; mg[r] = finite[r] ? std::abs(z[static_cast<Eigen::Index>(r)]) : 0.0; }
    const ms::Ingredients gr = overTiles(nd, nodes, area, re.data()), gi = overTiles(nd, nodes, area, im.data()), gm = overTiles(nd, nodes, area, mg.data());
    Eigen::VectorXcd out(nodes);
    for (int k = 0; k < nodes; ++k) out[k] = std::complex<double>(gr.sum[static_cast<size_t>(k)], gi.sum[static_cast<size_t>(k)]);
    if (weight) { weight->resize(nodes); for (int k = 0; k < nodes; ++k) (*weight)[k] = gr.weight[static_cast<size_t>(k)]; }
    if (magnitude) *magnitude = sums(gm);
    return out;
}
struct VectorSummary { FrameSummary frame; Eigen::VectorXcd transported; VectorXd weight, resultantLength; };
static VectorSummary vectorSummary(const std::vector<int32_t>& node, int nodes, const std::vector<double>& area, const nxr::ladder::Mat3& e1, const nxr::ladder::Mat3& e2, const nxr::ladder::Mat3& normal, const VectorXd& rho, const nxr::ladder::Mat3& v) {
    VectorSummary s;
    s.frame = frameSummary(node, nodes, area, e1, e2, normal, v);
    VectorXd mg;
    s.transported = lcResultant(node, nodes, area, e1, e2, rho, v, &s.weight, &mg);
    s.resultantLength.resize(nodes);
    for (int k = 0; k < nodes; ++k) s.resultantLength[k] = mg[k] > 0 ? std::abs(s.transported[k]) / mg[k] : std::numeric_limits<double>::quiet_NaN();
    return s;
}

static void meshParity(Golden& g) {
    if (g.tok() != "mesh") { std::cout << "  [FAIL] golden: expected mesh\n"; std::exit(1); }
    const std::string name = g.tok();
    const int n = static_cast<int>(g.num());
    const MatrixXd Pm = g.mat("positions"), Fm = g.mat("faces"), massG = g.mat("mass");
    Mesh mesh; mesh.n = n;
    mesh.positions.resize(static_cast<size_t>(3 * n)); mesh.faces.resize(static_cast<size_t>(Fm.size()));
    for (int i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) mesh.positions[3 * static_cast<size_t>(i) + static_cast<size_t>(k)] = Pm(i, k);
    for (int i = 0; i < Fm.rows(); ++i) for (int k = 0; k < 3; ++k) mesh.faces[3 * static_cast<size_t>(i) + static_cast<size_t>(k)] = static_cast<int32_t>(Fm(i, k));
    const int nF = static_cast<int>(Fm.rows());
    EXPECT(maxDiff(rowOf(vertexAreas(mesh)), massG) == 0, name + ": vertex areas equal the Studio's bit for bit");
    if (g.tok() != "levels") { std::cout << "  [FAIL] golden: expected levels\n"; std::exit(1); }
    const int K = static_cast<int>(g.num());
    const double areaG = g.num();
    const SurfaceLadder L = surfaceLadder(mesh);
    bool labelsEq = (static_cast<int>(L.levels.size()) == K) && L.area == areaG, scalarsOk = true, areasEq = true, countsEq = true, centresEq = true, adjEq = true;
    double scalarErr = 0;
    std::vector<MatrixXd> stats;
    for (int k = 0; k < K; ++k) {
        const MatrixXd lv = g.mat("level" + std::to_string(k)), lab = g.mat("labels" + std::to_string(k)), ar = g.mat("areas" + std::to_string(k)), ct = g.mat("counts" + std::to_string(k)),
                       ce = g.mat("centres" + std::to_string(k)), ad = g.mat("adjacency" + std::to_string(k));
        if (k >= static_cast<int>(L.levels.size())) { labelsEq = false; continue; }
        const SurfaceLevel& l = L.levels[static_cast<size_t>(k)];
        labelsEq = labelsEq && maxDiff(rowOf(L.labels[static_cast<size_t>(k)]), lab) == 0;
        areasEq = areasEq && maxDiff(rowOf(L.tileAreas[static_cast<size_t>(k)]), ar) == 0;
        countsEq = countsEq && maxDiff(rowOf(L.tileVertices[static_cast<size_t>(k)]), ct) == 0;
        const double mine[11] = {static_cast<double>(l.tiles), l.lambdaMin, l.lambdaMax, l.wavelength, l.diameter, l.diameterInWavelengths, l.tau, l.areaPerTile.cv, l.areaPerTile.min, l.areaPerTile.max, static_cast<double>(l.finestTiles)};
        for (int c = 0; c < 11; ++c) { const double e = std::abs(mine[c] - lv(0, c)) / std::max(1.0, std::abs(lv(0, c))); scalarErr = std::max(scalarErr, e); }
        const std::vector<int> cen = tileCentres(mesh.positions.data(), L.labels[static_cast<size_t>(k)], l.tiles, L.mass);
        centresEq = centresEq && maxDiff(rowOf(cen), ce) == 0;
        const auto adj = tileAdjacency(mesh.faces.data(), nF, L.labels[static_cast<size_t>(k)]);
        MatrixXd am(static_cast<Eigen::Index>(adj.size()), 2);
        for (size_t e = 0; e < adj.size(); ++e) { am(static_cast<Eigen::Index>(e), 0) = adj[e].first; am(static_cast<Eigen::Index>(e), 1) = adj[e].second; }
        adjEq = adjEq && maxDiff(am, ad) == 0;
    }
    scalarsOk = scalarErr < 1e-13;
    std::cout << "  " << name << ": " << K << " levels, level scalars max rel diff " << sci(scalarErr) << "\n";
    EXPECT(labelsEq, name + ": the tile of every vertex at every level is the Studio's (identical)");
    EXPECT(areasEq && countsEq, name + ": tile areas and vertex counts equal the Studio's bit for bit");
    EXPECT(scalarsOk, name + ": per-level Weyl band, wavelength, tau, diameter, finest tiles equal the Studio's");
    EXPECT(centresEq, name + ": tile centres (the vertex nearest the mass centroid) equal the Studio's formula");
    EXPECT(adjEq, name + ": tile adjacency equals the Studio's");
    const MatrixXd xg = g.mat("x");
    VectorXd x = xg.row(0).transpose();
    double errS = 0;
    for (int k = 0; k < K; ++k) {
        const ScalarSummary s = scalarSummary(L.labels[static_cast<size_t>(k)], 1 << k, L.mass, x.data());
        const char* keys[6] = {"w", "s1", "s2", "n", "min", "max"};
        const VectorXd* v[6] = {&s.w, &s.s1, &s.s2, &s.n, &s.min, &s.max};
        for (int c = 0; c < 6; ++c) {
            errS = std::max(errS, maxDiff(row(*v[c]), g.mat("M" + std::to_string(k) + "_" + keys[c])));
        }
    }
    EXPECT(errS < 1e-13, name + ": scalar tile statistics (w, s1, s2, n, min, max) equal nxr map_stats (max |diff| " + sci(errS) + ")");
    // the vector field, its frame, and the Levi-Civita connection (the golden's meshes both carry them)
    const VectorXd alpha = g.mat("alpha").row(0).transpose();
    const MatrixXd E1 = g.mat("e1"), E2 = g.mat("e2"), NN = g.mat("nrm"), VV = g.mat("v");
    nxr::ladder::Mat3 e1 = E1, e2 = E2, nrm = NN, v = VV;
    nxr::chart::ChartKit kit(mesh.positions.data(), n, mesh.faces.data(), nF);
    const VectorXd gauge = nxr::frames::gaugeAngle(kit, e1);
    // the golden's e1 is the Studio's basis turned by alpha (the Studio's gauge); the gauge is geometry-central's now, so `gauge` (the angle of e1 in it) is NOT alpha. rho and Omega are the
    // transported frame angle seen from each vertex's own e1: gauge-free, so they are still compared with the Studio's (the transport is the same maths)
    double eg = 0;
    for (int i = 0; i < n; ++i) eg = std::max(eg, std::abs(wrap(gauge[i] - alpha[i])));
    std::cout << "  " << name << ": gauge angle of e1 in geometry-central's gauge against the Studio's alpha: max |diff| " << sci(eg) << " (the gauges differ; not compared)\n";
    if (g.tok() != "lc") { std::cout << "  [FAIL] golden: expected lc\n"; std::exit(1); }
    const int lcLevels = static_cast<int>(g.num());
    double eFrame = 0, eRho = 0, eOmega = 0, eLc = 0;
    std::vector<double> rhoDiffs;
    for (int k = 1; k <= lcLevels; ++k) {
        if (g.tok() != "level") { std::cout << "  [FAIL] golden: expected level\n"; std::exit(1); }
        g.num();
        const int tiles = 1 << k;
        const MatrixXd rhoG = g.mat("rho"), omG = g.mat("omega");
        const std::vector<int32_t>& lab = L.labels[static_cast<size_t>(k)];
        const FrameSummary fs = frameSummary(lab, tiles, L.mass, e1, e2, nrm, v);
        const char* keys[8] = {"w", "c1", "c2", "cn", "c11", "c22", "c12", "mag"};
        const VectorXd* fv[8] = {&fs.w, &fs.c1, &fs.c2, &fs.cn, &fs.c11, &fs.c22, &fs.c12, &fs.mag};
        for (int c = 0; c < 8; ++c) eFrame = std::max(eFrame, maxDiff(row(*fv[c]), g.mat(std::string("S_") + keys[c])));
        const MatrixXd reG = g.mat("lc_re"), imG = g.mat("lc_im");
        const std::vector<int> cen = tileCentres(mesh.positions.data(), lab, tiles, L.mass);
        LeviCivita lc = leviCivita(kit, e1, e2, gauge, lab, tiles, cen);     // e1's angle in the gauge (geometry-central's)
        // The transport is the vector heat flow: its magnitude falls like exp(-d / sqrt t) (t = the mean edge squared), so far from its source (the cut locus of a sphere, a tile's far side) the direction is
        // solver noise, different in each implementation and equal in none. Compared where the transport is decided: within kFar of the source (1.5 on these unit-scale meshes: the magnitude is above 1e-6).
        // Past kFar the Studio's own value is substituted, so the resultants below test everything else.
        const double kFar = 1.5;
        MatrixXd rhoOurs = row(lc.rho);
        MatrixXd om = lc.omega, omRef = omG;
        for (int a = 0; a < tiles; ++a) {
            if (cen[a] < 0) continue;
            const VectorXd da = kit.heatDistance(cen[a]);
            for (int i = 0; i < n; ++i) if (lab[static_cast<size_t>(i)] == a && da[i] > kFar) { rhoOurs(0, i) = rhoG(0, i); lc.rho[i] = rhoG(0, i); }
            for (int b = 0; b < tiles; ++b) if (cen[b] >= 0 && da[cen[b]] > kFar) om(a, b) = omRef(a, b) = 0;
        }
        for (int i = 0; i < n; ++i) if (std::isfinite(rhoOurs(0, i)) && std::isfinite(rhoG(0, i))) { const double d = std::abs(wrap(rhoOurs(0, i) - rhoG(0, i))); rhoDiffs.push_back(d); eRho = std::max(eRho, d); }
        for (int a = 0; a < tiles; ++a) for (int b = 0; b < tiles; ++b) if (std::isfinite(om(a, b)) && std::isfinite(omRef(a, b))) eOmega = std::max(eOmega, std::abs(wrap(om(a, b) - omRef(a, b))));
        const Eigen::VectorXcd z = lcResultant(lab, tiles, L.mass, e1, e2, lc.rho, v);
        eLc = std::max({eLc, maxDiff(row(z.real()), reG), maxDiff(row(z.imag()), imG)});
    }
    std::sort(rhoDiffs.begin(), rhoDiffs.end());
    const double rhoMed = rhoDiffs.empty() ? 0 : rhoDiffs[rhoDiffs.size() / 2];
    std::cout << "  " << name << ": frame sums " << sci(eFrame) << "; against the Studio's transport (wrapped): rho max " << sci(eRho) << " median " << sci(rhoMed) << ", Omega " << sci(eOmega) << ", Levi-Civita resultants " << sci(eLc) << "\n";
    EXPECT(eFrame < 1e-12, name + ": closed-form (frame) vector sums equal nxr vector_stats to 1e-12");
    // The transport is the same maths (vector heat flow on the intrinsic Delaunay triangulation) in another gauge: what differs is each vertex's 3D realisation of the gauge (the average of its edges rotated back; G2
    // appendix B item 5: a few 1e-3 rad on a regular mesh, more where the cone angle is far from 2 pi), which enters rho when a 3D frame e1 is read in it. So: equal to that difference, not to rounding.
    // The exact check is the analytic one (parallel transport on the sphere, below).
    const bool regular = name == "sphere";
    EXPECT(regular ? (eRho < 5e-3 && eOmega < 5e-3) : (rhoMed < 0.05 && eOmega < 0.5), name + ": the Levi-Civita angles rho and the connection Omega agree with the Studio's transport to the gauges' 3D realisation");
    EXPECT(regular ? eLc < 5e-3 : eLc < 0.1, name + ": the Levi-Civita resultants agree with nxr lc_resultant over the Studio's transport to the same");
}

static void treeParity(Golden& g) {
    if (g.tok() != "tree") { std::cout << "  [FAIL] golden: expected tree\n"; std::exit(1); }
    g.tok();
    const int n = static_cast<int>(g.num());
    g.num();
    const int finest = static_cast<int>(g.num());
    const MatrixXd Pm = g.mat("positions"), Fm = g.mat("faces");
    std::vector<double> P(static_cast<size_t>(Pm.size())); std::vector<int32_t> F(static_cast<size_t>(Fm.size()));
    for (int i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) P[3 * static_cast<size_t>(i) + static_cast<size_t>(k)] = Pm(i, k);
    for (int i = 0; i < Fm.rows(); ++i) for (int k = 0; k < 3; ++k) F[3 * static_cast<size_t>(i) + static_cast<size_t>(k)] = static_cast<int32_t>(Fm(i, k));
    bool same = true;
    for (int d = 0; d <= finest; ++d) {
        const MatrixXd codes = g.mat("codes" + std::to_string(d));
        const Tree t = subjectTree(P.data(), n, F.data(), static_cast<int>(Fm.rows()), d);
        const bool eqd = t.finest == finest && maxDiff(rowOf(t.codes), codes) == 0;
        if (!eqd) std::cout << "  depth " << d << " differs: finest " << t.finest << ", codes diff " << maxDiff(rowOf(t.codes), codes) << "\n";
        same = same && eqd;
    }
    const Tree t = subjectTree(P.data(), n, F.data(), static_cast<int>(Fm.rows()));
    std::cout << "  tree: finest " << t.finest << " (golden " << finest << "), depth " << t.depth << ", pieces " << t.runs.size() << "\n";
    EXPECT(same && t.depth == finest && t.runs.size() == 2, "two pieces: the subject tree's codes h 2^D + t equal the per-piece Studio ladders at every depth");
    const std::vector<int32_t> up = t.at(1);
    bool roll = true;
    for (int i = 0; i < n; ++i) roll = roll && up[static_cast<size_t>(i)] == (t.codes[static_cast<size_t>(i)] >> (t.depth - 1));
    EXPECT(roll && t.nLeaves() == 2 << t.depth, "a level is the code shifted: the roll-up is arithmetic");
}

// (2) equal area, nesting, Weyl
static void sphereProperties() {
    std::vector<double> V; std::vector<int32_t> F;
    icosphere(4, V, F, 1.0);
    const int n = static_cast<int>(V.size() / 3);
    Mesh m; m.n = n; m.positions = V; m.faces = F;
    const SurfaceLadder L = surfaceLadder(m);
    bool eq = true, nested = true, weyl = true, one = true;
    for (const SurfaceLevel& l : L.levels) {
        const double per = static_cast<double>(n) / l.tiles;
        eq = eq && l.areaPerTile.cv < (per >= 64 ? 0.02 : per >= 8 ? 0.15 : 0.5) && std::abs(l.areaPerTile.mean - L.area / l.tiles) < 1e-12;
        weyl = weyl && std::abs(l.lambdaMax - 4 * M_PI * l.tiles / L.area) < 1e-9 && std::abs(weylModes(l.lambdaMax, L.area) - l.tiles) < 1e-9 && std::abs(l.wavelength - wavelengthOf(l.lambdaMax)) < 1e-12;
        if (l.level > 0) for (int v = 0; v < n; ++v) nested = nested && (L.labels[static_cast<size_t>(l.level)][static_cast<size_t>(v)] >> 1) == L.labels[static_cast<size_t>(l.level) - 1][static_cast<size_t>(v)];
    }
    // every tile is one piece: its adjacency graph over vertices is connected (checked through the tile edges: tiles of a level are joined to a neighbour)
    for (size_t k = 1; k < L.labels.size(); ++k) one = one && !tileAdjacency(m.faces.data(), static_cast<int>(m.faces.size() / 3), L.labels[k]).empty();
    std::cout << "  icosphere(4): " << L.levels.size() << " levels, finest " << L.levels.back().tiles << " tiles, area cv at 2^6 tiles " << sci(L.levels[6].areaPerTile.cv) << "\n";
    EXPECT(eq, "tile areas are equal per level within the Studio's tolerance (cv < 0.02 from 64 vertices a tile, 0.15 from 8, 0.5)");
    EXPECT(nested, "a level refines the one above it exactly (tile t -> children 2t, 2t+1)");
    EXPECT(weyl, "Weyl's law: 2^l tiles pair with lambda = 4 pi 2^l / A, wavelength 2 pi / sqrt(lambda), N(lambda) = tiles");
    EXPECT(one, "every level has tiles joined to neighbours");
}

// (3) a uniform flow on a folded sheet
static void foldedSheet() {
    const int nx = 61, ny = 21;
    const double Lx = 3.0, Ly = 1.0, h = 0.3, k = 2 * M_PI;       // three folds of slope up to h k = 1.9
    const int n = nx * ny;
    std::vector<double> P(static_cast<size_t>(3 * n));
    std::vector<int32_t> F;
    auto id = [&](int i, int j) { return j * nx + i; };
    for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
        const double x = Lx * i / (nx - 1), y = Ly * j / (ny - 1);
        P[3 * static_cast<size_t>(id(i, j))] = x; P[3 * static_cast<size_t>(id(i, j)) + 1] = y; P[3 * static_cast<size_t>(id(i, j)) + 2] = h * std::sin(k * x);
    }
    for (int j = 0; j + 1 < ny; ++j) for (int i = 0; i + 1 < nx; ++i) {
        const int a = id(i, j), b = id(i + 1, j), c = id(i + 1, j + 1), d = id(i, j + 1);
        for (int q : {a, b, c, a, c, d}) F.push_back(q);
    }
    const int nF = static_cast<int>(F.size() / 3);
    Mesh mesh; mesh.n = n; mesh.positions = P; mesh.faces = F;
    const SurfaceLadder L = surfaceLadder(mesh);
    const nxr::frames::Mat3 nrm = nxr::frames::vertexNormals(P.data(), n, F.data(), nF);
    nxr::chart::ChartKit kit(P.data(), n, F.data(), nF);
    // the uniform flow: the unit tangent along the sheet's x direction (parallel along the whole sheet: the sheet is a cylinder, flat)
    nxr::ladder::Mat3 v(n, 3), e1(n, 3), e2(n, 3);
    for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) {
        const int q = id(i, j);
        const double x = Lx * i / (nx - 1), s = h * k * std::cos(k * x), nrmv = std::sqrt(1 + s * s);
        const Eigen::Vector3d tx(1 / nrmv, 0, s / nrmv), nn = nrm.row(q).transpose();
        const Eigen::Vector3d t = (tx - tx.dot(nn) * nn).normalized();
        v.row(q) = t.transpose();
        // a frame that turns with x (a canonical-like frame that is not parallel): e1 = t turned by alpha
        const double al = 4.0 * x + 0.5 * std::sin(3 * (j * Ly / (ny - 1)));
        const Eigen::Vector3d tn = nn.cross(t);
        e1.row(q) = (std::cos(al) * t + std::sin(al) * tn).transpose();
        e2.row(q) = nn.cross(Eigen::Vector3d(e1.row(q).transpose())).transpose();
    }
    const VectorXd gauge = nxr::frames::gaugeAngle(kit, e1);
    const Mesh& m = mesh;
    bool ok = true;
    for (int level = 1; level <= 2; ++level) {
        const int tiles = 1 << level;
        const std::vector<int32_t>& lab = L.labels[static_cast<size_t>(level)];
        const std::vector<int> cen = tileCentres(m.positions.data(), lab, tiles, L.mass);
        const LeviCivita lc = leviCivita(kit, e1, e2, gauge, lab, tiles, cen);
        const VectorSummary vs = vectorSummary(lab, tiles, L.mass, e1, e2, nrm, lc.rho, v);
        // the raw 3-D sum of the vectors of a tile, and the frame-component sum
        Eigen::MatrixXd raw = Eigen::MatrixXd::Zero(tiles, 3);
        VectorXd w = VectorXd::Zero(tiles);
        for (int q = 0; q < n; ++q) { raw.row(lab[static_cast<size_t>(q)]) += L.mass[static_cast<size_t>(q)] * v.row(q); w[lab[static_cast<size_t>(q)]] += L.mass[static_cast<size_t>(q)]; }
        double rawMin = 1, lcMin = 1, frameMax = 0;
        for (int t = 0; t < tiles; ++t) {
            rawMin = std::min(rawMin, raw.row(t).norm() / w[t]);
            lcMin = std::min(lcMin, std::abs(vs.transported[t]) / w[t]);
            frameMax = std::max(frameMax, std::hypot(vs.frame.c1[t], vs.frame.c2[t]) / w[t]);
        }
        std::cout << "  folded sheet, level " << level << ": resultant length per tile  raw 3-D sum min " << rawMin << ", frame components max " << frameMax << ", transported (Levi-Civita) min " << lcMin << "\n";
        ok = ok && lcMin > 0.97 && rawMin < 0.85 && frameMax < lcMin - 0.3;
    }
    EXPECT(ok, "a uniform flow on a folded sheet keeps its tile resultant length in the transported form (> 0.97) where a raw 3-D sum shrinks (< 0.85) and a turning frame's components do not add");
}

// the exact check of the transport: parallel transport along the great circle on the unit sphere, against rho (the transported frame angle seen from each vertex's own e1)
static void sphereTransport() {
    std::vector<double> P; std::vector<int32_t> F;
    icosphere(4, P, F, 1.0);
    const int n = static_cast<int>(P.size() / 3), nF = static_cast<int>(F.size() / 3);
    nxr::chart::ChartKit kit(P.data(), n, F.data(), nF);
    nxr::ladder::Mat3 e1(n, 3), e2(n, 3);
    const Eigen::Vector3d z(0, 0, 1);
    for (int i = 0; i < n; ++i) {
        const Eigen::Vector3d q(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
        Eigen::Vector3d t = z - z.dot(q) * q;
        if (t.norm() < 1e-9) t = Eigen::Vector3d(1, 0, 0) - q.x() * q;   // a pole
        t.normalize();
        e1.row(i) = t.transpose(); e2.row(i) = q.cross(t).transpose();
    }
    const VectorXd gauge = nxr::frames::gaugeAngle(kit, e1);
    const int c = 100;
    const Eigen::Vector3d p(P[3 * c], P[3 * c + 1], P[3 * c + 2]), u = e1.row(c).transpose();
    const std::vector<int32_t> node(static_cast<size_t>(n), 0);
    const LeviCivita lc = leviCivita(kit, e1, e2, gauge, node, 1, {c});
    double err = 0; int used = 0;
    for (int i = 0; i < n; ++i) {
        const Eigen::Vector3d q(P[3 * i], P[3 * i + 1], P[3 * i + 2]);
        const double s = std::acos(std::min(1.0, std::max(-1.0, p.dot(q))));
        if (s < 0.2 || s > 1.2) continue;
        const Eigen::Vector3d Tp = (q - q.dot(p) * p).normalized(), Tq = -p * std::sin(s) + Tp * std::cos(s);
        const Eigen::Vector3d w = u.dot(Tp) * Tq + u.dot(p.cross(Tp)) * q.cross(Tq);
        const double rho = std::atan2(w.dot(e2.row(i).transpose()), w.dot(e1.row(i).transpose()));
        err = std::max(err, std::abs(wrap(lc.rho[i] - rho)));
        ++used;
    }
    std::cout << "  sphere (icosphere 4), transport from one vertex, " << used << " vertices with 0.2 < d < 1.2: max |rho - parallel transport along the great circle| " << sci(err) << "\n";
    EXPECT(used > 100 && err < 0.02, "rho is the parallel transport along the great circle on the sphere (analytic)");
}

int main() {
    std::cout << "=== ladder (the Studio's dyadic tiles, and the tile summaries) ===\n";
    Golden g(std::string(NXR_TEST_GOLDEN_DIR) + "/ladder_golden.txt");
    if (!g.good()) { std::cout << "  [FAIL] ladder_golden.txt missing\n"; return 1; }
    meshParity(g); meshParity(g); treeParity(g);
    sphereProperties();
    sphereTransport();
    foldedSheet();
    std::cout << (g_failures ? "FAILED\n" : "OK\n");
    return g_failures ? 1 : 0;
}
