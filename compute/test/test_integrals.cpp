// nxr::integrals against nxr's measure.integrals (golden: test/golden/integrals_golden.txt, gen_integrals_golden.py; synthetic closed icosphere and folded plane):
// the face field, the lumped vertex areas, and each tile's flux and circulation per frame to rounding; additive over tiles (a merged labelling's integral is the sum);
// a closed surface's tiles together have no net outflow; refusals (NaN, wrong size). And the INTRINSIC form (nxr::intrinsic::tileFlux / tileCirculation, the exact tile integrals of a 1-form on the
// complex): the circulation of a gradient is 0, the flux of d0 f over all the tiles of a closed surface is 0, a parent's flux is the sum of its children's, and for a smooth f it agrees with the extrinsic
// flux of the sharpened field (nxr::integrals) to a few percent.
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "nxr/integrals.h"
#include "nxr/intrinsic_forms.h"

static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

namespace {
std::vector<double> nums(std::istringstream& s) { std::vector<double> v; double x; while (s >> x) v.push_back(x); return v; }
double relmax(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return 1e30;
  double e = 0, sc = 0;
  for (std::size_t i = 0; i < a.size(); ++i) { e = std::max(e, std::abs(a[i] - b[i])); sc = std::max(sc, std::abs(b[i])); }
  return sc > 0 ? e / sc : e;
}
struct Case { std::string tag; int nV = 0, nF = 0, T = 0, tiles = 0; std::map<std::string, std::vector<double>> a; };
}  // namespace

int main() {
  std::ifstream in(std::string(NXR_TEST_GOLDEN_DIR) + "/integrals_golden.txt");
  EXPECT(in.good(), "the golden is there");
  std::vector<Case> cases;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream s(line);
    std::string tag;
    s >> tag;
    if (tag == "case") { Case c; s >> c.tag >> c.nV >> c.nF >> c.T >> c.tiles; cases.push_back(c); }
    else if (!cases.empty()) cases.back().a[tag] = nums(s);
  }
  EXPECT(cases.size() == 2, "two cases");
  for (const Case& c : cases) {
    std::vector<std::int32_t> F(c.a.at("F").size()), codes(c.a.at("codes").size());
    for (std::size_t i = 0; i < F.size(); ++i) F[i] = static_cast<std::int32_t>(c.a.at("F")[i]);
    for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<std::int32_t>(c.a.at("codes")[i]);
    const std::vector<double>& P = c.a.at("P");
    Eigen::MatrixXd J(3 * c.nV, c.T);
    for (int r = 0; r < 3 * c.nV; ++r) for (int t = 0; t < c.T; ++t) J(r, t) = c.a.at("J")[static_cast<std::size_t>(r) * c.T + t];
    const auto va = nxr::integrals::vertexAreas(P.data(), c.nV, F.data(), c.nF);
    EXPECT(relmax(va, c.a.at("vertex_area")) < 1e-13, c.tag << ": vertex areas");
    const Eigen::MatrixXd ff = nxr::integrals::faceField(P.data(), c.nV, F.data(), c.nF, J);
    std::vector<double> ffv;
    for (int r = 0; r < ff.rows(); ++r) for (int t = 0; t < c.T; ++t) ffv.push_back(ff(r, t));
    EXPECT(relmax(ffv, c.a.at("face_field")) < 1e-13, c.tag << ": the face field");
    // the golden's tiles are the codes >= 0 ascending: map them to a dense range
    std::map<int, int> dense;
    for (std::size_t i = 0; i < c.a.at("tile_codes").size(); ++i) dense[static_cast<int>(c.a.at("tile_codes")[i])] = static_cast<int>(i);
    std::vector<std::int32_t> dc(codes.size());
    for (std::size_t i = 0; i < codes.size(); ++i) dc[i] = codes[i] >= 0 ? dense.at(codes[i]) : -1;
    const int nT = static_cast<int>(dense.size());
    EXPECT(nT == c.tiles, c.tag << ": the tiles");
    const auto phi = nxr::integrals::flux(P.data(), c.nV, F.data(), c.nF, J, dc.data(), nT);
    const auto gam = nxr::integrals::circulation(P.data(), c.nV, F.data(), c.nF, J, dc.data(), nT);
    std::vector<double> pv, gv;
    for (int k = 0; k < nT; ++k) for (int t = 0; t < c.T; ++t) { pv.push_back(phi.value(k, t)); gv.push_back(gam.value(k, t)); }
    const double ep = relmax(pv, c.a.at("flux")), eg = relmax(gv, c.a.at("circulation")), ea = relmax(phi.area, c.a.at("area"));
    std::cout << "  " << c.tag << ": flux " << ep << ", circulation " << eg << ", area " << ea << " (relative to nxr)\n";
    EXPECT(ep < 1e-12 && eg < 1e-12 && ea < 1e-13, c.tag << ": flux, circulation and area are nxr's");
    // ADDITIVE: merging two tiles into one gives the sum of their integrals
    if (nT >= 2) {
      std::vector<std::int32_t> merged = dc;
      for (auto& x : merged) if (x == 1) x = 0;
      const auto m = nxr::integrals::flux(P.data(), c.nV, F.data(), c.nF, J, merged.data(), nT);
      for (int t = 0; t < c.T; ++t) EXPECT(std::abs(m.value(0, t) - (phi.value(0, t) + phi.value(1, t))) < 1e-12 * (1 + std::abs(m.value(0, t))), c.tag << ": a parent's flux is the sum of its children's");
    }
  }
  // a closed sphere: all its tiles together have no net outflow (and no circulation) of any field
  {
    const Case& c = cases[0];
    std::vector<std::int32_t> F(c.a.at("F").size()), one(static_cast<std::size_t>(c.nV), 0);
    for (std::size_t i = 0; i < F.size(); ++i) F[i] = static_cast<std::int32_t>(c.a.at("F")[i]);
    Eigen::MatrixXd J(3 * c.nV, c.T);
    for (int r = 0; r < 3 * c.nV; ++r) for (int t = 0; t < c.T; ++t) J(r, t) = c.a.at("J")[static_cast<std::size_t>(r) * c.T + t];
    const auto p = nxr::integrals::flux(c.a.at("P").data(), c.nV, F.data(), c.nF, J, one.data(), 1);
    const auto g = nxr::integrals::circulation(c.a.at("P").data(), c.nV, F.data(), c.nF, J, one.data(), 1);
    EXPECT(std::abs(p.value(0, 0)) < 1e-12 && std::abs(g.value(0, 0)) < 1e-12, "a closed surface has no net outflow and no circulation");
    // refusals
    bool threw = false;
    Eigen::MatrixXd bad = J; bad(0, 0) = NAN;
    try { nxr::integrals::flux(c.a.at("P").data(), c.nV, F.data(), c.nF, bad, one.data(), 1); } catch (const std::invalid_argument&) { threw = true; }
    EXPECT(threw, "a NaN field is refused");
    threw = false;
    try { nxr::integrals::flux(c.a.at("P").data(), c.nV, F.data(), c.nF, J.topRows(3), one.data(), 1); } catch (const std::invalid_argument&) { threw = true; }
    EXPECT(threw, "a field of the wrong size is refused");
  }
  {   // THE INTRINSIC FORM on the closed sphere
    const Case& c = cases[0];
    std::vector<std::int32_t> F(c.a.at("F").size()), codes(c.a.at("codes").size());
    for (std::size_t i = 0; i < F.size(); ++i) F[i] = static_cast<std::int32_t>(c.a.at("F")[i]);
    for (std::size_t i = 0; i < codes.size(); ++i) codes[i] = static_cast<std::int32_t>(c.a.at("codes")[i]);
    const std::vector<double>& P = c.a.at("P");
    nxr::manifold::IntrinsicSurface dm(P.data(), c.nV, F.data(), c.nF);
    const nxr::intrinsic::Complex cx = nxr::intrinsic::complexOf(dm);
    Eigen::VectorXd f(c.nV);
    for (int v = 0; v < c.nV; ++v) f(v) = P[static_cast<std::size_t>(3 * v + 2)] / 0.07 + 0.3 * P[static_cast<std::size_t>(3 * v)] / 0.07;   // a smooth function (degree 1: its Laplacian is -2 f / R^2)
    const Eigen::VectorXd u = cx.d0 * f;                                                      // the 1-form of grad f
    const int nT = 8;
    const Eigen::VectorXd circ = nxr::intrinsic::tileCirculation(cx, u, codes.data(), nT);
    EXPECT(circ.cwiseAbs().maxCoeff() < 1e-12, "the circulation of a gradient is 0: " << circ.cwiseAbs().maxCoeff());
    const Eigen::VectorXd fl = nxr::intrinsic::tileFlux(cx, u, codes.data(), nT);
    EXPECT(std::abs(fl.sum()) < 1e-12 * (1 + fl.cwiseAbs().maxCoeff()), "a closed surface has no net outflow: " << fl.sum());
    std::vector<std::int32_t> merged = codes;
    for (auto& x : merged) if (x == 1) x = 0;
    const Eigen::VectorXd fm = nxr::intrinsic::tileFlux(cx, u, merged.data(), nT);
    EXPECT(std::abs(fm(0) - (fl(0) + fl(1))) < 1e-12 * (1 + std::abs(fm(0))), "a parent's flux is the sum of its children's");
    // against the extrinsic flux of the sharpened gradient (a face-constant tangent field): the same tile integrals up to the representation
    const Eigen::MatrixXd sf = nxr::intrinsic::sharp(cx, u);
    const auto ex = nxr::integrals::flux(P.data(), c.nV, F.data(), c.nF, sf, codes.data(), nT);
    double worst = 0, big = 0;
    for (int k = 0; k < nT; ++k) big = std::max(big, std::abs(fl(k)));
    for (int k = 0; k < nT; ++k) worst = std::max(worst, std::abs(fl(k) - ex.value(k, 0)));
    std::cout << "  intrinsic tile flux of a gradient vs the extrinsic one of its sharpened field: worst difference " << worst / big << " of the largest\n";
    EXPECT(worst < 0.1 * big, "the two tile fluxes agree: " << worst / big);
  }
  if (fails) { std::cout << fails << " failed\n"; return 1; }
  std::cout << "test_integrals: ok\n";
  return 0;
}
