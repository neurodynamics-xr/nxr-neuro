// Template ends on the subject: the match template vertex -> subject vertex, and the reference's quirk FIXED: a template vertex no subject vertex takes weight from goes to its
// NEAREST subject vertex on the registration sphere, within its piece (a brute-force search), never to subject vertex 0.
#include <cmath>
#include <iostream>

#include "connectome_helpers.h"
#include "nxr/connectome.h"
#include "nxr/nearest.h"
#include "nxr/sphere_registration.h"

using namespace nxr::connectome;
using nxr::registration::Piece;

static void glue(const std::vector<double>& V, const std::vector<int32_t>& F, std::vector<double>& GV, std::vector<int32_t>& GF) {
  GV = V; GV.insert(GV.end(), V.begin(), V.end());
  GF = F; for (auto f : F) GF.push_back(f + static_cast<int32_t>(V.size() / 3));
}

int main() {
  std::vector<double> cv, sv; std::vector<int32_t> cf, sf;
  std::vector<double> ov; std::vector<int32_t> of;
  icosphere(2, cv, cf);   // 162
  icosphere(1, ov, of);   // 42
  icosphere(3, sv, sf);   // 642
  // ---- a subject DENSER than the template: every template vertex is taken by some subject vertex (no empty column): the plain strongest-weight match
  {
    std::vector<double> T, S; std::vector<int32_t> tf, sff;
    glue(cv, cf, T, tf); glue(sv, sf, S, sff);
    const std::vector<Piece> tp = {{0, 162}, {162, 162}}, sp = {{0, 642}, {642, 642}};
    MatchReport rep;
    const auto match = registrationMatch<double>(T, tp, S, sp, 8, &rep);
    nxr::registration::Options o;
    const auto w = nxr::registration::resample_operator<double>(T, {}, tp, S, sp, o);
    EXPECT(rep.empty_columns == 0 && strongestDestination(w) == match, "a template not denser than the subject: no empty column; the match is the strongest destination");
    bool pieces = true;
    for (int v = 0; v < 324; ++v) pieces = pieces && (v < 162 ? match[static_cast<std::size_t>(v)] < 642 : match[static_cast<std::size_t>(v)] >= 642);
    EXPECT(pieces, "the pieces never mix: left goes to left, right to right");
    // a template vertex is carried to the subject vertex at the same place on the same sphere (cv is a subset of sv: vertex i of level 2 is vertex i of level 3)
    bool same = true;
    for (int v = 0; v < 162; ++v) same = same && match[static_cast<std::size_t>(v)] == v;
    EXPECT(same, "a vertex common to both spheres goes to itself");
  }
  // ---- the quirk: a template DENSER than the subject: empty columns
  for (int pass = 0; pass < 2; ++pass) {
    std::vector<double> T, S; std::vector<int32_t> tf, sff;
    glue(sv, sf, T, tf); if (pass == 0) glue(cv, cf, S, sff); else glue(ov, of, S, sff);   // template 642 x 2; subject 162 x 2, then 42 x 2 (turned)
    // pass 1: the subject sphere turned a little and of another radius: the radius is irrelevant
    if (pass == 1) for (std::size_t i = 0; i < S.size(); i += 3) { const double x = S[i], y = S[i + 1]; S[i] = 5 * (0.99 * x - 0.14 * y); S[i + 1] = 5 * (0.14 * x + 0.99 * y); S[i + 2] *= 5; }
    const int ns = pass == 0 ? 162 : 42;
    const std::vector<Piece> tp = {{0, 642}, {642, 642}}, sp = {{0, ns}, {ns, ns}};
    MatchReport rep;
    const auto match = registrationMatch<double>(T, tp, S, sp, 8, &rep);
    nxr::registration::Options o;
    const auto w = nxr::registration::resample_operator<double>(T, {}, tp, S, sp, o);
    const auto strongest = strongestDestination(w);
    EXPECT(rep.empty_columns > 0, "pass " << pass << ": a template denser than the subject has empty columns (" << rep.empty_columns << ")");
    std::int64_t empties = 0, to_zero_wrong = 0;
    bool brute = true, ownPiece = true, kept = true;
    for (int v = 0; v < 1284; ++v) {
      const int piece = v < 642 ? 0 : 1;
      const int lo = piece ? ns : 0, hi = piece ? 2 * ns : ns;
      ownPiece = ownPiece && match[static_cast<std::size_t>(v)] >= lo && match[static_cast<std::size_t>(v)] < hi;
      if (strongest[static_cast<std::size_t>(v)] >= 0) { kept = kept && match[static_cast<std::size_t>(v)] == strongest[static_cast<std::size_t>(v)]; continue; }
      ++empties;
      // brute force on the unit-normalised spheres: the subject vertex of the piece nearest by chord
      double q[3], n = std::sqrt(T[3 * v] * T[3 * v] + T[3 * v + 1] * T[3 * v + 1] + T[3 * v + 2] * T[3 * v + 2]);
      for (int k = 0; k < 3; ++k) q[k] = T[static_cast<std::size_t>(3 * v + k)] / n;
      int best = lo; double bd = 1e300;
      for (int s = lo; s < hi; ++s) {
        const double m = std::sqrt(S[3 * s] * S[3 * s] + S[3 * s + 1] * S[3 * s + 1] + S[3 * s + 2] * S[3 * s + 2]);
        double d = 0;
        for (int k = 0; k < 3; ++k) { const double t = S[static_cast<std::size_t>(3 * s + k)] / m - q[k]; d += t * t; }
        if (d < bd) { bd = d; best = s; }
      }
      // the chosen vertex is a nearest one: its chord distance equals the minimum (an edge midpoint of the coarse sphere is a tie of two vertices to the last bit, and which one
      // rounding favours is not the point)
      auto chord = [&](int s) {
        const double m = std::sqrt(S[3 * s] * S[3 * s] + S[3 * s + 1] * S[3 * s + 1] + S[3 * s + 2] * S[3 * s + 2]);
        double d = 0;
        for (int k = 0; k < 3; ++k) { const double t = S[static_cast<std::size_t>(3 * s + k)] / m - q[k]; d += t * t; }
        return d;
      };
      brute = brute && std::abs(chord(match[static_cast<std::size_t>(v)]) - bd) < 1e-12;
      if (std::abs(chord(lo) - bd) < 1e-12) best = lo;   // vertex lo is allowed when it is a nearest one
      if (match[static_cast<std::size_t>(v)] == lo && best != lo) ++to_zero_wrong;
    }
    EXPECT(empties == rep.empty_columns, "pass " << pass << ": the report counts the empty columns");
    EXPECT(brute, "pass " << pass << ": every template vertex of an empty column goes to its nearest subject vertex on the sphere (brute force)");
    EXPECT(to_zero_wrong == 0, "pass " << pass << ": no end goes to the piece's first vertex unless that is the nearest (the reference's defect, fixed)");
    EXPECT(ownPiece && kept, "pass " << pass << ": pieces never mix; vertices with a subject neighbour keep the strongest-weight match");
    // the ends carried: an end on template vertex v goes to match[v]
    const std::vector<std::int32_t> endv = {0, 700, 5, 641};
    const auto carried = carryEnds(endv, match);
    EXPECT(carried[0] == match[0] && carried[1] == match[700] && carried[2] == match[5] && carried[3] == match[641], "carryEnds is match[end vertex]");
    bool bad = false;
    try { carryEnds({5000}, match); } catch (const std::invalid_argument&) { bad = true; }
    EXPECT(bad, "an end outside the registration is refused");
  }
  // precision follows the input
  {
    const std::vector<float> a(sv.begin(), sv.end()), b(cv.begin(), cv.end());
    const auto m = registrationMatch<float>(a, {{0, 642}}, b, {{0, 162}});
    EXPECT(m.size() == 642, "float32 spheres");
  }
  std::cout << (g_failures ? "FAILED\n" : "all registration tests passed\n");
  return g_failures ? 1 : 0;
}
