// nxr::montage — the STANDARD EEG ELECTRODE POSITIONS (10-05, with 10-10 and 10-20 as subsets) placed on a closed scalp surface by the published rule. PURE MATHS:
// arrays in, arrays out; no file, no subject, no knowledge of which head it is.
//
// The rule is Oostenveld R, Praamstra P (2001) The five percent electrode system for high-resolution EEG and ERP measurements, Clin Neurophysiol 112:713-719 (after Jasper 1958
// and Chatrian et al. 1985; the names and subsets follow Jurcak V, Tsuzuki D, Dan I 2007, NeuroImage 34:1600-1611). It is written here from the paper. The four landmarks are
// the nasion Nz, the inion Iz and the two preauricular points (LPA = T9, RPA = T10). Every contour is the intersection of the scalp mesh with the plane through three points,
// and the points sit at fixed fractions of that contour's length between its end points:
//   1. the midline Nz - Cz - Iz at 5% steps (21 points), Cz being the point that is at 50% of both the midline and of the coronal contour LPA - Cz - RPA (found by iteration);
//   2. the lowest contours Nz - T9 - Iz and Nz - T10 - Iz (T9 = LPA, T10 = RPA), and the coronal contour T9 - Cz - T10, each at 5% steps;
//   3. the contours NFpz - T9h - OIz and NFpz - T10h - OIz (T9h, T10h the 5% points of the coronal contour), and Fpz - T7 - Oz and Fpz - T8 - Oz (T7, T8 at 10%, 90% of it);
//   4. the 14 coronal rows (AFp ... POO), each from its point on the Fpz - T7 - Oz contour through its midline point to its point on the Fpz - T8 - Oz contour, at 1/16 steps.
// A point defined by an earlier contour is never moved by a later one. On a sphere every contour is a great circle and the positions are the unit-sphere positions of the system.
//
// Units are the surface's own (metres for a head in an SCS or RAS-like frame): nothing here knows them. The handedness is read from the landmarks, not assumed.
//
// What it cannot resolve: these are the SYSTEM's positions on this head model, not where a cap sits on a person (a cap's own placement, the hair, the head's change in shape);
// the arcs are lengths along a triangle mesh, so a coarse mesh shortens them (the error goes as the square of the edge length); the inion is a landmark of the caller's.
#pragma once
#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nxr::montage {

using Point = std::array<double, 3>;

struct Landmarks {
  Point nasion{}, inion{}, lpa{}, rpa{};   // in the scalp's coordinates, in its unit
};

struct Options {
  // the iteration for Cz stops when the points it moves between have moved less than this fraction of the nasion - inion distance (default 1e-7, a fraction)
  double tolerance = 1e-7;
  // the most iterations before it refuses (default 100, a count)
  int max_iterations = 100;
};

struct Placement {
  std::vector<std::string> labels;     // the 10-05 names, in the order they are placed: the midline first
  std::vector<double> positions;       // [labels x 3], row-major, on the scalp, in its unit
  Landmarks on_scalp;                  // the four landmarks as the rule used them: each moved to its nearest point of the scalp
  double landmark_shift[4] = {0, 0, 0, 0};   // how far each landmark (nasion, inion, lpa, rpa) was moved to reach the scalp, in the scalp's unit
  int iterations = 0;                  // the iterations Cz took
  double cz_residual = 0;              // how far Cz still moved on the last one, as a fraction of the nasion - inion distance
};

/** The 345 names of the 10-05 system (Nz and Iz included), in placement order. */
const std::vector<std::string>& labels1005();
/** The 10-10 system (77 names, Nz and Iz included) and the 10-20 system (21 names): subsets of the 10-05 names. */
const std::vector<std::string>& labels1010();
const std::vector<std::string>& labels1020();
/** The old 10-20 names and the names they became: T3 -> T7, T4 -> T8, T5 -> P7, T6 -> P8. */
const std::map<std::string, std::string>& oldNames();

/** The 10-05 positions on a closed (or neck-open) triangle mesh: vertices [V x 3] and faces [F x 3] (0-based vertex indices), both row-major. std::invalid_argument for a bad
 *  shape, a mesh the rule cannot be drawn on (a contour that does not close or branches, an end point outside its arc) or a Cz that did not settle. */
Placement tenFive(const std::vector<double>& vertices, const std::vector<std::int32_t>& faces, const Landmarks& landmarks, const Options& options = {});

}  // namespace nxr::montage
