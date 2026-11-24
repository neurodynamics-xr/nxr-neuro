// nxr-pipelines — THE FLOW THROUGH THE FIBRE NETWORK AND ITS CONNECTOME EIGENMODES as graph nodes (spec docs/specs/2026-11-24-connectome-flow.md; definitions `tile_network`,
// `connectome_flow`). A port of nxr's flow through the fibre network (flownet): the tiles of the surface's dyadic grid are the network's nodes, tile pairs joined by streamlines
// its edges; each tile's current is the source kernel averaged over its vertices; the tiles' cross-spectrum gives the edge flow; the network's divergence and graph potential
// are read in the connectome's eigenmodes.
//
// THE GRAPH (tile_network). before: fibres --[meeg.fiber_ends, meeg.nearest_vertex, meeg.end_gate]--> each fibre's end vertices (REUSED); fibres --[meeg.fiber_lengths]--> lengths;
//         end vertices + lengths + the grid's tiles at the stated level + the surface --[connectome.tile_pairs]--> tiles, pairs (count, median length, distance, across);
//         pairs --[connectome.laplacian_eigenmodes]--> the connectome Laplacian's complete eigenmodes.
// THE GRAPH (connectome_flow). before: the sensor serving operator (signal.mode_analysis, REUSED) + kernel + tiles --[connectome.tile_operator]--> ONE operator [3T x K];
//         stored chunk spectra + operator + atoms --[connectome.tile_flow]--> per atom the whole run's tile cross-spectrum (pair sums, tile power);
//         --[connectome.tile_time_cells]--> the same per time cell of each atom --[stats.sign_flip_consistency, REUSED]--> per pair R1 and its p;
//         pairs + eigenmodes + tile flow --[connectome.network_flow]--> per atom divergence, graph potential, its eigenmode coefficients, the gradient part's sums.
// Every node is a function of arrays: no file, no subject, no loop over a recording.
#include "nxr/pipelines/connectome_flow_steps.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "nxr/graph/node.h"
#include "nxr/meeg/tile_network.h"
#include "nxr/select/grid.h"
#include "nxr/signal/spectral_flow.h"
#include "nxr/signal/tile_flow.h"

namespace nxr::pipelines::graph {

namespace ng = nxr::graph;
using nxr::graph::Json;
using RowMajor = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using cd = std::complex<double>;

namespace {

const ng::Array& handed(const ng::Arrays& c, const char* port, const char* who) {
  const auto it = c.find(port);
  if (it == c.end() || it->second.meta.value("placeholder", false)) ng::refuse(std::string(who) + ": " + port + " was not handed over");
  return it->second;
}
bool was_handed(const ng::Arrays& c, const char* port) { const auto it = c.find(port); return it != c.end() && !it->second.meta.value("placeholder", false); }
ng::PortType statistic_port() { ng::PortType t; t.cls = ng::PortClass::constant; t.facts["cls"] = "statistic"; return t; }
ng::PortType array_port() { ng::PortType t; t.cls = ng::PortClass::constant; t.facts["cls"] = "array"; return t; }
double number_param(const Json& p, const char* key, const char* who) {
  if (!p.contains(key) || !p[key].is_number()) ng::refuse(std::string(who) + ": " + key + " has no default: name it (a number)");
  return p[key].get<double>();
}
std::string text_param(const Json& p, const char* key, const char* who) {
  if (!p.contains(key) || !p[key].is_string()) ng::refuse(std::string(who) + ": " + key + " has no default: name it");
  return p[key].get<std::string>();
}
Json indexed_axis(const char* name, std::int64_t size, const Json& values = Json()) {
  Json v = values;
  if (v.is_null()) { v = Json::array(); for (std::int64_t i = 0; i < size; ++i) v.push_back(i); }
  return Json{{"name", name}, {"size", size}, {"values", v}};
}
// what a statistic records about itself: fresh from a node (meta.describe.producer) or read back from the store (meta.producer)
Json producer_of(const ng::Array& a) {
  if (a.meta.contains("producer") && a.meta["producer"].is_object() && !a.meta["producer"].empty()) return a.meta["producer"];
  return a.meta.value("describe", Json::object()).value("producer", Json::object());
}
ng::Array table(const std::vector<std::vector<double>>& columns, std::int64_t rows) {
  ng::Array a;
  a.rows = rows; a.cols = static_cast<std::int64_t>(columns.size());
  a.data.reserve(static_cast<std::size_t>(a.rows * a.cols));
  for (std::int64_t r = 0; r < rows; ++r) for (const auto& c : columns) a.data.push_back(c[static_cast<std::size_t>(r)]);
  return a;
}
double cell(const ng::Array& a, std::int64_t row, std::int64_t column) { return a.at(static_cast<std::size_t>(row * a.cols + column)); }

// ── the tiles: the grid's finest codes shifted to the level of the stated tiles per piece ───────────────────────────
struct Tiles { std::vector<std::int64_t> codes; int level = 0, depth = 0; std::int64_t parts = 0; };
Tiles tiles_of(const ng::Array& grid, const Json& p, const char* who) {
  if (!grid.meta.contains("depth")) ng::refuse(std::string(who) + ": tiles is not a dyadic grid with levels (`nxr-neuro run select_dyadic_surface` first)");
  Tiles t;
  t.depth = grid.meta["depth"].get<int>();
  t.parts = static_cast<std::int64_t>(number_param(p, "tiles_per_piece", who));
  try { t.level = nxr::select::level_of_parts(static_cast<double>(t.parts), t.depth).level; } catch (const std::exception& e) { ng::refuse(std::string(who) + ": tiles_per_piece: " + e.what()); }
  std::vector<std::int64_t> finest(static_cast<std::size_t>(grid.size()));
  for (std::size_t v = 0; v < finest.size(); ++v) finest[v] = static_cast<std::int64_t>(std::llround(grid.at(v)));
  t.codes = nxr::meeg::tileCodesAtLevel(finest, t.depth, t.level);
  return t;
}
// each vertex's row among the tiles (codes ascending), -1 off them
std::vector<std::int32_t> tile_rows(const std::vector<std::int64_t>& codes, const std::vector<std::int64_t>& tileCodes) {
  std::map<std::int64_t, std::int32_t> row;
  for (std::size_t i = 0; i < tileCodes.size(); ++i) row[tileCodes[i]] = static_cast<std::int32_t>(i);
  std::vector<std::int32_t> out(codes.size(), -1);
  for (std::size_t v = 0; v < codes.size(); ++v) if (codes[v] >= 0) { const auto it = row.find(codes[v]); out[v] = it == row.end() ? -1 : it->second; }
  return out;
}
std::vector<std::int64_t> distinct_codes(const std::vector<std::int64_t>& codes) {
  std::vector<std::int64_t> out;
  for (const auto c : codes) if (c >= 0) out.push_back(c);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}

// ── the stored chunk spectra and the atoms: read as the sensor chain reads them (signal.spectral_flow's readers, the same rules) ──────────────────────────────
// (kept here until the sensor chain's readers are declared in a shared header; no rule differs)
struct Spectra {
  std::int64_t chunks = 0, bins = 0, rows = 0, first = 0, transform = 0, overlap = 0;
  double sfreq = 0;
  std::vector<std::int64_t> modes, owns;
  const ng::Array* array = nullptr;
  cd at(std::int64_t t, std::int64_t k, std::int64_t r) const {
    const std::size_t i = static_cast<std::size_t>(((t * bins + k) * rows + r) * 2);
    return cd(array->at(i), array->at(i + 1));
  }
  double frequency(std::int64_t k) const { return static_cast<double>(first + k) * sfreq / static_cast<double>(transform); }
};
Spectra spectra_of(const ng::Array& A, const char* who) {
  const Json& d = A.meta.contains("shape") ? A.meta : A.meta.value("describe", Json::object());
  const Json pr = producer_of(A);
  if (!d.contains("shape") || d["shape"].size() != 4 || d["shape"][3] != 2) ng::refuse(std::string(who) + ": spectra are not [chunks x bins x coordinates x 2] (signal.chunk_spectra)");
  Spectra s;
  s.chunks = d["shape"][0]; s.bins = d["shape"][1]; s.rows = d["shape"][2];
  for (const char* k : {"sfreq", "transform_samples", "first_bin", "modes"}) if (!pr.contains(k)) ng::refuse(std::string(who) + ": the spectra do not record " + k);
  s.sfreq = pr["sfreq"]; s.transform = pr["transform_samples"]; s.first = pr["first_bin"]; s.overlap = pr.value("overlap_samples", std::int64_t(0));
  s.modes = pr["modes"].get<std::vector<std::int64_t>>();
  for (const auto& c : pr.value("chunks", Json::array())) s.owns.push_back(c.value("samples", std::int64_t(0)));
  if (static_cast<std::int64_t>(s.modes.size()) != s.rows) ng::refuse(std::string(who) + ": the spectra's coordinates are not their modes");
  s.array = &A;
  return s;
}
std::vector<Json> members_of(const ng::Array& sel, const char* who) {
  const Json p = sel.meta.value("params", Json::object());
  if (!p.contains("kernels") || !p["kernels"].contains("members")) ng::refuse(std::string(who) + ": frequency is not a kernel set (`nxr-neuro run select_kernels`)");
  return p["kernels"]["members"].get<std::vector<Json>>();
}
// the atoms' gains per bin (Gaussians or boxes on a linear or log2 chart), normalised as the parameter says; their centres and time standard deviations
struct Atoms { Eigen::MatrixXd gains; std::vector<double> centres, standardDeviations; };
Atoms atoms_of(const Spectra& s, const ng::Array& sel, const Json& p, const char* who) {
  const std::vector<Json> members = members_of(sel, who);
  if (members.empty()) ng::refuse(std::string(who) + ": a kernel set with no member");
  Atoms a;
  a.gains = Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(members.size()), s.bins);
  for (std::size_t m = 0; m < members.size(); ++m) {
    const Json& k = members[m];
    const double c = k.at("start").get<double>(), gain = k.value("gain", 1.0);
    const std::string family = k.value("family", std::string()), chart = k.value("chart", std::string("linear"));
    const Json params = k.value("params", Json::object());
    double fwhm = params.value("fwhm", 0.0);
    if (params.value("width_rule", std::string("absolute")) == "proportional") fwhm *= c;
    const double sd = fwhm / (2.0 * std::sqrt(2.0 * std::log(2.0)));
    a.centres.push_back(c);
    a.standardDeviations.push_back(family == "gaussian" && chart == "linear" ? sd : std::numeric_limits<double>::quiet_NaN());
    for (std::int64_t i = 0; i < s.bins; ++i) {
      const double f = s.frequency(i);
      if (!(f > 0)) continue;                                                                   // one-sided: nothing at 0
      const double u = chart == "log2" ? std::log2(f) - std::log2(c) : f - c;
      if (family == "gaussian") a.gains(static_cast<Eigen::Index>(m), i) = gain * std::exp(-u * u / (2.0 * sd * sd));
      else if (family == "box") a.gains(static_cast<Eigen::Index>(m), i) = std::abs(u) <= fwhm / 2.0 ? gain : 0.0;
      else ng::refuse(std::string(who) + ": an atom's profile is gaussian or box, not " + family);
    }
  }
  const std::string normalisation = text_param(p, "frequency_normalisation", who);
  if (normalisation == "sum_of_squared_gains") a.gains = nxr::signal::normaliseBySumOfSquaredGains(a.gains, number_param(p, "support_floor", who));
  else if (normalisation != "none") ng::refuse(std::string(who) + ": frequency_normalisation is sum_of_squared_gains or none");
  return a;
}
Eigen::MatrixXd matrix_of(const ng::Array& a) { const std::vector<double> v = a.values(); return Eigen::Map<const RowMajor>(v.data(), a.rows, a.cols); }
// the tile operator's rows and the spectra's coordinates must be one set
void check_operator_against_spectra(const ng::Array& op, const Spectra& s, const char* who) {
  if (op.cols != s.rows || op.meta.value("columns", Json::object()).value("members", Json::array()) != Json(s.modes))
    ng::refuse(std::string(who) + ": the tile operator's columns are not the spectra's coordinates (made from another serving operator)");
  if (op.rows % 3 != 0) ng::refuse(std::string(who) + ": the tile operator's rows are not three components of each tile");
}
const char* const kWarnings[] = {
    "a tile's current is the kernel's mean over its vertices: activity of opposite sign within a tile cancels; the tiles' size sets the network's scale",
    "tractography ends are not cortical terminations; an end's vertex is the nearest vertex of the end surface",
    "leakage of the inverse correlates neighbouring tiles at zero lag: the imaginary part is blind to it but lags are pulled towards zero, so speeds are upper bounds",
    "a consistent lag between joined tiles does not show that the fibres carry it: compare with the distance-matched not-joined pairs; a common drive with different delays gives the same signature",
    "a whole-run cross-spectrum keeps the net quadrature only; the time cells test its consistency"};
Json warnings() { Json w = Json::array(); for (const char* s : kWarnings) w.push_back(s); return w; }

// ── meeg.fiber_lengths ───────────────────────────────────────────────────────────────────────────────────────────
const char* const kLengths = "meeg.fiber_lengths";
ng::NodeSpec fiber_lengths() {
  ng::NodeSpec n;
  n.name = kLengths;
  n.kind = ng::Kind::array;
  n.inputs = {{"fibers", "the fibre bundle: [nF·nP x 3] points, fibre-major", false}};
  n.outputs = {{"lengths", "each fibre's length [nF], in the points' unit (SCS metres for a converted bundle)", false}};
  n.parameters = {{"points_per_fiber", Json(), "the points of each fibre (0: read from the bundle)", "points", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "no time axis: runs once on arrays; the fibres in any order");
  n.matches = "nxr flownet's streamline lengths (Σ ||diff(points)||); compute nxr::meeg::fiberLengths";
  n.description = "each fibre's length: the sum of its segments' lengths";
  n.plan = [](const ng::Types& in, const Json&) {
    if (!in.count("fibers")) ng::refuse(std::string(kLengths) + ": fibers is not given");
    ng::Planned out; out.out["lengths"] = array_port(); return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(std::make_pair(&handed(c, "fibers", kLengths), p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [fibres, p] = std::any_cast<const std::pair<const ng::Array*, Json>&>(prepared);
    std::int64_t points = p.value("points_per_fiber", Json(0)).is_number_integer() ? p["points_per_fiber"].get<std::int64_t>() : 0;
    if (points == 0) points = fibres->meta.value("points_per_fiber", std::int64_t(0));
    if (points < 2) ng::refuse(std::string(kLengths) + ": the points of each fibre are not known (name points_per_fiber)");
    const std::vector<double> lengths = fibres->is_single() ? nxr::meeg::fiberLengths(fibres->single, points) : nxr::meeg::fiberLengths(fibres->data, points);
    ng::Array a;
    a.rows = static_cast<std::int64_t>(lengths.size()); a.cols = 1; a.data = lengths;
    return ng::Arrays{{"lengths", std::move(a)}};
  };
  return n;
}

// ── connectome.tile_pairs ────────────────────────────────────────────────────────────────────────────────────────
const char* const kPairs = "connectome.tile_pairs";
ng::NodeSpec tile_pairs() {
  ng::NodeSpec n;
  n.name = kPairs;
  n.kind = ng::Kind::array;
  n.inputs = {{"end_vertex", "each fibre's two end vertices on the surface [2nF], block order (meeg.nearest_vertex)", false},
              {"keep", "the gate: 1 for a fibre kept [nF] (meeg.end_gate)", true},
              {"lengths", "each fibre's length [nF] (meeg.fiber_lengths)", true},
              {"tiles", "the surface's dyadic grid (a grid selection: its finest code per vertex, its depth)", false},
              {"surface", "the surface the tiles and the ends are on (its vertices)", false}};
  n.outputs = {{"tiles_table", "per tile: code, piece, vertices, centroid x y z", false},
               {"pairs", "per pair of tiles i < j: first, second (tile rows), streamline count, median length, centroid distance, across the pieces", false}};
  n.parameters = {{"tiles_per_piece", Json(), "the tiles of each piece (hemisphere), a power of two: sets the grid's level (no default; nxr's flow through the fibres: 32, level 5)", "parts", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "no time axis: runs once; the fibres are summed in any order");
  n.matches = "nxr flownet's network (tiles of surface_ladder at --level; CNT and LEN over the tile pairs, centroid distances, across the hemispheres); compute nxr::meeg::tilePairs";
  n.description = "the fibre network on the tiles: per pair of tiles the streamlines joining them, their median length, the tiles' distance and whether they lie on different pieces";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"end_vertex", "tiles", "surface"}) if (!in.count(k)) ng::refuse(std::string(kPairs) + ": " + k + " is not given");
    number_param(p, "tiles_per_piece", kPairs);
    ng::Planned out; out.out["tiles_table"] = statistic_port(); out.out["pairs"] = statistic_port(); return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(std::make_pair(c, p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [c, p] = std::any_cast<const std::pair<ng::Arrays, Json>&>(prepared);
    const Tiles t = tiles_of(handed(c, "tiles", kPairs), p, kPairs);
    const ng::Array& S = handed(c, "surface", kPairs);
    if (S.cols != 3 || S.rows != static_cast<std::int64_t>(t.codes.size())) ng::refuse(std::string(kPairs) + ": the tiles are not on the surface (" + std::to_string(t.codes.size()) + " labels, " + std::to_string(S.rows) + " vertices)");
    const Eigen::MatrixXd positions = matrix_of(S);
    std::vector<std::int32_t> ends;
    for (const double e : handed(c, "end_vertex", kPairs).values()) ends.push_back(static_cast<std::int32_t>(std::llround(e)));
    std::vector<std::uint8_t> keep;
    if (was_handed(c, "keep")) for (const double k : c.at("keep").values()) keep.push_back(k != 0 ? 1 : 0);
    std::vector<double> lengths;
    if (was_handed(c, "lengths")) lengths = c.at("lengths").values();
    const nxr::meeg::TileNetwork net = nxr::meeg::tilePairs(t.codes, positions, t.level, ends, keep, lengths);
    const std::int64_t T = static_cast<std::int64_t>(net.codes.size()), P = net.count.size();
    std::vector<std::vector<double>> tc(6, std::vector<double>(static_cast<std::size_t>(T)));
    for (std::int64_t i = 0; i < T; ++i) {
      tc[0][static_cast<std::size_t>(i)] = static_cast<double>(net.codes[static_cast<std::size_t>(i)]); tc[1][static_cast<std::size_t>(i)] = static_cast<double>(net.pieces[static_cast<std::size_t>(i)]);
      tc[2][static_cast<std::size_t>(i)] = static_cast<double>(net.vertexCount[static_cast<std::size_t>(i)]);
      for (int k = 0; k < 3; ++k) tc[static_cast<std::size_t>(3 + k)][static_cast<std::size_t>(i)] = net.centroids(i, k);
    }
    std::vector<std::vector<double>> pc(6, std::vector<double>(static_cast<std::size_t>(P)));
    for (std::int64_t q = 0; q < P; ++q) {
      pc[0][static_cast<std::size_t>(q)] = net.first[static_cast<std::size_t>(q)]; pc[1][static_cast<std::size_t>(q)] = net.second[static_cast<std::size_t>(q)];
      pc[2][static_cast<std::size_t>(q)] = net.count(q); pc[3][static_cast<std::size_t>(q)] = net.medianLength(q); pc[4][static_cast<std::size_t>(q)] = net.distance(q); pc[5][static_cast<std::size_t>(q)] = net.across(q);
    }
    Json codes = Json::array();
    for (const auto code : net.codes) codes.push_back(code);
    const Json producer = Json{{"tiles_per_piece", t.parts}, {"level", t.level}, {"grid_depth", t.depth}, {"tiles", T}, {"pairs", P}, {"codes", codes},
                               {"fibres_left_out", Json{{"an_end_on_no_tile", net.endsOutside}, {"both_ends_in_one_tile", net.sameTile}, {"gated_out", net.gatedOut}}},
                               {"length_unit", "the fibres' points' unit (SCS metres)"}, {"distance_unit", "the surface's positions' unit"}, {"warnings", warnings()}};
    ng::Array tiles = table(tc, T);
    tiles.meta["shape"] = Json::array({T, 6}); tiles.meta["producer"] = producer;
    tiles.meta["describe"] = Json{{"method", "tile_pairs"}, {"measure", "tile"}, {"shape", Json::array({T, 6})},
                                  {"axes", Json::array({indexed_axis("tile", T, codes), indexed_axis("column", 6, Json::array({"code", "piece", "vertices", "centroid_x", "centroid_y", "centroid_z"}))})},
                                  {"producer", producer}, {"description", "the tiles at the stated level: code (2^level x piece + tile), piece, vertex count, centroid (the mean of the vertices' positions)"}};
    ng::Array pairs = table(pc, P);
    pairs.meta["shape"] = Json::array({P, 6}); pairs.meta["producer"] = producer;
    pairs.meta["describe"] = Json{{"method", "tile_pairs"}, {"measure", "fiber_count"}, {"shape", Json::array({P, 6})},
                                  {"axes", Json::array({indexed_axis("pair", P), indexed_axis("column", 6, Json::array({"first", "second", "fiber_count", "median_length", "distance", "across"}))})},
                                  {"producer", producer},
                                  {"description", "every pair of tiles i < j (rows of the tiles' table, in the order (0,1), (0,2), ..., (1,2), ...): the streamlines with one end in each, their median length (NaN: none), the centroids' distance, 1 when on different pieces"}};
    return ng::Arrays{{"tiles_table", std::move(tiles)}, {"pairs", std::move(pairs)}};
  };
  n.record = [](const std::any& prepared) { return Json{{"tiles_per_piece", std::any_cast<const std::pair<ng::Arrays, Json>&>(prepared).second.value("tiles_per_piece", Json())}}; };
  return n;
}

// the pairs table, read fresh or from the store: its tiles count and each pair's weight (count when >= min_streamlines, else 0)
struct PairsRead { std::int64_t tiles = 0, pairs = 0; Eigen::VectorXd count, weight; Json codes; };
PairsRead pairs_read(const ng::Array& pairs, double minStreamlines, const char* who) {
  const Json pr = producer_of(pairs);
  if (!pr.contains("tiles") || pairs.cols != 6) ng::refuse(std::string(who) + ": pairs is not connectome.tile_pairs' pairs (`nxr-neuro run tile_network` first)");
  PairsRead r;
  r.tiles = pr["tiles"]; r.pairs = pairs.rows; r.codes = pr.value("codes", Json::array());
  if (r.pairs != r.tiles * (r.tiles - 1) / 2) ng::refuse(std::string(who) + ": the pairs are not every pair of the tiles");
  r.count.resize(r.pairs); r.weight.resize(r.pairs);
  for (std::int64_t q = 0; q < r.pairs; ++q) { r.count(q) = cell(pairs, q, 2); r.weight(q) = r.count(q) >= minStreamlines ? r.count(q) : 0.0; }
  return r;
}

// ── connectome.laplacian_eigenmodes ──────────────────────────────────────────────────────────────────────────────
const char* const kModes = "connectome.laplacian_eigenmodes";
ng::NodeSpec laplacian_eigenmodes() {
  ng::NodeSpec n;
  n.name = kModes;
  n.kind = ng::Kind::array;
  n.inputs = {{"pairs", "the tile pairs (connectome.tile_pairs)", false}};
  n.outputs = {{"eigenmodes", "the connectome Laplacian's eigenmodes [T x (1 + T)]: row k mode k, column 0 its eigenvalue, columns 1 .. T its value on each tile", false}};
  n.parameters = {{"min_streamlines", Json(), "a pair is an edge of the network when at least this many streamlines join it (no default; nxr: 10)", "streamlines", {}, false},
                  {"null_tolerance", Json(), "an eigenvalue at most this fraction of the largest is a null mode (one per connected component) (no default)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "once: one dense symmetric eigensolve of a T x T matrix");
  n.matches = "nxr flownet's Lc = diag(Wg.sum(1)) - Wg, Wg the edges' streamline counts, np.linalg.eigh; compute nxr::meeg::laplacianEigenmodes";
  n.description = "the connectome eigenmodes of the tiles: the graph Laplacian of the network's edges weighted by their streamline counts, all its eigenmodes";
  n.plan = [](const ng::Types& in, const Json& p) {
    if (!in.count("pairs")) ng::refuse(std::string(kModes) + ": pairs is not given");
    number_param(p, "min_streamlines", kModes); number_param(p, "null_tolerance", kModes);
    ng::Planned out; out.out["eigenmodes"] = statistic_port(); return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(std::make_pair(handed(c, "pairs", kModes), p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [pairs, p] = std::any_cast<const std::pair<ng::Array, Json>&>(prepared);
    const double minStreamlines = p["min_streamlines"].get<double>(), tolerance = p["null_tolerance"].get<double>();
    const PairsRead r = pairs_read(pairs, minStreamlines, kModes);
    const nxr::meeg::LaplacianModes m = nxr::meeg::laplacianEigenmodes(r.weight, r.tiles, tolerance);
    const std::int64_t T = r.tiles;
    ng::Array a;
    a.rows = T; a.cols = T + 1;
    for (std::int64_t k = 0; k < T; ++k) { a.data.push_back(m.eigenvalues(k)); for (std::int64_t i = 0; i < T; ++i) a.data.push_back(m.eigenvectors(i, k)); }
    std::int64_t edges = 0;
    for (std::int64_t q = 0; q < r.pairs; ++q) edges += r.weight(q) > 0;
    const Json producer = Json{{"tiles", T}, {"edges", edges}, {"min_streamlines", minStreamlines}, {"null_tolerance", tolerance}, {"null_modes", m.nullModes}, {"codes", r.codes},
                               {"weights", "the edges' streamline counts"}, {"complete", true}};
    a.meta["shape"] = Json::array({T, T + 1}); a.meta["producer"] = producer;
    a.meta["describe"] = Json{{"method", "laplacian_eigenmodes"}, {"measure", "eigenmode"}, {"shape", Json::array({T, T + 1})},
                              {"axes", Json::array({indexed_axis("mode", T), indexed_axis("column", T + 1)})}, {"producer", producer},
                              {"description", "row k: mode k (eigenvalues ascending): column 0 its eigenvalue (streamlines), columns 1 .. T its value on each tile (the tiles' table order); orthonormal"}};
    return ng::Arrays{{"eigenmodes", std::move(a)}};
  };
  return n;
}

// ── connectome.tile_operator ─────────────────────────────────────────────────────────────────────────────────────
const char* const kOperator = "connectome.tile_operator";
ng::NodeSpec tile_operator() {
  ng::NodeSpec n;
  n.name = kOperator;
  n.kind = ng::Kind::array;
  n.inputs = {{"kernel", "the inverse kernel (free orientation: x y z rows per vertex of the tiles' surface; its columns its channels)", false},
              {"tiles", "the surface's dyadic grid (a grid selection)", false},
              {"serving", "the operator the chunk spectra were served with (signal.mode_analysis: rows the field coordinates, columns the recording's channels)", false}};
  n.outputs = {{"operator", "the tile operator [3T x K] (rows 3 t + component of the tiles in code order; columns the served coordinates)", false}};
  n.parameters = {{"tiles_per_piece", Json(), "the tiles of each piece, a power of two (as connectome.tile_pairs')", "parts", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "made in double, once a run: a change of coordinates composed once");
  n.linear_map = true;
  n.matches = "nxr flownet's Kt (the kernel averaged over each tile's vertices, three components) applied to the channels, here composed with the served coordinates' synthesis; compute nxr::signal::tileOperator, servedSynthesis";
  n.description = "each tile's current from the served coordinates: the kernel averaged over the tile's vertices, composed with the synthesis of the coordinates back to the kernel's channels (one operator)";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"kernel", "tiles", "serving"}) if (!in.count(k)) ng::refuse(std::string(kOperator) + ": " + k + " is not given");
    number_param(p, "tiles_per_piece", kOperator);
    ng::Planned out; ng::PortType t; t.cls = ng::PortClass::operator_; t.facts["cls"] = "operator"; out.out["operator"] = t; return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(std::make_pair(c, p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [c, p] = std::any_cast<const std::pair<ng::Arrays, Json>&>(prepared);
    const ng::Array& kernel = handed(c, "kernel", kOperator);
    const ng::Array& serving = handed(c, "serving", kOperator);
    const Tiles t = tiles_of(handed(c, "tiles", kOperator), p, kOperator);
    const std::string name = kernel.meta.value("name", std::string());
    if (kernel.meta.value("to_value_type", std::string("vector3")) == "scalar" || kernel.rows % 3 != 0) ng::refuse(std::string(kOperator) + ": " + name + " is not a free-orientation kernel (x y z rows per vertex)");
    if (kernel.rows / 3 != static_cast<std::int64_t>(t.codes.size())) ng::refuse(std::string(kOperator) + ": " + name + " has " + std::to_string(kernel.rows / 3) + " vertices, the tiles' surface " + std::to_string(t.codes.size()));
    const std::vector<std::int32_t> kernelChannels = kernel.meta.value("channels", std::vector<std::int32_t>()), servingChannels = serving.meta.value("channels", std::vector<std::int32_t>());
    if (static_cast<std::int64_t>(kernelChannels.size()) != kernel.cols || static_cast<std::int64_t>(servingChannels.size()) != serving.cols) ng::refuse(std::string(kOperator) + ": an operator's columns are not its channels");
    // the synthesis of the served coordinates back to the serving operator's channels, then its rows put in the kernel's column order (matched by channel)
    const nxr::signal::ServedSynthesis synthesis = nxr::signal::servedSynthesis(matrix_of(serving));
    std::map<std::int32_t, Eigen::Index> servingRow;
    for (std::size_t i = 0; i < servingChannels.size(); ++i) servingRow[servingChannels[i]] = static_cast<Eigen::Index>(i);
    Eigen::MatrixXd byKernelChannel(kernel.cols, serving.rows);
    for (std::size_t j = 0; j < kernelChannels.size(); ++j) {
      const auto it = servingRow.find(kernelChannels[j]);
      if (it == servingRow.end()) ng::refuse(std::string(kOperator) + ": the kernel's channel " + std::to_string(kernelChannels[j]) + " is not one the spectra were served from");
      byKernelChannel.row(static_cast<Eigen::Index>(j)) = synthesis.synthesis.row(it->second);
    }
    if (servingChannels.size() != kernelChannels.size()) ng::refuse(std::string(kOperator) + ": the spectra were served from " + std::to_string(servingChannels.size()) + " channels, the kernel reads " + std::to_string(kernelChannels.size()) + ": they must be the same channels");
    const std::vector<std::int64_t> codes = distinct_codes(t.codes);
    const Eigen::MatrixXd composed = nxr::signal::tileOperator(matrix_of(kernel), tile_rows(t.codes, codes), static_cast<std::int64_t>(codes.size()), byKernelChannel);
    ng::Array op;
    op.rows = composed.rows(); op.cols = composed.cols();
    op.data.resize(static_cast<std::size_t>(op.rows * op.cols));
    Eigen::Map<RowMajor>(op.data.data(), op.rows, op.cols) = composed;
    op.meta["columns"] = Json{{"are", "the served field coordinates"}, {"members", serving.meta.value("modes", Json::array())}};
    op.meta["rows"] = "tile-major: 3 t + component (x y z) of the tiles in code order";
    op.meta["codes"] = codes;
    op.meta["kernel"] = name;
    op.meta["synthesis_rank"] = synthesis.rank;
    op.meta["coordinates"] = serving.rows;
    op.meta["tiles_per_piece"] = t.parts;
    op.meta["level"] = t.level;
    op.meta["sizes"] = Json{{"kernel", Json::array({kernel.rows, kernel.cols})}, {"averaged", Json::array({composed.rows(), kernel.cols})}, {"synthesis", Json::array({kernel.cols, serving.rows})}, {"operator", Json::array({composed.rows(), composed.cols()})}};
    return ng::Arrays{{"operator", std::move(op)}};
  };
  n.record = [](const std::any& prepared) {
    const auto& [c, p] = std::any_cast<const std::pair<ng::Arrays, Json>&>(prepared);
    return Json{{"kernel", c.count("kernel") ? c.at("kernel").meta.value("name", std::string()) : std::string()}, {"tiles_per_piece", p.value("tiles_per_piece", Json())}};
  };
  return n;
}

// ── connectome.tile_flow ─────────────────────────────────────────────────────────────────────────────────────────
struct FlowPrepared { ng::Array spectra, op, frequency; Json p; };
FlowPrepared prepare_flow(const ng::Arrays& c, const Json& p) {
  return FlowPrepared{handed(c, "spectra", "connectome flow"), handed(c, "operator", "connectome flow"), handed(c, "frequency", "connectome flow"), p};
}
const char* const kFlow = "connectome.tile_flow";
ng::NodeSpec tile_flow() {
  ng::NodeSpec n;
  n.name = kFlow;
  n.kind = ng::Kind::array;
  n.inputs = {{"spectra", "the chunk spectra in the served coordinates (signal.chunk_spectra)", false},
              {"operator", "the tile operator (connectome.tile_operator)", false},
              {"frequency", "the atoms: a kernel set on the spectra's frequency line", false}};
  n.outputs = {{"flow", "per atom: Re S_ij and Im S_ij per pair i < j, S_ii per tile (the whole run's tile cross-spectrum, the components traced)", false}};
  n.parameters = {{"frequency_normalisation", Json(), "sum_of_squared_gains | none (as signal.spectral_flow's; no default)", "-", {"sum_of_squared_gains", "none"}, false},
                  {"support_floor", Json(), "with sum_of_squared_gains: as signal.spectral_flow's (no default)", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64,
                             "once a run, chunk by chunk of the stored spectra: the tile operator applied to each chunk's coefficients once, then per atom the weighted sum of products over the bins");
  n.matches = "nxr flownet's Stot (the per-segment C = w wᴴ traced over components, summed): here the same sums from the factored cross-spectrum; compute nxr::signal::accumulateTileCrossSpectra";
  n.description = "the whole run's cross-spectrum of the tiles' currents per atom: the ingredients of the edge flow, the lag and the speed (derived at read)";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"spectra", "operator", "frequency"}) if (!in.count(k)) ng::refuse(std::string(kFlow) + ": " + k + " is not given");
    text_param(p, "frequency_normalisation", kFlow);
    ng::Planned out; out.out["flow"] = statistic_port(); return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(prepare_flow(c, p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const FlowPrepared& pr = std::any_cast<const FlowPrepared&>(prepared);
    const Spectra s = spectra_of(pr.spectra, kFlow);
    check_operator_against_spectra(pr.op, s, kFlow);
    const Atoms atoms = atoms_of(s, pr.frequency, pr.p, kFlow);
    const Eigen::MatrixXd M = matrix_of(pr.op);
    const std::int64_t T = M.rows() / 3, P = T * (T - 1) / 2, A = atoms.gains.rows();
    std::vector<Eigen::MatrixXcd> sums;
    for (std::int64_t t = 0; t < s.chunks; ++t) {
      Eigen::MatrixXcd Y(s.rows, s.bins);
      for (std::int64_t k = 0; k < s.bins; ++k) for (std::int64_t r = 0; r < s.rows; ++r) Y(r, k) = s.at(t, k, r);
      nxr::signal::accumulateTileCrossSpectra(M.cast<cd>() * Y, atoms.gains, sums);              // [3T x B] once per chunk, every atom reads it
    }
    ng::Array out;
    out.rows = A; out.cols = 2 * P + T;
    for (std::int64_t a = 0; a < A; ++a) {
      const Eigen::MatrixXcd& S = sums[static_cast<std::size_t>(a)];
      for (std::int64_t i = 0; i < T; ++i) for (std::int64_t j = i + 1; j < T; ++j) out.data.push_back(S(i, j).real());
      for (std::int64_t i = 0; i < T; ++i) for (std::int64_t j = i + 1; j < T; ++j) out.data.push_back(S(i, j).imag());
      for (std::int64_t i = 0; i < T; ++i) out.data.push_back(S(i, i).real());
    }
    Json centres = Json::array();
    for (const double c : atoms.centres) centres.push_back(c);
    const Json producer = Json{{"tiles", T}, {"pairs", P}, {"atoms", A}, {"frequencies_hz", centres}, {"codes", pr.op.meta.value("codes", Json::array())}, {"kernel", pr.op.meta.value("kernel", std::string())},
                               {"chunks", s.chunks}, {"frequency_normalisation", pr.p["frequency_normalisation"]}, {"warnings", warnings()},
                               {"derived_at_read", Json{{"edge_flow", "Im S_ij / sqrt(S_ii S_jj): > 0 when tile j lags tile i (the phase flows i -> j)"}, {"lag_s", "-arg(S_ij) / (2 pi f)"}, {"speed", "median length / |lag| (an upper bound)"}}}};
    out.meta["shape"] = Json::array({A, 2 * P + T}); out.meta["producer"] = producer;
    out.meta["describe"] = Json{{"method", "tile_flow"}, {"measure", "tile_cross_spectrum"}, {"shape", Json::array({A, 2 * P + T})},
                                {"axes", Json::array({indexed_axis("atom", A, centres), indexed_axis("column", 2 * P + T)})},
                                {"measures", Json::array({Json{{"measure", "cross_spectrum_real"}, {"columns", Json::array({0, P})}}, Json{{"measure", "cross_spectrum_imaginary"}, {"columns", Json::array({P, 2 * P})}},
                                                          Json{{"measure", "power"}, {"columns", Json::array({2 * P, 2 * P + T})}}})},
                                {"producer", producer},
                                {"description", "per atom, the whole run's cross-spectrum of the tiles' currents (Σ over chunks and the atom's bins of g² Σ_c z_i conj(z_j)): Re and Im per pair i < j, then each tile's power"}};
    return ng::Arrays{{"flow", std::move(out)}};
  };
  n.record = [](const std::any& prepared) { return Json{{"kernel", std::any_cast<const FlowPrepared&>(prepared).op.meta.value("kernel", std::string())}}; };
  return n;
}

// ── connectome.tile_time_cells ───────────────────────────────────────────────────────────────────────────────────
const char* const kCells = "connectome.tile_time_cells";
ng::NodeSpec tile_time_cells() {
  ng::NodeSpec n;
  n.name = kCells;
  n.kind = ng::Kind::array;
  n.inputs = {{"spectra", "the chunk spectra in the served coordinates (signal.chunk_spectra)", false},
              {"operator", "the tile operator (connectome.tile_operator)", false},
              {"frequency", "the atoms: Gaussians on the spectra's frequency line (their time width sets the cells)", false}};
  n.outputs = {{"cells", "per atom, per time cell of the atom's own width: Σ Im(z_i conj z_j) per pair i < j, then Σ |z_i|² per tile (the format of signal.atom_time_cells: one band)", false}};
  n.parameters = {{"frequency_normalisation", Json(), "sum_of_squared_gains | none (as connectome.tile_flow's; no default)", "-", {"sum_of_squared_gains", "none"}, false},
                  {"support_floor", Json(), "with sum_of_squared_gains (no default)", "-", {}, true},
                  {"cell_width", Json(), "a time cell's width in the atom's time standard deviations (1 / (2 pi sigma_f)), snapped to the nearest power of two of samples (no default; the flow analysis: 2)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64,
                             "once a run, chunk by chunk of the stored spectra: the tile operator composed on the atom's bins, one inverse transform per row, summed over the cells");
  n.matches = "nxr flownet's per-segment C = w wᴴ (its 4 s segments replaced by the atoms' cells); compute nxr::signal::tileTimeCells";
  n.description = "the tiles' cross-spectrum of each atom in the atom's own time cells: the sums the sign-flip test reads";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"spectra", "operator", "frequency"}) if (!in.count(k)) ng::refuse(std::string(kCells) + ": " + k + " is not given");
    text_param(p, "frequency_normalisation", kCells);
    if (!(number_param(p, "cell_width", kCells) > 0)) ng::refuse(std::string(kCells) + ": cell_width > 0");
    ng::Planned out; out.out["cells"] = statistic_port(); return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(prepare_flow(c, p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const FlowPrepared& pr = std::any_cast<const FlowPrepared&>(prepared);
    const Spectra s = spectra_of(pr.spectra, kCells);
    check_operator_against_spectra(pr.op, s, kCells);
    if (static_cast<std::int64_t>(s.owns.size()) != s.chunks) ng::refuse(std::string(kCells) + ": the spectra do not record each chunk's own samples");
    const Atoms atoms = atoms_of(s, pr.frequency, pr.p, kCells);
    const Eigen::MatrixXd M = matrix_of(pr.op);
    const std::int64_t T = M.rows() / 3, P = T * (T - 1) / 2, A = atoms.gains.rows();
    const double width = pr.p["cell_width"].get<double>();
    ng::Array out;
    out.cols = P + T;
    Json groups = Json::array();
    std::int64_t row = 0;
    for (std::int64_t a = 0; a < A; ++a) {
      const double sd = atoms.standardDeviations[static_cast<std::size_t>(a)];
      if (!std::isfinite(sd)) ng::refuse(std::string(kCells) + ": the time cells need Gaussian atoms on a linear chart");
      const double sigmaT = 1.0 / (2.0 * M_PI * sd);
      const std::int64_t cellSamples = std::int64_t(1) << static_cast<int>(std::lround(std::log2(width * sigmaT * s.sfreq)));
      std::int64_t cells = 0;
      for (std::int64_t t = 0; t < s.chunks; ++t) {
        Eigen::MatrixXcd X(s.rows, s.bins);
        for (std::int64_t k = 0; k < s.bins; ++k) for (std::int64_t r = 0; r < s.rows; ++r) X(r, k) = atoms.gains(a, k) * s.at(t, k, r);
        const nxr::signal::TileCells f = nxr::signal::tileTimeCells(X, s.first, s.transform, s.overlap, s.owns[static_cast<std::size_t>(t)], cellSamples, M);
        for (Eigen::Index j = 0; j < f.power.cols(); ++j) {
          for (std::int64_t q = 0; q < P; ++q) out.data.push_back(f.pairImaginary(q, j));
          for (std::int64_t i = 0; i < T; ++i) out.data.push_back(f.power(i, j));
        }
        cells += f.power.cols();
      }
      groups.push_back(Json{{"atom", a}, {"band", 0}, {"first_row", row}, {"cells", cells}, {"cell_samples", cellSamples}, {"cell_seconds", static_cast<double>(cellSamples) / s.sfreq}, {"frequency_hz", atoms.centres[static_cast<std::size_t>(a)]}});
      row += cells;
    }
    out.rows = row;
    const Json producer = Json{{"tiles", T}, {"pairs", P}, {"atoms", A}, {"codes", pr.op.meta.value("codes", Json::array())}, {"cell_width", width}, {"kernel", pr.op.meta.value("kernel", std::string())}};
    out.meta["shape"] = Json::array({row, P + T}); out.meta["producer"] = producer;
    out.meta["describe"] = Json{{"method", "tile_time_cells"}, {"measure", "time_cell_flow"}, {"shape", Json::array({row, P + T})}, {"axes", Json::array({indexed_axis("cell", row), indexed_axis("column", P + T)})},
                                {"measures", Json::array({Json{{"measure", "phase_current"}, {"columns", Json::array({0, P})}}, Json{{"measure", "power"}, {"columns", Json::array({P, P + T})}}})},
                                {"values_shape", Json::array({row, P + T})}, {"producer", producer},
                                {"attributes", Json{{"groups", groups}, {"rows", "the time cells of each atom, in the groups' order; within a group chunk by chunk, a trailing part cell of a chunk left out"},
                                                    {"columns", "per pair i < j: Σ_c Σ_t Im(z_i conj z_j) (> 0 when tile j lags tile i), then each tile's Σ |z|²"}, {"cell_width", width}, {"sum_over", "the cell's samples (each sample once)"}}}};
    return ng::Arrays{{"cells", std::move(out)}};
  };
  n.record = [](const std::any& prepared) { return Json{{"cell_width", std::any_cast<const FlowPrepared&>(prepared).p.value("cell_width", Json())}}; };
  return n;
}

// ── connectome.network_flow ──────────────────────────────────────────────────────────────────────────────────────
const char* const kNetwork = "connectome.network_flow";
ng::NodeSpec network_flow() {
  ng::NodeSpec n;
  n.name = kNetwork;
  n.kind = ng::Kind::array;
  n.inputs = {{"pairs", "the tile pairs (connectome.tile_pairs)", false},
              {"eigenmodes", "the connectome eigenmodes (connectome.laplacian_eigenmodes), made with the same min_streamlines", false},
              {"flow", "the whole run's tile cross-spectrum per atom (connectome.tile_flow)", false}};
  n.outputs = {{"network", "per atom: divergence per tile, graph potential per tile, its coefficient per connectome eigenmode, Σ w (B phi)², Σ w F²", false}};
  n.parameters = {{"min_streamlines", Json(), "a pair is an edge when at least this many streamlines join it (the eigenmodes' own; no default)", "streamlines", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "once: a derived measure of the stored sums (the edge flow, a ratio), then linear algebra on the T tiles");
  n.matches = "nxr flownet's div = Beᵀ Fj, phi = lstsq(sqrt(w) Be, sqrt(w) Fj), gshare, spec_ = (Uᵀ (phi - mean))²; here phi as the 1/eigenvalue gain on the complete eigenmodes; compute nxr::meeg::networkFlow";
  n.description = "the network's view of the edge flow per atom: the net flow into each tile, the graph potential whose gradient fits the flow best (a filter on the connectome eigenmodes) and its coefficients";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"pairs", "eigenmodes", "flow"}) if (!in.count(k)) ng::refuse(std::string(kNetwork) + ": " + k + " is not given");
    number_param(p, "min_streamlines", kNetwork);
    ng::Planned out; out.out["network"] = statistic_port(); return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) {
    return std::any(std::make_tuple(handed(c, "pairs", kNetwork), handed(c, "eigenmodes", kNetwork), handed(c, "flow", kNetwork), p));
  };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [pairs, eigen, flow, p] = std::any_cast<const std::tuple<ng::Array, ng::Array, ng::Array, Json>&>(prepared);
    const double minStreamlines = p["min_streamlines"].get<double>();
    const PairsRead r = pairs_read(pairs, minStreamlines, kNetwork);
    const Json ep = producer_of(eigen), fp = producer_of(flow);
    if (ep.value("min_streamlines", -1.0) != minStreamlines) ng::refuse(std::string(kNetwork) + ": the eigenmodes were made with min_streamlines " + ep.value("min_streamlines", Json()).dump() + ", not " + std::to_string(minStreamlines));
    if (ep.value("codes", Json()) != r.codes || fp.value("codes", Json()) != r.codes) ng::refuse(std::string(kNetwork) + ": the pairs, the eigenmodes and the flow are not on the same tiles");
    const std::int64_t T = r.tiles, P = r.pairs, A = flow.rows;
    if (eigen.rows != T || eigen.cols != T + 1 || flow.cols != 2 * P + T) ng::refuse(std::string(kNetwork) + ": the eigenmodes or the flow do not fit the pairs' tiles");
    nxr::meeg::LaplacianModes modes;
    modes.eigenvalues.resize(T); modes.eigenvectors.resize(T, T);
    for (std::int64_t k = 0; k < T; ++k) { modes.eigenvalues(k) = cell(eigen, k, 0); for (std::int64_t i = 0; i < T; ++i) modes.eigenvectors(i, k) = cell(eigen, k, 1 + i); }
    modes.nullModes = ep.value("null_modes", std::int64_t(0));
    ng::Array out;
    out.rows = A; out.cols = 3 * T + 2;
    for (std::int64_t a = 0; a < A; ++a) {
      Eigen::VectorXd edgeFlow = Eigen::VectorXd::Zero(P);                                     // derived at read: Im S_ij / sqrt(S_ii S_jj)
      for (std::int64_t i = 0; i < T; ++i)
        for (std::int64_t j = i + 1; j < T; ++j) {
          const std::int64_t q = nxr::meeg::pairIndex(i, j, T);
          const double denominator = std::sqrt(cell(flow, a, 2 * P + i) * cell(flow, a, 2 * P + j));
          edgeFlow(q) = denominator > 0 ? cell(flow, a, P + q) / denominator : 0.0;
        }
      const nxr::meeg::NetworkFlow nf = nxr::meeg::networkFlow(edgeFlow, r.weight, modes);
      for (std::int64_t i = 0; i < T; ++i) out.data.push_back(nf.divergence(i));
      for (std::int64_t i = 0; i < T; ++i) out.data.push_back(nf.potential(i));
      for (std::int64_t k = 0; k < T; ++k) out.data.push_back(nf.coefficients(k));
      out.data.push_back(nf.weightedGradientSquares); out.data.push_back(nf.weightedFlowSquares);
    }
    const Json producer = Json{{"tiles", T}, {"atoms", A}, {"min_streamlines", minStreamlines}, {"null_modes", modes.nullModes}, {"codes", r.codes}, {"frequencies_hz", fp.value("frequencies_hz", Json::array())},
                               {"edge_flow", "Im S_ij / sqrt(S_ii S_jj) of the whole run (connectome.tile_flow)"}, {"weights", "the edges' streamline counts"},
                               {"derived_at_read", Json{{"gradient_share", "Σ w (B phi)² / Σ w F²"}, {"potential_energy_per_mode", "coefficient²"}}}, {"warnings", warnings()}};
    out.meta["shape"] = Json::array({A, 3 * T + 2}); out.meta["producer"] = producer;
    out.meta["describe"] = Json{{"method", "network_flow"}, {"measure", "network_flow"}, {"shape", Json::array({A, 3 * T + 2})},
                                {"axes", Json::array({indexed_axis("atom", A, fp.value("frequencies_hz", Json())), indexed_axis("column", 3 * T + 2)})},
                                {"measures", Json::array({Json{{"measure", "divergence"}, {"columns", Json::array({0, T})}}, Json{{"measure", "potential"}, {"columns", Json::array({T, 2 * T})}},
                                                          Json{{"measure", "potential_coefficient"}, {"columns", Json::array({2 * T, 3 * T})}}, Json{{"measure", "weighted_gradient_squares"}, {"columns", Json::array({3 * T, 3 * T + 1})}},
                                                          Json{{"measure", "weighted_flow_squares"}, {"columns", Json::array({3 * T + 1, 3 * T + 2})}}})},
                                {"producer", producer},
                                {"description", "per atom on the network's edges: divergence Bᵀ F per tile, the graph potential phi (L phi = Bᵀ W F, the 1/eigenvalue gain on the connectome eigenmodes, the null modes left out), its coefficients, the gradient part's and the flow's weighted sums of squares"}};
    return ng::Arrays{{"network", std::move(out)}};
  };
  return n;
}

}  // namespace

void register_connectome_flow_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {fiber_lengths(), tile_pairs(), laplacian_eigenmodes(), tile_operator(), tile_flow(), tile_time_cells(), network_flow()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
