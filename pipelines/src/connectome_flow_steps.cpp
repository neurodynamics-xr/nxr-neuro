// nxr-pipelines — THE FLOW THROUGH THE FIBRE NETWORK AND ITS CONNECTOME EIGENMODES as graph nodes (spec docs/specs/2026-11-24-connectome-flow.md; definitions `tile_network`,
// `connectome_flow`). A port of nxr's flow through the fibre network (flownet): the tiles of the surface's dyadic grid are the network's nodes, tile pairs joined by streamlines
// its edges; each tile's current is the source kernel averaged over its vertices; the tiles' cross-spectrum gives the edge flow; the network's divergence and graph potential
// are read in the connectome's eigenmodes.
//
// THE GRAPH (tile_network). before: fibres --[meeg.fiber_ends, meeg.nearest_vertex, meeg.end_gate]--> each fibre's end vertices (REUSED); fibres --[meeg.fiber_lengths]--> lengths;
//         end vertices + lengths + the grid's tiles at the level a stated scale sets + the surface --[meeg.tile_pairs]--> tiles, pairs (count, median length, distance, across) and
//         the network as a GRAPH manifold (the tiles its vertices at their centroids, the joined pairs its rank-1 cells) with its metric (tile area, edge conductance) as fields.
// THE GRAPH (eigenmodes --graph). graph + area + conductance --[meeg.graph_hodge_dirac]--> the graph's Hodge-Dirac modes in the surfaces' four-class form (exact and cycle modes).
// THE GRAPH (connectome_flow). before: the sensor serving operator (signal.mode_analysis, REUSED) + kernel + tiles --[signal.tile_operator]--> ONE operator [3T x K];
//         stored chunk spectra + operator + atoms --[signal.tile_flow]--> per atom the whole run's tile cross-spectrum (pair sums, tile power);
//         --[signal.tile_time_cells]--> the same per time cell of each atom --[stats.sign_flip_consistency, REUSED]--> per pair R1 and its p;
//         graph + its conductance + its STORED Hodge-Dirac modes + the mean edge flow --[meeg.network_flow]--> per atom divergence, graph potential, the flow's coefficient on every
//         mode, the gradient and cycle parts' sums (no Laplacian is solved here: the modes are the stored solve's).
// Every node is a function of arrays: no file, no subject, no loop over a recording.
#include "nxr/pipelines/connectome_flow_steps.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <limits>
#include <map>
#include <tuple>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "chunk_spectra_readers.h"
#include "nxr/graph/node.h"
#include "flow_measures.h"
#include "nxr/meeg/tile_network.h"
#include "nxr/quantity/quantity.h"
#include "nxr/select/dyadic.h"
#include "nxr/select/grid.h"
#include "nxr/signal/spectral_flow.h"
#include "nxr/signal/flow_readout.h"
#include "nxr/signal/source_flow.h"
#include "nxr/signal/tile_flow.h"
#include "nxr/stats/network_flip.h"

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
ng::PortType measurement_port() { ng::PortType t; t.cls = ng::PortClass::constant; t.facts["cls"] = "measurement"; return t; }
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

// THE FLOW'S MEASUREMENTS over the atoms (a value over a selection's members is a measurement): a per-atom array [atoms x columns] whose columns hold blocks of
// `rows` values, one block a measure, from column `first` -> [rows x atoms x measures], value (r atoms + a) measures + k; its meta (the producer) kept
ng::Array measurement_of_blocks(const ng::Array& perAtom, std::int64_t first, std::int64_t rows, const std::vector<std::string>& measures, const char* rowsAre) {
  const std::int64_t atoms = perAtom.rows, K = static_cast<std::int64_t>(measures.size());
  ng::Array m;
  m.rows = rows * atoms; m.cols = K;
  m.data.assign(static_cast<std::size_t>(rows * atoms * K), 0.0);
  for (std::int64_t r = 0; r < rows; ++r)
    for (std::int64_t a = 0; a < atoms; ++a)
      for (std::int64_t k = 0; k < K; ++k) m.data[static_cast<std::size_t>((r * atoms + a) * K + k)] = cell(perAtom, a, first + k * rows + r);
  if (perAtom.meta.contains("producer")) m.meta["producer"] = perAtom.meta["producer"];
  Json ms = Json::array();
  for (const auto& name : measures) ms.push_back(Json{{"measure", name}});
  m.meta["describe"] = Json{{"measures", ms}, {"shape", Json::array({rows, atoms, K})}, {"values_shape", Json::array({rows, atoms})},
                            {"attributes", Json{{"rows", rowsAre}, {"members", "the atoms"}}}};
  declare_flow_measures(m.meta["describe"]["measures"]);   // the flow measures it writes, declared (flow_measures.h)
  return m;
}
// one value of a flow measurement: in a run [rows x atoms x measures] (its describe's shape), or a STORED measurement read by its measure [rows x atoms]
double measured(const ng::Array& a, std::int64_t row, std::int64_t atom, std::int64_t measure) {
  if (a.meta.contains("measure") && !a.meta.contains("describe")) return a.at(static_cast<std::size_t>(row * a.cols + atom));
  const Json shape = a.meta.at("describe").at("shape");
  const std::int64_t atoms = shape[1].get<std::int64_t>(), K = shape[2].get<std::int64_t>();
  return a.at(static_cast<std::size_t>((row * atoms + atom) * K + measure));
}
// the name of a flow measurement's first measure: in a run its describe's first, stored its meta's
std::string first_measure(const ng::Array& a) {
  if (a.meta.contains("measure") && !a.meta.contains("describe")) return a.meta["measure"].get<std::string>();
  const Json d = a.meta.value("describe", Json::object());
  return d.contains("measures") && !d["measures"].empty() ? d["measures"][0].value("measure", std::string()) : std::string();
}
// the atoms and rows of a flow measurement
std::pair<std::int64_t, std::int64_t> atoms_rows(const ng::Array& a) {
  if (a.meta.contains("measure") && !a.meta.contains("describe")) return {a.cols, a.rows};
  const Json shape = a.meta.at("describe").at("shape");
  return {shape[1].get<std::int64_t>(), shape[0].get<std::int64_t>()};
}

// ── the tiles: the grid's finest codes shifted to the level a stated SCALE sets (a quantity: "parts 32" per piece, a tile diameter "20 mm", an area "area 400 mm^2") ──
struct Tiles { std::vector<std::int64_t> codes; int level = 0, depth = 0; std::int64_t parts = 0; Json scale; };
Tiles tiles_of(const ng::Array& grid, const Json& p, const char* who) {
  if (!grid.meta.contains("depth")) ng::refuse(std::string(who) + ": tiles is not a dyadic grid with levels (`nxr-neuro run select_dyadic_surface` first)");
  Tiles t;
  t.depth = grid.meta["depth"].get<int>();
  if (!p.contains("scale") || !p["scale"].is_string()) ng::refuse(std::string(who) + ": scale has no default: name the tiles' level as a quantity (\"parts 32\", \"20 mm\", \"area 400 mm^2\")");
  nxr::quantity::Quantity q;
  try { q = nxr::quantity::parse(p["scale"].get<std::string>()); } catch (const std::exception& e) { ng::refuse(std::string(who) + ": scale: " + e.what()); }
  nxr::select::LevelOf lo;
  try {
    if (q.chart == "parts") lo = nxr::select::level_of_parts(q.hi, t.depth);
    else if (q.chart == "area" || q.chart == "diameter") {
      const Json params = grid.meta.value("params", Json::object());
      const Json g = params.value("grid", Json::object());
      if (q.chart == "area") {                                                    // each level's mean tile area: a piece's area halved per level (the grid's record: its area and pieces)
        const double area = params.contains("area_m2") ? params["area_m2"].get<double>() : g.value("area_m2", 0.0);
        const std::int64_t pieces = g.value("pieces", grid.meta.value("slots", std::int64_t(1)));
        if (!(area > 0)) ng::refuse(std::string(who) + ": the grid records no area: name the scale in parts (\"parts 32\") or as a diameter");
        std::vector<double> areas;
        for (int l = 0; l <= t.depth; ++l) areas.push_back(area / static_cast<double>(std::max<std::int64_t>(1, pieces)) / std::ldexp(1.0, l) * 1e6);   // mm^2
        lo = nxr::select::level_of_area(areas, q.hi * nxr::quantity::unit_factor(q.unit, "mm^2"), "mm");
      } else {                                                                    // the level dictionary's measured mean tile diameter (mm)
        std::vector<double> diameters;
        for (const auto& l : grid.meta.value("levels", Json::array())) diameters.push_back(l.value("diameter", std::nan("")));
        if (static_cast<int>(diameters.size()) != t.depth + 1) ng::refuse(std::string(who) + ": the grid's level dictionary records no tile diameters: name the scale in parts");
        lo = nxr::select::level_of_diameter(diameters, q.hi * nxr::quantity::unit_factor(q.unit, "mm"), "mm");
      }
    } else ng::refuse(std::string(who) + ": scale " + p["scale"].dump() + " is in the chart " + q.chart + ": parts, a tile diameter (mm) or an area (mm^2)");
  } catch (const std::invalid_argument& e) { ng::refuse(std::string(who) + ": scale: " + e.what()); }
  t.level = lo.level;
  t.parts = std::int64_t(1) << t.level;
  t.scale = Json{{"stated", p["scale"]}, {"level", t.level}, {"snapped", lo.snapped}, {"unit", lo.unit}, {"parts_per_piece", t.parts}};
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

// ── the stored chunk spectra and the atoms: the sensor chain's readers (chunk_spectra_readers.h), the same rules ──────────────────────────────
using Spectra = spectra_readers::Spectra;
using spectra_readers::spectra_of;
// the atoms' gains per bin (a kernel set: nxr::signal::kernelGains), normalised as the parameter says; their centres and time standard deviations (Gaussians on a linear chart)
// every pair of tiles i < j, i the slower index (the order of the stored pair columns): [P x 2] flat
std::vector<std::int32_t> tile_pairs(std::int64_t tiles) {
  std::vector<std::int32_t> pairs;
  for (std::int32_t i = 0; i < tiles; ++i) for (std::int32_t j = i + 1; j < tiles; ++j) { pairs.push_back(i); pairs.push_back(j); }
  return pairs;
}
struct Atoms { Eigen::MatrixXd gains; std::vector<double> centres, standardDeviations; };
Atoms atoms_of(const Spectra& s, const ng::Array& sel, const Json& p, const char* who) {
  Eigen::VectorXd frequency(s.bins);
  for (std::int64_t i = 0; i < s.bins; ++i) frequency(i) = s.frequency(i);
  const nxr::signal::KernelGains kernels = spectra_readers::kernel_gains_of(sel, frequency, who, "frequency");
  Atoms a;
  a.gains = kernels.gains;
  const bool timeWidths = kernels.family == "gaussian" && kernels.chart == "linear";
  for (Eigen::Index m = 0; m < kernels.centres.size(); ++m) {
    a.centres.push_back(kernels.centres(m));
    a.standardDeviations.push_back(timeWidths ? kernels.fullWidths(m) / (2.0 * std::sqrt(2.0 * std::log(2.0))) : std::numeric_limits<double>::quiet_NaN());
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

// ── meeg.tile_pairs ────────────────────────────────────────────────────────────────────────────────────────
// connected components of a graph of `n` vertices (union-find over the edges)
int components_of(std::int64_t n, const std::vector<std::int32_t>& edges) {
  std::vector<std::int64_t> parent(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) parent[static_cast<std::size_t>(i)] = i;
  std::function<std::int64_t(std::int64_t)> root = [&](std::int64_t i) { while (parent[static_cast<std::size_t>(i)] != i) i = parent[static_cast<std::size_t>(i)] = parent[static_cast<std::size_t>(parent[static_cast<std::size_t>(i)])]; return i; };
  int c = static_cast<int>(n);
  for (std::size_t e = 0; e + 1 < edges.size(); e += 2) { const auto a = root(edges[e]), b = root(edges[e + 1]); if (a != b) { parent[static_cast<std::size_t>(a)] = b; --c; } }
  return c;
}
ng::Array one_column(std::vector<double> v, const Json& describe) { ng::Array a; a.rows = static_cast<std::int64_t>(v.size()); a.cols = 1; a.data = std::move(v); a.meta["describe"] = describe; return a; }

const char* const kPairs = "meeg.tile_pairs";
ng::NodeSpec tile_pairs() {
  ng::NodeSpec n;
  n.name = kPairs;
  n.kind = ng::Kind::array;
  n.inputs = {{"end_vertex", "each fibre's two end vertices on the surface [2nF], block order (meeg.nearest_vertex)", false},
              {"keep", "the gate: 1 for a fibre kept [nF] (meeg.end_gate)", true},
              {"lengths", "each fibre's length [nF] (meeg.fiber_lengths)", true},
              {"tiles", "the surface's dyadic grid (a grid selection: its finest code per vertex, its depth, its level dictionary)", false},
              {"surface", "the surface the tiles and the ends are on (its vertices and faces)", false}};
  n.outputs = {{"tiles_table", "per tile: code, piece, vertices, centroid x y z", false},
               {"pairs", "per pair of tiles i < j: first, second (tile rows), streamline count, median length, centroid distance, across the pieces", false},
               {"graph", "the network as a GRAPH manifold: the tiles its vertices at their centroids [T x 3], the pairs joined by at least min_streamlines its edges (rank-1 cells, in pair order)", false},
               {"area", "each tile's area (the sum of its vertices' barycentric areas): the node weight, on the graph's vertices", false},
               {"conductance", "each edge's streamline count / median length: the edge weight, on the graph's edges", false},
               {"streamlines", "each edge's streamline count, on the graph's edges", false},
               {"median_length", "each edge's streamlines' median length, on the graph's edges", false}};
  n.parameters = {{"scale", Json(), "the tiles' level as a quantity: parts per piece (\"parts 32\", a power of two), a tile diameter (\"20 mm\") or an area (\"area 400 mm^2\") (no default; nxr's flow through the fibres: parts 32)", "-", {}, false},
                  {"min_streamlines", Json(), "a pair of tiles is an edge of the graph when at least this many streamlines join it (no default; nxr: 10)", "streamlines", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "no time axis: runs once; the fibres are summed in any order");
  n.matches = "nxr flownet's network (tiles of surface_ladder at --level; CNT and LEN over the tile pairs, centroid distances, across the hemispheres); compute nxr::meeg::tilePairs, nxr::select::bisection::vertexAreas";
  n.description = "the fibre network on the tiles: per pair of tiles the streamlines joining them, their median length, the tiles' distance and whether they lie on different pieces; the joined pairs as a graph with its metric";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"end_vertex", "tiles", "surface"}) if (!in.count(k)) ng::refuse(std::string(kPairs) + ": " + k + " is not given");
    if (!p.contains("scale") || !p["scale"].is_string()) ng::refuse(std::string(kPairs) + ": scale has no default: name it (a quantity, \"parts 32\")");
    number_param(p, "min_streamlines", kPairs);
    ng::Planned out;
    out.out["tiles_table"] = statistic_port(); out.out["pairs"] = statistic_port();
    ng::PortType g; g.cls = ng::PortClass::constant; g.facts["cls"] = "manifold"; out.out["graph"] = g;
    for (const char* f : {"area", "conductance", "streamlines", "median_length"}) { ng::PortType t; t.cls = ng::PortClass::constant; t.facts["cls"] = "field"; out.out[f] = t; }
    return out;
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
    // THE GRAPH: the joined pairs (count >= min_streamlines) its edges, in pair order; its metric: each tile's area (the barycentric areas of its vertices), each edge's conductance
    const double minStreamlines = p["min_streamlines"].get<double>();
    std::vector<std::int32_t> edges;
    std::vector<double> conductance, streamlines, medians;
    for (std::int64_t q = 0; q < P; ++q)
      if (net.count(q) >= minStreamlines && net.count(q) > 0) {
        if (!(net.medianLength(q) > 0)) ng::refuse(std::string(kPairs) + ": a joined pair's median length is not positive (the fibres' lengths were not handed over)");
        edges.push_back(net.first[static_cast<std::size_t>(q)]); edges.push_back(net.second[static_cast<std::size_t>(q)]);
        streamlines.push_back(net.count(q)); medians.push_back(net.medianLength(q)); conductance.push_back(net.count(q) / net.medianLength(q));
      }
    nxr::select::bisection::Mesh mesh;
    mesh.positions = S.values(); mesh.n = static_cast<int>(S.rows);
    for (const auto& f : S.meta.value("faces", Json::array())) mesh.faces.push_back(f.get<std::int32_t>());
    if (mesh.faces.empty()) ng::refuse(std::string(kPairs) + ": the surface has no faces (its tiles' areas need them)");
    const std::vector<double> vertexArea = nxr::select::bisection::vertexAreas(mesh);
    std::map<std::int64_t, std::int64_t> row;
    for (std::int64_t i = 0; i < T; ++i) row[net.codes[static_cast<std::size_t>(i)]] = i;
    std::vector<double> tileArea(static_cast<std::size_t>(T), 0.0);
    for (std::size_t v = 0; v < t.codes.size(); ++v) if (t.codes[v] >= 0) { const auto it = row.find(t.codes[v]); if (it != row.end()) tileArea[static_cast<std::size_t>(it->second)] += vertexArea[v]; }
    const int components = components_of(T, edges);
    const std::int64_t E = static_cast<std::int64_t>(conductance.size());
    Json codes = Json::array();
    for (const auto code : net.codes) codes.push_back(code);
    const std::string unit = S.meta.value("unit", std::string("m"));
    const Json producer = Json{{"scale", t.scale}, {"level", t.level}, {"grid_depth", t.depth}, {"tiles", T}, {"pairs", P}, {"codes", codes}, {"min_streamlines", minStreamlines},
                               {"edges", E}, {"components", components},
                               {"metric", Json{{"node_weight", "each tile's area (the sum of its vertices' barycentric areas), " + unit + "^2: the graph's star0"},
                                               {"edge_weight", "each edge's conductance, streamline count / median length (1/" + unit + "): the graph's star1"}}},
                               {"fibres_left_out", Json{{"an_end_on_no_tile", net.endsOutside}, {"both_ends_in_one_tile", net.sameTile}, {"gated_out", net.gatedOut}}},
                               {"length_unit", "the fibres' points' unit (SCS metres)"}, {"distance_unit", "the surface's positions' unit"}, {"warnings", warnings()}};
    // the statistics' measures and method are not in the statistics registry: DECLARED by their writer (the producer's `declared`, D172)
    const Json method = Json{{"description", "the fibre network on a surface's dyadic tiles: each fibre's two ends on the nearest vertices, counted per pair of tiles (nxr's flow through the fibres)"}};
    Json tilesProducer = producer, pairsProducer = producer;
    tilesProducer["declared"] = Json{{"measure", Json{{"description", "a tile of the network: its code, piece, vertex count and centroid (columns)"}, {"unit_rule", "per column: none, none, vertices, the surface's unit"}}}, {"method", method}};
    pairsProducer["declared"] = Json{{"measure", Json{{"description", "a pair of tiles: its streamline count, their median length, the centroids' distance and whether the tiles lie on different pieces (columns)"},
                                                     {"unit_rule", "per column: none, none, streamlines, the fibres' unit, the surface's unit, none"}}}, {"method", method}};
    ng::Array tiles = table(tc, T);
    tiles.meta["shape"] = Json::array({T, 6}); tiles.meta["producer"] = tilesProducer;
    tiles.meta["describe"] = Json{{"method", "tile_pairs"}, {"measure", "tile"}, {"shape", Json::array({T, 6})},
                                  {"axes", Json::array({indexed_axis("tile", T, codes), indexed_axis("column", 6, Json::array({"code", "piece", "vertices", "centroid_x", "centroid_y", "centroid_z"}))})},
                                  {"producer", tilesProducer}, {"description", "the tiles at the stated level: code (2^level x piece + tile), piece, vertex count, centroid (the mean of the vertices' positions)"}};
    ng::Array pairs = table(pc, P);
    pairs.meta["shape"] = Json::array({P, 6}); pairs.meta["producer"] = pairsProducer;
    pairs.meta["describe"] = Json{{"method", "tile_pairs"}, {"measure", "fiber_count"}, {"shape", Json::array({P, 6})},
                                  {"axes", Json::array({indexed_axis("pair", P), indexed_axis("column", 6, Json::array({"first", "second", "fiber_count", "median_length", "distance", "across"}))})},
                                  {"producer", pairsProducer},
                                  {"description", "every pair of tiles i < j (rows of the tiles' table, in the order (0,1), (0,2), ..., (1,2), ...): the streamlines with one end in each, their median length (NaN: none), the centroids' distance, 1 when on different pieces"}};
    ng::Array graph;
    graph.rows = T; graph.cols = 3;
    for (std::int64_t i = 0; i < T; ++i) for (int k = 0; k < 3; ++k) graph.data.push_back(net.centroids(i, k));
    graph.meta["describe"] = Json{{"edges", edges}, {"components", components}, {"producer", producer},
                                  {"description", "the fibre network: the tiles of the surface's grid at level " + std::to_string(t.level) + " (its vertices, at the tiles' centroids) and the pairs joined by at least " +
                                                      std::to_string(static_cast<std::int64_t>(minStreamlines)) + " streamlines (its edges, tail < tip)"}};
    auto field = [&](std::vector<double> v, const char* cell, const std::string& u, const std::string& what) {
      return one_column(std::move(v), Json{{"value_type", "scalar"}, {"cell", cell}, {"unit", u}, {"description", what}, {"producer", producer}});
    };
    return ng::Arrays{{"tiles_table", std::move(tiles)}, {"pairs", std::move(pairs)}, {"graph", std::move(graph)},
                      {"area", field(tileArea, "0", unit + "^2", "each tile's area, the sum of its vertices' barycentric areas: the graph's node weight (star0)")},
                      {"conductance", field(conductance, "1", "1/" + unit, "each edge's streamline count / median length: the graph's edge weight (star1)")},
                      {"streamlines", field(streamlines, "1", "streamlines", "each edge's streamline count")},
                      {"median_length", field(medians, "1", unit, "each edge's streamlines' median length")}};
  };
  n.record = [](const std::any& prepared) { const Json& p = std::any_cast<const std::pair<ng::Arrays, Json>&>(prepared).second; return Json{{"scale", p.value("scale", Json())}, {"min_streamlines", p.value("min_streamlines", Json())}}; };
  return n;
}

// the pairs table, read fresh or from the store: its tiles count and each pair's weight (count when >= min_streamlines, else 0)
struct PairsRead { std::int64_t tiles = 0, pairs = 0; Eigen::VectorXd count, weight; Json codes; };
PairsRead pairs_read(const ng::Array& pairs, double minStreamlines, const char* who) {
  const Json pr = producer_of(pairs);
  if (!pr.contains("tiles") || pairs.cols != 6) ng::refuse(std::string(who) + ": pairs is not meeg.tile_pairs' pairs (`nxr-neuro run tile_network` first)");
  PairsRead r;
  r.tiles = pr["tiles"]; r.pairs = pairs.rows; r.codes = pr.value("codes", Json::array());
  if (r.pairs != r.tiles * (r.tiles - 1) / 2) ng::refuse(std::string(who) + ": the pairs are not every pair of the tiles");
  r.count.resize(r.pairs); r.weight.resize(r.pairs);
  for (std::int64_t q = 0; q < r.pairs; ++q) { r.count(q) = cell(pairs, q, 2); r.weight(q) = r.count(q) >= minStreamlines ? r.count(q) : 0.0; }
  return r;
}

// ── meeg.graph_hodge_dirac ──────────────────────────────────────────────────────────────────────────────
// a graph handed by the runner (its vertices [V x 3], its edges in meta) and a field on its cells, checked against it
std::vector<std::int32_t> graph_edges(const ng::Array& g, const char* who) {
  if (g.meta.value("type", std::string()) != "graph" || !g.meta.contains("edges")) ng::refuse(std::string(who) + ": the network is not a graph manifold (`nxr-neuro run tile_network` stores it)");
  return g.meta["edges"].get<std::vector<std::int32_t>>();
}
Eigen::VectorXd cell_values(const ng::Array& f, std::int64_t n, const char* port, const char* who) {
  if (f.size() != static_cast<std::size_t>(n)) ng::refuse(std::string(who) + ": " + port + " has " + std::to_string(f.size()) + " values, the graph " + std::to_string(n) + " cells of its rank");
  const std::vector<double> v = f.values();
  return Eigen::Map<const Eigen::VectorXd>(v.data(), n);
}
const char* const kGraphModes = "meeg.graph_hodge_dirac";
ng::NodeSpec graph_hodge_dirac() {
  ng::NodeSpec n;
  n.name = kGraphModes;
  n.kind = ng::Kind::array;
  n.inputs = {{"graph", "a graph manifold (vertices and stored rank-1 cells)", false},
              {"area", "the node weights: a field on the graph's vertices (star0)", false},
              {"conductance", "the edge weights: a field on the graph's edges (star1)", false}};
  n.outputs = {{"modes", "the eigenvalue line: mu = sqrt(lambda) per mode, ascending (the cycle modes at 0 first)", false},
               {"eigenvalues", "lambda = mu^2 per mode", false},
               {"kinds", "a labelling of the line: exact (0) or harmonic, a cycle (1)", false},
               {"eigenvectors", "ONE operator from the line to the graph's cells 0+1: each mode's mixed form [(V + E) x K], orthonormal under the stars", false}};
  n.parameters = {{"null_tolerance", Json(), "an eigenvalue at most this fraction of the largest is a constant (one per connected component), left out (no default)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "once: one dense generalized symmetric eigensolve of V x V and a rank-revealing QR of E x V");
  n.matches = "the surfaces' Hodge-Dirac solve (operators.hodge_dirac: the mixed forms, rows by rank, orthonormal under the stars) on a 1-complex; nxr flownet's connectome eigenmodes are the unit-weight node part; compute nxr::meeg::graphHodgeDirac";
  n.description = "the Hodge-Dirac modes of a graph under its node and edge weights: the exact (gradient) modes and the harmonic (cycle) modes, stored as the surfaces' solve is";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"graph", "area", "conductance"}) if (!in.count(k)) ng::refuse(std::string(kGraphModes) + ": " + k + " is not given");
    number_param(p, "null_tolerance", kGraphModes);
    ng::Planned out;
    for (const auto& [o, cls] : std::vector<std::pair<const char*, const char*>>{{"modes", "manifold"}, {"eigenvalues", "field"}, {"kinds", "selection"}, {"eigenvectors", "operator"}}) {
      ng::PortType t; t.cls = std::string(cls) == "operator" ? ng::PortClass::operator_ : ng::PortClass::constant; t.facts["cls"] = cls; out.out[o] = t;
    }
    return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(std::make_pair(c, p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [c, p] = std::any_cast<const std::pair<ng::Arrays, Json>&>(prepared);
    const ng::Array& g = handed(c, "graph", kGraphModes);
    const std::vector<std::int32_t> edges = graph_edges(g, kGraphModes);
    const std::int64_t V = g.rows, E = static_cast<std::int64_t>(edges.size() / 2);
    const Eigen::VectorXd area = cell_values(handed(c, "area", kGraphModes), V, "area", kGraphModes), conductance = cell_values(handed(c, "conductance", kGraphModes), E, "conductance", kGraphModes);
    const double tolerance = p["null_tolerance"].get<double>();
    nxr::meeg::GraphModes m;
    try { m = nxr::meeg::graphHodgeDirac(edges, V, area, conductance, tolerance); } catch (const std::invalid_argument& e) { ng::refuse(std::string(kGraphModes) + ": " + e.what()); }
    const std::int64_t K = m.mu.size();
    if (K == 0) ng::refuse(std::string(kGraphModes) + ": the graph has no mode (no edge)");
    const std::string name = g.meta.value("name", std::string()), stem = "hodgeDirac__" + name;
    const Json producer = Json{{"graph", name}, {"vertices", V}, {"edges", E}, {"components", m.components}, {"cycles", m.cycles}, {"exact", K - m.cycles}, {"null_tolerance", tolerance},
                               {"metric", Json{{"star0", c.at("area").meta.value("name", std::string("area"))}, {"star1", c.at("conductance").meta.value("name", std::string("conductance"))}}},
                               {"solve", "exact: B^T *1 B psi = lambda *0 psi (dense generalized symmetric), the constants out; cycles: the kernel of B^T *1 (rank-revealing QR), *1-orthonormal"}};
    ng::Array line;
    line.rows = K; line.cols = 1;
    line.data.assign(m.mu.data(), m.mu.data() + K);
    line.meta["describe"] = Json{{"cs", "eigenvalue:" + stem}, {"operator", "hodgeDirac"}, {"producer", producer},
                                 {"description", "the eigenvalues mu = sqrt(lambda) >= 0 of the Hodge-Dirac operator d + delta on the graph " + name + ", ascending (its cycle modes at 0): the eigenvalue line"}};
    ng::Array lambda;
    lambda.rows = K; lambda.cols = 1;
    for (std::int64_t k = 0; k < K; ++k) lambda.data.push_back(m.mu(k) * m.mu(k));
    lambda.meta["describe"] = Json{{"value_type", "scalar"}, {"precision", "float64"}, {"description", "per mode: the eigenvalue lambda = mu^2 (0 on a cycle mode)"}};
    ng::Array kinds;
    kinds.rows = K; kinds.cols = 1;
    for (const int k : m.kind) kinds.data.push_back(k);
    kinds.meta["describe"] = Json{{"entries", Json::array({"exact", "harmonic"})}, {"description", "the kind of each mode: exact (label 0, a gradient: a 0-form potential) or harmonic (label 1, a cycle of the graph; no coexact mode on a graph)"}};
    ng::Array vectors;
    vectors.rows = V + E; vectors.cols = K;
    vectors.data.resize(static_cast<std::size_t>((V + E) * K));
    Eigen::Map<RowMajor>(vectors.data.data(), V + E, K) = m.forms;
    vectors.meta["describe"] = Json{{"precision", "float64"}, {"from_value_type", "scalar"}, {"to_value_type", "scalar"}, {"from_cell", "0"}, {"to_cell", "0+1"}, {"producer", producer},
                                    {"params", Json{{"rows", "0-forms [0, " + std::to_string(V) + ") · 1-forms [" + std::to_string(V) + ", " + std::to_string(V + E) + ")"},
                                                    {"forms", "each mode's mixed form, orthonormal under the stars *0 (the area) and *1 (the conductance): exact [psi/sqrt2 ; B psi/(mu sqrt2)], harmonic [0 ; h]"},
                                                    {"analysis", "the line <- graph map is this operator's D106 pair (its star-adjoint), not stored"}}},
                                    {"description", "Hodge-Dirac modes as mixed forms on the graph " + name + ": rows 0-forms (the vertices), 1-forms (the edges), one column a mode"}};
    return ng::Arrays{{"modes", std::move(line)}, {"eigenvalues", std::move(lambda)}, {"kinds", std::move(kinds)}, {"eigenvectors", std::move(vectors)}};
  };
  n.record = [](const std::any& prepared) { return Json{{"null_tolerance", std::any_cast<const std::pair<ng::Arrays, Json>&>(prepared).second.value("null_tolerance", Json())}}; };
  return n;
}

// ── signal.tile_operator ─────────────────────────────────────────────────────────────────────────────────────
const char* const kOperator = "signal.tile_operator";
ng::NodeSpec tile_operator() {
  ng::NodeSpec n;
  n.name = kOperator;
  n.kind = ng::Kind::array;
  n.inputs = {{"kernel", "the inverse kernel (free orientation: x y z rows per vertex of the tiles' surface; its columns its channels)", false},
              {"tiles", "the surface's dyadic grid (a grid selection)", false},
              {"serving", "the operator the chunk spectra were served with (signal.mode_analysis: rows the field coordinates, columns the recording's channels)", false}};
  n.outputs = {{"operator", "the tile operator [3T x K] (rows 3 t + component of the tiles in code order; columns the served coordinates)", false}};
  n.parameters = {{"scale", Json(), "the tiles' level as a quantity (the network's: meeg.tile_pairs')", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "made in double, once a run: a change of coordinates composed once");
  n.linear_map = true;
  n.matches = "nxr flownet's Kt (the kernel averaged over each tile's vertices, three components) applied to the channels, here composed with the served coordinates' synthesis; compute nxr::signal::tileOperator, servedSynthesis";
  n.description = "each tile's current from the served coordinates: the kernel averaged over the tile's vertices, composed with the synthesis of the coordinates back to the kernel's channels (one operator)";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"kernel", "tiles", "serving"}) if (!in.count(k)) ng::refuse(std::string(kOperator) + ": " + k + " is not given");
    text_param(p, "scale", kOperator);
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
    op.meta["scale"] = t.scale;
    op.meta["level"] = t.level;
    op.meta["sizes"] = Json{{"kernel", Json::array({kernel.rows, kernel.cols})}, {"averaged", Json::array({composed.rows(), kernel.cols})}, {"synthesis", Json::array({kernel.cols, serving.rows})}, {"operator", Json::array({composed.rows(), composed.cols()})}};
    return ng::Arrays{{"operator", std::move(op)}};
  };
  n.record = [](const std::any& prepared) {
    const auto& [c, p] = std::any_cast<const std::pair<ng::Arrays, Json>&>(prepared);
    return Json{{"kernel", c.count("kernel") ? c.at("kernel").meta.value("name", std::string()) : std::string()}, {"scale", p.value("scale", Json())}};
  };
  return n;
}

// ── signal.tile_flow ─────────────────────────────────────────────────────────────────────────────────────────
struct FlowPrepared { ng::Array op, frequency; Json p; };   // the spectra are read in place from the run's `in` (never copied)
FlowPrepared prepare_flow(const ng::Arrays& c, const Json& p) {
  handed(c, "spectra", "connectome flow");
  return FlowPrepared{handed(c, "operator", "connectome flow"), handed(c, "frequency", "connectome flow"), p};
}
const char* const kFlow = "signal.tile_flow";
ng::NodeSpec tile_flow() {
  ng::NodeSpec n;
  n.name = kFlow;
  n.kind = ng::Kind::array;
  n.inputs = {{"spectra", "the chunk spectra in the served coordinates (signal.chunk_spectra)", false},
              {"operator", "the tile operator (signal.tile_operator)", false},
              {"frequency", "the atoms: a kernel set on the spectra's frequency line", false}};
  n.outputs = {{"pair_flow", "a MEASUREMENT over the atoms: per tile pair i < j, Re S_ij and Im S_ij (the whole run's tile cross-spectrum, the components traced)", false},
               {"tile_power", "a MEASUREMENT over the atoms: per tile, S_ii", false}};
  n.parameters = {{"frequency_normalisation", Json(), "sum_of_squared_gains | none (as signal.spectral_flow's; no default)", "-", {"sum_of_squared_gains", "none"}, false},
                  {"support_floor", Json(), "with sum_of_squared_gains: as signal.spectral_flow's (no default)", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64,
                             "once a run, chunk by chunk of the stored spectra: the tile operator applied to each chunk's coefficients once, then per atom the weighted sum of products over the bins");
  n.matches = "nxr flownet's Stot (the per-segment C = w wᴴ traced over components, summed): here the same sums from the factored cross-spectrum; compute nxr::signal::pairFlow";
  n.description = "the whole run's cross-spectrum of the tiles' currents per atom: the ingredients of the edge flow, the lag and the speed (derived at read)";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"spectra", "operator", "frequency"}) if (!in.count(k)) ng::refuse(std::string(kFlow) + ": " + k + " is not given");
    text_param(p, "frequency_normalisation", kFlow);
    ng::Planned out; out.out["pair_flow"] = measurement_port(); out.out["tile_power"] = measurement_port(); return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(prepare_flow(c, p)); };
  n.run = [](const std::any& prepared, const ng::Arrays& in) {
    const FlowPrepared& pr = std::any_cast<const FlowPrepared&>(prepared);
    const Spectra s = spectra_of(handed(in, "spectra", "connectome flow"), kFlow);
    check_operator_against_spectra(pr.op, s, kFlow);
    const Atoms atoms = atoms_of(s, pr.frequency, pr.p, kFlow);
    const Eigen::MatrixXd M = matrix_of(pr.op);
    const std::int64_t T = M.rows() / 3, P = T * (T - 1) / 2, A = atoms.gains.rows();
    const std::vector<Eigen::MatrixXd> tileReadout{nxr::signal::componentMajor(M, 3)};       // the tile operator, component-major: one readout of three components
    const std::vector<std::int32_t> tilePairs = tile_pairs(T);
    ng::Array out;
    out.rows = A; out.cols = 2 * P + T;
    for (std::int64_t a = 0; a < A; ++a) {
      // the atom's weighting of the cross-spectrum, factored (C = Y Yᴴ, a SUM over chunks and the atom's bins), read through the tiles: Σ conj(z_i) z_j = conj(S_ij)
      const nxr::signal::PairFlow flow = nxr::signal::pairFlow(spectra_readers::weighted_columns(s, atoms.gains.row(a), 0.0), tileReadout, 3, tilePairs, nullptr);
      for (std::int64_t q = 0; q < P; ++q) out.data.push_back(flow.pairReal(q, 0));
      for (std::int64_t q = 0; q < P; ++q) out.data.push_back(-flow.pairImaginary(q, 0));          // S_ij = Σ z_i conj(z_j)
      for (std::int64_t i = 0; i < T; ++i) out.data.push_back(flow.power(i, 0));
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
    declare_flow_measures(out.meta["describe"]["measures"]);   // the flow measures it writes, declared (flow_measures.h)
    return ng::Arrays{{"pair_flow", measurement_of_blocks(out, 0, P, {"cross_spectrum_real", "cross_spectrum_imaginary"}, "the tile pairs i < j (S_ij = sum z_i conj(z_j); Im > 0 when tile j lags tile i)")},
                      {"tile_power", measurement_of_blocks(out, 2 * P, T, {"power"}, "the tiles (S_ii)")}};
  };
  n.record = [](const std::any& prepared) { return Json{{"kernel", std::any_cast<const FlowPrepared&>(prepared).op.meta.value("kernel", std::string())}}; };
  return n;
}

// ── signal.tile_time_cells ───────────────────────────────────────────────────────────────────────────────────
const char* const kCells = "signal.tile_time_cells";
ng::NodeSpec tile_time_cells() {
  ng::NodeSpec n;
  n.name = kCells;
  n.kind = ng::Kind::array;
  n.inputs = {{"spectra", "the chunk spectra in the served coordinates (signal.chunk_spectra)", false},
              {"operator", "the tile operator (signal.tile_operator)", false},
              {"frequency", "the atoms: Gaussians on the spectra's frequency line", false}};
  n.outputs = {{"cells", "per atom, per cell of a level of the time line (whole cells only): Σ Re and Σ Im of z_i conj(z_j) per pair i < j, then Σ |z_i|² per tile", false}};
  n.parameters = {{"frequency_normalisation", Json(), "sum_of_squared_gains | none (as signal.tile_flow's; no default)", "-", {"sum_of_squared_gains", "none"}, false},
                  {"support_floor", Json(), "with sum_of_squared_gains (no default)", "-", {}, true},
                  {"cells", Json(), "segments (the reference's: non-overlapping segments of segment_seconds from the line's first sample, the same for every atom) | atom_width (each atom's own cells, cell_width time standard deviations snapped to a power of two of samples) (no default)", "-", {"segments", "atom_width"}, false},
                  {"segment_seconds", Json(), "cells segments: a segment's length (nxr's flow through the fibres: 4 s)", "s", {}, true},
                  {"cell_width", Json(), "cells atom_width: a cell's width in the atom's time standard deviations (the sensor flow's: 2)", "-", {}, true}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64,
                             "once a run, chunk by chunk of the stored spectra: the tile operator composed on the atom's bins, one inverse transform per row, summed over the cells (a cell cut by a chunk's edge is added up across the two chunks)");
  n.matches = "nxr flownet's per-segment C = w wᴴ traced over components (segments of --seg seconds, S = N // L, from the run's first sample); compute nxr::signal::timeCellPairFlow, takeWholeCells";
  n.description = "the tiles' cross-spectrum of each atom per cell of a level of the time line: the sums the reference's sign-flip test and its mean edge flow read";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"spectra", "operator", "frequency"}) if (!in.count(k)) ng::refuse(std::string(kCells) + ": " + k + " is not given");
    text_param(p, "frequency_normalisation", kCells);
    const std::string cells = text_param(p, "cells", kCells);
    if (cells == "segments") { if (!(number_param(p, "segment_seconds", kCells) > 0)) ng::refuse(std::string(kCells) + ": segment_seconds > 0"); }
    else if (cells == "atom_width") { if (!(number_param(p, "cell_width", kCells) > 0)) ng::refuse(std::string(kCells) + ": cell_width > 0"); }
    else ng::refuse(std::string(kCells) + ": cells is segments or atom_width");
    ng::Planned out; out.out["cells"] = statistic_port(); return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(prepare_flow(c, p)); };
  n.run = [](const std::any& prepared, const ng::Arrays& in) {
    const FlowPrepared& pr = std::any_cast<const FlowPrepared&>(prepared);
    const Spectra s = spectra_of(handed(in, "spectra", "connectome flow"), kCells);
    check_operator_against_spectra(pr.op, s, kCells);
    if (static_cast<std::int64_t>(s.owns.size()) != s.chunks || static_cast<std::int64_t>(s.firsts.size()) != s.chunks) ng::refuse(std::string(kCells) + ": the spectra do not record each chunk's samples");
    const Atoms atoms = atoms_of(s, pr.frequency, pr.p, kCells);
    const Eigen::MatrixXd M = matrix_of(pr.op);
    const std::int64_t T = M.rows() / 3, P = T * (T - 1) / 2, A = atoms.gains.rows();
    const bool segments = pr.p["cells"].get<std::string>() == "segments";
    const Eigen::MatrixXd tileReadout = nxr::signal::componentMajor(M, 3);                    // the tile operator, component-major
    const std::vector<std::int32_t> tilePairs = tile_pairs(T);
    ng::Array out;
    out.cols = 2 * P + T;
    Json groups = Json::array();
    std::int64_t row = 0;
    for (std::int64_t a = 0; a < A; ++a) {
      std::int64_t cellSamples = 0;
      if (segments) cellSamples = static_cast<std::int64_t>(std::llround(pr.p["segment_seconds"].get<double>() * s.sfreq));   // the reference's int(seg x fs)
      else {
        const double sd = atoms.standardDeviations[static_cast<std::size_t>(a)];
        if (!std::isfinite(sd)) ng::refuse(std::string(kCells) + ": atom-width cells need Gaussian atoms on a linear chart");
        cellSamples = std::int64_t(1) << static_cast<int>(std::lround(std::log2(pr.p["cell_width"].get<double>() / (2.0 * M_PI * sd) * s.sfreq)));
      }
      // whole cells only (the reference's S = N // L): a cell cut by a chunk's edge completed from the next chunk
      std::int64_t cells = 0, firstCell = -1;
      nxr::signal::TimeCellPairFlow carried;
      for (std::int64_t t = 0; t < s.chunks; ++t) {
        Eigen::MatrixXcd X(s.rows, s.bins);
        for (std::int64_t k = 0; k < s.bins; ++k) for (std::int64_t r = 0; r < s.rows; ++r) X(r, k) = atoms.gains(a, k) * s.at(t, k, r);
        const nxr::signal::TimeCellPairFlow part = nxr::signal::timeCellPairFlow(X, s.first, s.transform, s.overlap, s.owns[static_cast<std::size_t>(t)], s.firsts[static_cast<std::size_t>(t)],
                                                                                 cellSamples, tileReadout, 3, tilePairs, nullptr);
        const nxr::signal::TimeCellPairFlow f = nxr::signal::takeWholeCells(carried, part, cellSamples);
        if (f.power.cols() > 0 && firstCell < 0) firstCell = f.firstCell;
        for (Eigen::Index j = 0; j < f.power.cols(); ++j) {
          for (std::int64_t q = 0; q < P; ++q) out.data.push_back(f.pairReal(q, j));
          for (std::int64_t q = 0; q < P; ++q) out.data.push_back(-f.pairImaginary(q, j));      // Σ z_i conj(z_j): the conjugate of Σ conj(z_i) z_j
          for (std::int64_t i = 0; i < T; ++i) out.data.push_back(f.power(i, j));
        }
        cells += f.power.cols();
      }
      groups.push_back(Json{{"atom", a}, {"band", 0}, {"first_row", row}, {"cells", cells}, {"first_cell", firstCell}, {"cell_samples", cellSamples}, {"cell_seconds", static_cast<double>(cellSamples) / s.sfreq},
                            {"frequency_hz", atoms.centres[static_cast<std::size_t>(a)]}});
      row += cells;
    }
    out.rows = row;
    Json centres = Json::array();
    for (const double c : atoms.centres) centres.push_back(c);
    const Json producer = Json{{"tiles", T}, {"pairs", P}, {"atoms", A}, {"frequencies_hz", centres}, {"codes", pr.op.meta.value("codes", Json::array())}, {"cells", pr.p["cells"]},
                               {"segment_seconds", pr.p.value("segment_seconds", Json())}, {"cell_width", pr.p.value("cell_width", Json())}, {"kernel", pr.op.meta.value("kernel", std::string())}, {"groups", groups}};
    out.meta["shape"] = Json::array({row, 2 * P + T}); out.meta["producer"] = producer;
    out.meta["describe"] = Json{{"method", "tile_time_cells"}, {"measure", "time_cell_cross_spectrum"}, {"shape", Json::array({row, 2 * P + T})}, {"axes", Json::array({indexed_axis("cell", row), indexed_axis("column", 2 * P + T)})},
                                {"measures", Json::array({Json{{"measure", "cross_spectrum_real"}, {"columns", Json::array({0, P})}}, Json{{"measure", "cross_spectrum_imaginary"}, {"columns", Json::array({P, 2 * P})}},
                                                          Json{{"measure", "power"}, {"columns", Json::array({2 * P, 2 * P + T})}}})},
                                {"producer", producer},
                                {"attributes", Json{{"groups", groups}, {"rows", "the whole cells of each atom, in time order, the groups' order"},
                                                    {"columns", "per pair i < j: Σ_c Σ_t Re(z_i conj z_j), then Σ Im (> 0 when tile j lags tile i), then each tile's Σ |z|²"}}}};
    return ng::Arrays{{"cells", std::move(out)}};
  };
  n.record = [](const std::any& prepared) { const Json& p = std::any_cast<const FlowPrepared&>(prepared).p; return Json{{"cells", p.value("cells", Json())}, {"segment_seconds", p.value("segment_seconds", Json())}, {"cell_width", p.value("cell_width", Json())}}; };
  return n;
}

// the cells of each atom (signal.tile_time_cells), read fresh or from the store
struct CellsRead { std::int64_t T = 0, P = 0; Json groups, codes, centres; };
CellsRead cells_read(const ng::Array& cells, const char* who) {
  const Json pr = producer_of(cells);
  if (!pr.contains("groups") || !pr.contains("tiles")) ng::refuse(std::string(who) + ": cells is not signal.tile_time_cells' cells");
  CellsRead r;
  r.T = pr["tiles"]; r.P = r.T * (r.T - 1) / 2; r.groups = pr["groups"]; r.codes = pr.value("codes", Json::array()); r.centres = pr.value("frequencies_hz", Json::array());
  if (cells.cols != 2 * r.P + r.T) ng::refuse(std::string(who) + ": the cells' columns do not fit their tiles");
  return r;
}

// ── stats.network_sign_flip ──────────────────────────────────────────────────────────────────────────────────────
const char* const kFlip = "stats.network_sign_flip";
ng::NodeSpec network_sign_flip() {
  ng::NodeSpec n;
  n.name = kFlip;
  n.kind = ng::Kind::array;
  n.inputs = {{"cells", "each atom's cells (signal.tile_time_cells)", false}, {"pairs", "the tile pairs (meeg.tile_pairs): their streamline counts and distances", false}};
  n.outputs = {{"consistency", "per atom and pair: the mean edge flow over the cells, its p and family-wise p, the control flag, the phase of the cells' summed cross-spectrum", false}};
  n.parameters = {{"patterns", Json(), "the random sign patterns over the cells (no default; nxr: 2000)", "-", {}, false},
                  {"seed", Json(), "the seed of numpy's default_rng, created once a run and drawn in the reference's order: per atom the flip, then the control's draws (no default; nxr: 0)", "-", {}, false},
                  {"min_streamlines", Json(), "a pair is joined when at least this many streamlines join it (no default; nxr: 10)", "streamlines", {}, false},
                  {"distance_bin_edges", Json(), "the edges of the distance bins the control is matched in, in the pairs' distance unit (no default; nxr: 0 30 45 60 80 100 130 170 250 mm)", "-", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "a statistic of the cells: one numpy generator, drawn in the reference's order");
  n.matches = "nxr flownet's flip (sign flip over segments, |null| >= |obs|, family-wise maximum on the sums normalised by Σ|F|) and its distance-matched control (rng.choice per np.digitize bin), numpy default_rng bit for bit; compute nxr::stats::networkSignFlip, distanceMatchedControl";
  n.description = "the reference's nulls of the flow through the network: the sign-flip (time reversal) test of each pair's edge flow over the cells with its family-wise maximum, and the distance-matched not-joined pairs as the control";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"cells", "pairs"}) if (!in.count(k)) ng::refuse(std::string(kFlip) + ": " + k + " is not given");
    number_param(p, "patterns", kFlip); number_param(p, "seed", kFlip); number_param(p, "min_streamlines", kFlip);
    if (!p.contains("distance_bin_edges") || !p["distance_bin_edges"].is_array()) ng::refuse(std::string(kFlip) + ": distance_bin_edges has no default: name them");
    ng::Planned out; out.out["consistency"] = measurement_port(); return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(std::make_tuple(handed(c, "cells", kFlip), handed(c, "pairs", kFlip), p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [cells, pairs, p] = std::any_cast<const std::tuple<ng::Array, ng::Array, Json>&>(prepared);
    const CellsRead cr = cells_read(cells, kFlip);
    const PairsRead pr = pairs_read(pairs, p["min_streamlines"].get<double>(), kFlip);
    if (pr.codes != cr.codes) ng::refuse(std::string(kFlip) + ": the cells and the pairs are not on the same tiles");
    const std::int64_t T = cr.T, P = cr.P;
    std::vector<std::uint8_t> joined(static_cast<std::size_t>(P));
    Eigen::VectorXd distance(P);
    for (std::int64_t q = 0; q < P; ++q) { joined[static_cast<std::size_t>(q)] = pr.weight(q) > 0; distance(q) = cell(pairs, q, 4); }
    const std::vector<std::int64_t> bins = nxr::stats::digitize(distance, p["distance_bin_edges"].get<std::vector<double>>());
    nxr::stats::NumpyGenerator generator(static_cast<std::uint64_t>(p["seed"].get<std::int64_t>()));
    const std::int64_t patterns = p["patterns"].get<std::int64_t>();
    ng::Array out;
    out.rows = static_cast<std::int64_t>(cr.groups.size()); out.cols = 5 * P;
    Json rows = Json::array();
    for (const auto& g : cr.groups) {
      const std::int64_t first = g["first_row"], S = g["cells"];
      if (S < 1) ng::refuse(std::string(kFlip) + ": an atom has no whole cell");
      Eigen::MatrixXd flows(S, P);
      Eigen::VectorXd re = Eigen::VectorXd::Zero(P), im = Eigen::VectorXd::Zero(P);
      for (std::int64_t s = 0; s < S; ++s)
        for (std::int64_t i = 0; i < T; ++i)
          for (std::int64_t j = i + 1; j < T; ++j) {
            const std::int64_t q = nxr::meeg::pairIndex(i, j, T);
            const double d = std::sqrt(cell(cells, first + s, 2 * P + i) * cell(cells, first + s, 2 * P + j));
            const double imag = cell(cells, first + s, P + q);
            flows(s, q) = static_cast<double>(static_cast<float>(imag / d));                    // the reference's C.imag / outer(d, d), kept in float32 as its FS array (no guard: as the reference)
            re(q) += cell(cells, first + s, q); im(q) += imag;
          }
      const nxr::stats::NetworkSignFlip f = nxr::stats::networkSignFlip(flows, patterns, generator);
      const std::vector<std::uint8_t> control = nxr::stats::distanceMatchedControl(joined, bins, generator);
      for (std::int64_t q = 0; q < P; ++q) out.data.push_back(f.sum(q) / static_cast<double>(S));
      for (std::int64_t q = 0; q < P; ++q) out.data.push_back(f.p(q));
      for (std::int64_t q = 0; q < P; ++q) out.data.push_back(f.familywiseP(q));
      for (std::int64_t q = 0; q < P; ++q) out.data.push_back(control[static_cast<std::size_t>(q)]);
      for (std::int64_t q = 0; q < P; ++q) out.data.push_back(std::atan2(im(q), re(q)));
      rows.push_back(Json{{"atom", g["atom"]}, {"frequency_hz", g.value("frequency_hz", Json())}, {"cells", S}, {"cell_seconds", g.value("cell_seconds", Json())}});
    }
    const Json producer = Json{{"tiles", T}, {"pairs", P}, {"codes", cr.codes}, {"frequencies_hz", cr.centres}, {"rows", rows}, {"patterns", patterns}, {"seed", p["seed"]},
                               {"min_streamlines", p["min_streamlines"]}, {"distance_bin_edges", p["distance_bin_edges"]},
                               {"generator", "numpy default_rng (SeedSequence, PCG64), one for the run: per atom random((patterns, cells)) < 0.5 -> -1, then choice(replace=False) per occupied distance bin ascending"},
                               {"derived_at_read", Json{{"lag_s", "-phase / (2 pi f)"}, {"speed", "median length / |lag| on joined pairs with family-wise p < 0.05 (an upper bound)"},
                                                        {"shares", "p < 0.05 among the joined pairs, the control pairs, the joined pairs across the hemispheres"}}},
                               {"warnings", warnings()}};
    out.meta["shape"] = Json::array({out.rows, 5 * P}); out.meta["producer"] = producer;
    out.meta["describe"] = Json{{"method", "network_sign_flip"}, {"measure", "edge_flow_consistency"}, {"shape", Json::array({out.rows, 5 * P})},
                                {"axes", Json::array({indexed_axis("atom", out.rows, rows), indexed_axis("column", 5 * P)})},
                                {"measures", Json::array({Json{{"measure", "mean_edge_flow"}, {"columns", Json::array({0, P})}}, Json{{"measure", "p"}, {"columns", Json::array({P, 2 * P})}},
                                                          Json{{"measure", "familywise_p"}, {"columns", Json::array({2 * P, 3 * P})}}, Json{{"measure", "control"}, {"columns", Json::array({3 * P, 4 * P})}},
                                                          Json{{"measure", "phase"}, {"columns", Json::array({4 * P, 5 * P})}}})},
                                {"producer", producer},
                                {"description", "per atom and pair i < j: the mean over the cells of Im C_ij / sqrt(C_ii C_jj) (> 0: tile j lags tile i), its sign-flip p and family-wise p, 1 for a distance-matched control pair, arg of Σ_cells C_ij"}};
    declare_flow_measures(out.meta["describe"]["measures"]);   // the flow measures it writes, declared (flow_measures.h)
    return ng::Arrays{{"consistency", measurement_of_blocks(out, 0, P, {"mean_edge_flow", "p", "familywise_p", "control", "phase"}, "the tile pairs i < j")}};
  };
  n.record = [](const std::any& prepared) { const Json& p = std::get<2>(std::any_cast<const std::tuple<ng::Array, ng::Array, Json>&>(prepared)); return Json{{"patterns", p["patterns"]}, {"seed", p["seed"]}}; };
  return n;
}

// ── stats.run_replication ────────────────────────────────────────────────────────────────────────────────────────
const char* const kReplication = "stats.run_replication";
ng::NodeSpec run_replication() {
  ng::NodeSpec n;
  n.name = kReplication;
  n.kind = ng::Kind::array;
  n.inputs = {{"first", "one run's edge flows (stats.network_sign_flip)", false}, {"second", "another run's, on the same tiles and atoms", false}, {"pairs", "the tile pairs (meeg.tile_pairs)", false}};
  n.outputs = {{"replication", "per atom: the correlation of the two runs' mean edge flows over the joined pairs, and over the not-joined pairs", false}};
  n.parameters = {{"min_streamlines", Json(), "a pair is joined when at least this many streamlines join it (no default; nxr: 10)", "streamlines", {}, false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "once: Pearson correlations over the pairs");
  n.matches = "nxr's replication of the flow through the fibres (views/netfig: np.corrcoef of run 1's and run 2's F_k over the joined and the not-joined pairs)";
  n.description = "run-to-run replication of the flow through the network: per atom the correlation of two runs' edge flows over the joined pairs and over the not-joined pairs";
  n.plan = [](const ng::Types& in, const Json& p) {
    for (const char* k : {"first", "second", "pairs"}) if (!in.count(k)) ng::refuse(std::string(kReplication) + ": " + k + " is not given");
    number_param(p, "min_streamlines", kReplication);
    ng::Planned out; out.out["replication"] = statistic_port(); return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(std::make_tuple(handed(c, "first", kReplication), handed(c, "second", kReplication), handed(c, "pairs", kReplication), p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [first, second, pairs, p] = std::any_cast<const std::tuple<ng::Array, ng::Array, ng::Array, Json>&>(prepared);
    const PairsRead pr = pairs_read(pairs, p["min_streamlines"].get<double>(), kReplication);
    // each run's mean edge flow: stats.network_sign_flip's measurement in a run, or the stored measurement mean_edge_flow (read by its measure) of each run
    const std::pair<std::int64_t, std::int64_t> firstShape = atoms_rows(first), secondShape = atoms_rows(second);
    const std::int64_t A = firstShape.first, firstRows = firstShape.second, secondAtoms = secondShape.first, secondRows = secondShape.second;
    if (firstRows != pr.pairs || secondRows != pr.pairs) ng::refuse(std::string(kReplication) + ": the runs' flows are not on the pairs' tiles");
    if (secondAtoms != A || (first.meta.contains("id") && second.meta.contains("id") && first.meta["id"] != second.meta["id"]))
      ng::refuse(std::string(kReplication) + ": the two runs were read with different atoms (measured over different selections)");
    const std::int64_t P = pr.pairs;
    auto pearson = [&](std::int64_t atom, bool wantJoined) {
      std::vector<double> x, y;
      for (std::int64_t q = 0; q < P; ++q) if ((pr.weight(q) > 0) == wantJoined) { x.push_back(measured(first, q, atom, 0)); y.push_back(measured(second, q, atom, 0)); }
      const Eigen::Map<const Eigen::VectorXd> X(x.data(), static_cast<Eigen::Index>(x.size())), Y(y.data(), static_cast<Eigen::Index>(y.size()));
      const Eigen::VectorXd dx = X.array() - X.mean(), dy = Y.array() - Y.mean();
      return dx.dot(dy) / std::sqrt(dx.squaredNorm() * dy.squaredNorm());
    };
    ng::Array out;
    out.rows = A; out.cols = 2;
    for (std::int64_t atom = 0; atom < A; ++atom) { out.data.push_back(pearson(atom, true)); out.data.push_back(pearson(atom, false)); }
    const Json producer = Json{{"min_streamlines", p["min_streamlines"]}, {"frequencies_hz", first.meta.contains("producer") ? producer_of(first).value("frequencies_hz", Json()) : Json()}, {"codes", pr.codes}};
    out.meta["shape"] = Json::array({A, 2}); out.meta["producer"] = producer;
    out.meta["describe"] = Json{{"method", "run_replication"}, {"measure", "replication"}, {"shape", Json::array({A, 2})},
                                {"axes", Json::array({indexed_axis("atom", A, producer.value("frequencies_hz", Json())), indexed_axis("column", 2, Json::array({"joined_correlation", "not_joined_correlation"}))})},
                                {"producer", producer}, {"description", "per atom: Pearson's r of the two runs' mean edge flows over the joined pairs, then over the not-joined pairs"}};
    return ng::Arrays{{"replication", std::move(out)}};
  };
  return n;
}

// ── meeg.network_flow ──────────────────────────────────────────────────────────────────────────────────────
// the graph's STORED Hodge-Dirac modes (meeg.graph_hodge_dirac, read back as a plain operator [(V + E) x K] and its eigenvalues [K]) as compute's GraphModes
nxr::meeg::GraphModes stored_graph_modes(const ng::Array& vectors, const ng::Array& eigenvalues, std::int64_t V, std::int64_t E, const char* who) {
  const std::int64_t K = vectors.cols;
  if (vectors.rows != V + E) ng::refuse(std::string(who) + ": the modes are not the graph's ([(V + E) x K] with V " + std::to_string(V) + " and E " + std::to_string(E) + "; they have " + std::to_string(vectors.rows) + " rows): solved on another graph");
  if (static_cast<std::int64_t>(eigenvalues.size()) != K) ng::refuse(std::string(who) + ": the eigenvalues are not one per mode");
  nxr::meeg::GraphModes m;
  m.nodes = V; m.edges = E;
  m.forms = matrix_of(vectors);
  m.mu.resize(K);
  const std::vector<double> lambda = eigenvalues.values();
  for (std::int64_t k = 0; k < K; ++k) {
    const double l = lambda[static_cast<std::size_t>(k)];
    m.mu(k) = l > 0 ? std::sqrt(l) : 0.0;
    m.kind.push_back(l > 0 ? 0 : 1);
    m.cycles += l > 0 ? 0 : 1;
  }
  return m;
}
const char* const kNetwork = "meeg.network_flow";
ng::NodeSpec network_flow() {
  ng::NodeSpec n;
  n.name = kNetwork;
  n.kind = ng::Kind::array;
  n.inputs = {{"pairs", "the tile pairs (meeg.tile_pairs): the codes the flow is on", false},
              {"network", "the network: a graph manifold (meeg.tile_pairs' graph): its edges are the joined pairs", false},
              {"conductance", "the graph's edge weights (star1), the field its modes were solved with", false},
              {"modes", "the graph's STORED Hodge-Dirac modes: the eigenvectors operator (`nxr-neuro run eigenmodes --graph`)", false},
              {"eigenvalues", "their eigenvalues (lambda; 0 on a cycle mode)", false},
              {"flow", "the edge flow per atom: stats.network_sign_flip's consistency (its first measure mean_edge_flow: the mean over the cells, the reference's), a measurement over the atoms", false}};
  n.outputs = {{"network_tiles", "a MEASUREMENT over the atoms: per tile its divergence and graph potential", false},
               {"network_modes", "a MEASUREMENT over the atoms: the flow's coefficient on each of the graph's modes", false},
               {"network_totals", "a MEASUREMENT over the atoms, one row: the gradient part's, the cycle part's and the flow's star1 energies", false}};
  n.traits = ng::node_traits(ng::TimeBehaviour::pointwise, ng::Precision::float64, ng::OutputPrecision::float64, "once: inner products with the stored modes, from the stored edge flows (no solve)");
  n.matches = "nxr flownet's div = Beᵀ Fj, phi = lstsq(sqrt(w) Be, sqrt(w) Fj), gshare; here the flow's coefficients on the graph's stored Hodge-Dirac modes (the surfaces' flowSplit on a graph), the weights the graph's conductance; compute nxr::meeg::networkFlow";
  n.description = "the network's view of the edge flow per atom: the net flow into each tile, the flow's coefficient on every mode of the graph (gradient and cycle parts), the graph potential and the parts' energies";
  n.plan = [](const ng::Types& in, const Json&) {
    for (const char* k : {"pairs", "network", "conductance", "modes", "eigenvalues", "flow"}) if (!in.count(k)) ng::refuse(std::string(kNetwork) + ": " + k + " is not given");
    ng::Planned out; for (const char* o : {"network_tiles", "network_modes", "network_totals"}) out.out[o] = measurement_port(); return out;
  };
  n.prepare = [](const ng::Types&, const ng::Arrays& c, const Json& p, ng::Shared&) { return std::any(std::make_pair(c, p)); };
  n.run = [](const std::any& prepared, const ng::Arrays&) {
    const auto& [c, p] = std::any_cast<const std::pair<ng::Arrays, Json>&>(prepared);
    const ng::Array& pairs = handed(c, "pairs", kNetwork);
    const ng::Array& graph = handed(c, "network", kNetwork);
    const ng::Array& flow = handed(c, "flow", kNetwork);
    const Json pp = producer_of(pairs), gp = graph.meta.value("producer", Json::object()), fp = producer_of(flow);
    const std::vector<std::int32_t> edges = graph_edges(graph, kNetwork);
    const std::int64_t T = graph.rows, E = static_cast<std::int64_t>(edges.size() / 2);
    if (pp.value("codes", Json()) != gp.value("codes", Json()) || fp.value("codes", Json()) != gp.value("codes", Json())) ng::refuse(std::string(kNetwork) + ": the pairs, the network and the flow are not on the same tiles");
    if (pp.value("tiles", std::int64_t(-1)) != T) ng::refuse(std::string(kNetwork) + ": the pairs are not on the network's tiles");
    const Eigen::VectorXd conductance = cell_values(handed(c, "conductance", kNetwork), E, "conductance", kNetwork);
    const nxr::meeg::GraphModes modes = stored_graph_modes(handed(c, "modes", kNetwork), handed(c, "eigenvalues", kNetwork), T, E, kNetwork);
    const std::int64_t K = modes.mu.size(), P = T * (T - 1) / 2;
    const std::pair<std::int64_t, std::int64_t> flowShape = atoms_rows(flow);
    const std::int64_t A = flowShape.first, flowRows = flowShape.second;                                                // stats.network_sign_flip's measurement: the mean edge flow per pair and atom
    if (flowRows != P) ng::refuse(std::string(kNetwork) + ": the flow is not on the network's tile pairs");
    if (first_measure(flow) != "mean_edge_flow") ng::refuse(std::string(kNetwork) + ": flow is not the mean edge flow (stats.network_sign_flip's consistency): its first measure is '" + first_measure(flow) + "'");
    ng::Array out;
    out.rows = A; out.cols = 2 * T + K + 3;
    for (std::int64_t a = 0; a < A; ++a) {
      Eigen::VectorXd edgeFlow(E);                                                    // the pairs' flow read on the graph's edges (tail < tip: the pair's i -> j)
      for (std::int64_t e = 0; e < E; ++e) edgeFlow(e) = measured(flow, nxr::meeg::pairIndex(edges[static_cast<std::size_t>(2 * e)], edges[static_cast<std::size_t>(2 * e + 1)], T), a, 0);
      const nxr::meeg::NetworkFlow nf = nxr::meeg::networkFlow(edgeFlow, edges, conductance, modes);
      for (std::int64_t i = 0; i < T; ++i) out.data.push_back(nf.divergence(i));
      for (std::int64_t i = 0; i < T; ++i) out.data.push_back(nf.potential(i));
      for (std::int64_t k = 0; k < K; ++k) out.data.push_back(nf.coefficients(k));
      out.data.push_back(nf.weightedGradientSquares); out.data.push_back(nf.weightedCycleSquares); out.data.push_back(nf.weightedFlowSquares);
    }
    const Json producer = Json{{"tiles", T}, {"edges", E}, {"modes", K}, {"cycles", modes.cycles}, {"atoms", A}, {"codes", gp.value("codes", Json())}, {"frequencies_hz", fp.value("frequencies_hz", Json::array())},
                               {"network", graph.meta.value("name", std::string())}, {"modes_read", c.at("modes").meta.value("name", std::string())},
                               {"edge_flow", "the mean over the cells of Im C_ij / sqrt(C_ii C_jj) (stats.network_sign_flip: the reference's), read on the graph's edges"},
                               {"weights", "the graph's metric: its conductance (streamline count / median length) on the edges, its tiles' areas on the vertices (the stored modes')"},
                               {"derived_at_read", Json{{"gradient_share", "weighted_gradient_squares / weighted_flow_squares"}, {"cycle_share", "weighted_cycle_squares / weighted_flow_squares"}, {"energy_per_mode", "flow_coefficient^2 x |phi_k|^2 (1/2 exact, 1 cycle)"}}},
                               {"warnings", warnings()}};
    out.meta["shape"] = Json::array({A, 2 * T + K + 3}); out.meta["producer"] = producer;
    out.meta["describe"] = Json{{"method", "network_flow"}, {"measure", "network_flow"}, {"shape", Json::array({A, 2 * T + K + 3})},
                                {"axes", Json::array({indexed_axis("atom", A, fp.value("frequencies_hz", Json())), indexed_axis("column", 2 * T + K + 3)})},
                                {"measures", Json::array({Json{{"measure", "divergence"}, {"columns", Json::array({0, T})}}, Json{{"measure", "potential"}, {"columns", Json::array({T, 2 * T})}},
                                                          Json{{"measure", "flow_coefficient"}, {"columns", Json::array({2 * T, 2 * T + K})}}, Json{{"measure", "weighted_gradient_squares"}, {"columns", Json::array({2 * T + K, 2 * T + K + 1})}},
                                                          Json{{"measure", "weighted_cycle_squares"}, {"columns", Json::array({2 * T + K + 1, 2 * T + K + 2})}},
                                                          Json{{"measure", "weighted_flow_squares"}, {"columns", Json::array({2 * T + K + 2, 2 * T + K + 3})}}})},
                                {"producer", producer},
                                {"description", "per atom on the network's edges: divergence Bᵀ F per tile, the graph potential (the exact part's: Σ c_k psi_k / mu_k), the flow's coefficient on every stored mode, the gradient part's, the cycle part's and the flow's star1 energies"}};
    declare_flow_measures(out.meta["describe"]["measures"]);   // the flow measures it writes, declared (flow_measures.h)
    return ng::Arrays{{"network_tiles", measurement_of_blocks(out, 0, T, {"divergence", "potential"}, "the tiles")},
                      {"network_modes", measurement_of_blocks(out, 2 * T, K, {"flow_coefficient"}, "the graph's Hodge-Dirac modes (its eigenvalue line)")},
                      {"network_totals", measurement_of_blocks(out, 2 * T + K, 1, {"weighted_gradient_squares", "weighted_cycle_squares", "weighted_flow_squares"}, "one row: the atom's totals")}};
  };
  return n;
}

}  // namespace

void register_connectome_flow_nodes() {
  ng::Registry& r = ng::registry();
  for (ng::NodeSpec n : {fiber_lengths(), tile_pairs(), graph_hodge_dirac(), tile_operator(), tile_flow(), tile_time_cells(), network_sign_flip(), run_replication(), network_flow()})
    if (!r.find(n.name)) r.add(std::move(n));
}

}  // namespace nxr::pipelines::graph
