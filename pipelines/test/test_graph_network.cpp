// The fibre network as a GRAPH MANIFOLD and its Hodge-Dirac modes (spec docs/specs/2026-11-26-graph-manifold-and-time-cells.md, G1) on a synthetic two-hemisphere brain (never a real
// dataset): `tile_network` stores the network as a manifold of type graph (the tiles at a level stated as a quantity its vertices, the joined pairs its stored rank-1 cells) with its metric
// as fields on its cells (the tiles' areas sum to the surface's; conductance = streamline count / median length); `eigenmodes --graph` stores its modes in the surfaces' four-class form
// (the line, the eigenvalues, the kinds, ONE eigenvectors operator to the graph's cells 0+1, orthonormal under the stars) and a rerun is a cache hit; a definition that names the modes
// (`connectome_flow`, `made_by`) makes them first when they are not stored, and never again once they are.
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>

#include "brain_helpers.h"
#include "check.h"
#include "make_grids.h"
#include "nxr/bst/frequency.h"
#include "nxr/io/array_reader.h"
#include "nxr/io/fibers.h"
#include "nxr/io/spatial_bands.h"
#include "nxr/io/task/task.h"
#include "run_definition.h"

using namespace nxr::io;
namespace fs = std::filesystem;
namespace tk = nxr::io::task;

namespace {
template <class F> std::string refusal(F f) { try { f(); } catch (const std::exception& e) { return e.what(); } return ""; }
std::vector<double> doubles(SubjectWriter& sub, const std::string& path) {     // a stored array in double, whatever its stored precision
  ArrayReader r = ArrayReader::open(sub.store(), path);
  const std::vector<std::uint8_t> raw = r.read();
  if (r.dtype() == DType::Float64) { std::vector<double> v(raw.size() / sizeof(double)); std::memcpy(v.data(), raw.data(), raw.size()); return v; }
  std::vector<float> f(raw.size() / sizeof(float));
  std::memcpy(f.data(), raw.data(), raw.size());
  return std::vector<double>(f.begin(), f.end());
}
}  // namespace

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_graph_network_" + std::to_string(::getpid()));
  fs::remove_all(root);
  struct Cleanup { fs::path p; ~Cleanup() { fs::remove_all(p); } } cleanup{root};
  const Brain b = brain(3, false);
  DatasetWriter ds = create_dataset(root, "demo");
  {
    SubjectWriter def = SubjectWriter::create(ds, "default_template", [] { SubjectOptions o; o.kind = "template"; return o; }());
    def.compose([&] { nxr::bst::ensure_frequency(def); ensure_eigenvalue(def); });
  }
  {
    SubjectWriter sub = SubjectWriter::create(ds, "S1");
    sub.compose([&] { SurfaceOptions so; so.name = "cortex"; so.sphere = b.Sph; add_surface(sub, so, b.P, b.F); });
    // the fibres: straight three-point streamlines between random vertices (SCS metres), ends exactly on vertices
    std::mt19937 rng(3);
    std::uniform_int_distribution<int> pick(0, b.n - 1);
    nxr::io::fibers::Bundle<double> bundle;
    bundle.name = "bundle"; bundle.points_per_fiber = 3; bundle.source = "synthetic";
    for (int f = 0; f < 600; ++f) {
      const int a = pick(rng), c = pick(rng);
      for (int t = 0; t < 3; ++t) for (int k = 0; k < 3; ++k) bundle.points.push_back(b.P[static_cast<std::size_t>(3 * a + k)] + 0.5 * t * (b.P[static_cast<std::size_t>(3 * c + k)] - b.P[static_cast<std::size_t>(3 * a + k)]));
    }
    bundle.n_fibers = 600;
    sub.compose([&] { nxr::io::fibers::store_bundle(sub, bundle); });
  }
  SubjectWriter sub = SubjectWriter::open(ds, "S1");
  CHECK(!nxrtest::make_tiles(ds, sub, "parts 4", std::string("cortex")).empty());     // levels 0..2: four tiles a hemisphere
  const auto ts = tk::Subject::of(sub);

  // ── tile_network: the network as a graph manifold, its metric as fields on its cells ──
  const Json net = Json{{"fibers", "bundle"}, {"end_surface", "cortex"}, {"tiles", "cortex tiles"}, {"scale", "parts 4"}, {"min_streamlines", 3}, {"name", "net"}};
  CHECK(refusal([&] { Json bad = net; bad["scale"] = "parts 3"; nxrtest::run_definition(ds, sub, "tile_network", bad); }).find("power") != std::string::npos);   // parts are powers of two
  CHECK(!nxrtest::run_definition(ds, sub, "tile_network", net).empty());
  const auto graph = ts.find({.cls = tk::Cls::manifold, .name = "net", .type = "graph"});
  CHECK(graph.has_value());
  if (!graph) return nxrtest::finish("test_graph_network");
  const std::int64_t V = graph->n_vertices.value_or(0), E = graph->fact("n_edges").is_number() ? graph->fact("n_edges").get<std::int64_t>() : 0;
  const Json made = graph->producer;
  const std::int64_t comps = made.contains("components") ? made["components"].get<std::int64_t>() : made.value("producer", Json::object()).value("components", std::int64_t(1));
  std::printf("  the network: %lld tiles, %lld edges, %lld component(s)\n", static_cast<long long>(V), static_cast<long long>(E), static_cast<long long>(comps));
  CHECK(V == 8 && E > V);                                                            // eight tiles; enough streamlines for cycles (the edges, stored rank-1 cells, are read back below by the solve)
  const auto area = ts.find({.cls = tk::Cls::field, .name = "net_area"});
  const auto conductance = ts.find({.cls = tk::Cls::field, .name = "net_conductance"});
  const auto streamlines = ts.find({.cls = tk::Cls::field, .name = "net_streamlines"});
  const auto medians = ts.find({.cls = tk::Cls::field, .name = "net_median_length"});
  CHECK(area && conductance && streamlines && medians);
  if (area && conductance && streamlines && medians) {
    const std::vector<double> A = doubles(sub, *area->path), C = doubles(sub, *conductance->path), N = doubles(sub, *streamlines->path), L = doubles(sub, *medians->path);
    CHECK(static_cast<std::int64_t>(A.size()) == V && static_cast<std::int64_t>(C.size()) == E);
    double total = 0;
    for (const double a : A) total += a;
    double surface = 0;                                                              // the surface's area, from its faces
    for (int f = 0; f < b.nF; ++f) {
      const double* p0 = &b.P[static_cast<std::size_t>(3 * b.F[static_cast<std::size_t>(3 * f)])];
      const double* p1 = &b.P[static_cast<std::size_t>(3 * b.F[static_cast<std::size_t>(3 * f + 1)])];
      const double* p2 = &b.P[static_cast<std::size_t>(3 * b.F[static_cast<std::size_t>(3 * f + 2)])];
      const double u[3] = {p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]}, w[3] = {p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]};
      surface += 0.5 * std::sqrt(std::pow(u[1] * w[2] - u[2] * w[1], 2) + std::pow(u[2] * w[0] - u[0] * w[2], 2) + std::pow(u[0] * w[1] - u[1] * w[0], 2));
    }
    CHECK(std::abs(total - surface) < 1e-9 * surface);                               // the tiles' areas: the whole surface
    bool conductances = true;
    for (std::int64_t e = 0; e < E; ++e) conductances = conductances && N[static_cast<std::size_t>(e)] >= 3 && std::abs(C[static_cast<std::size_t>(e)] - N[static_cast<std::size_t>(e)] / L[static_cast<std::size_t>(e)]) < 1e-9 * C[static_cast<std::size_t>(e)];
    CHECK(conductances);
  }
  CHECK(!ts.find({.cls = tk::Cls::statistic, .name = "net_eigenmodes"}).has_value());   // no Laplacian solved by the network
  CHECK(ts.find({.cls = tk::Cls::statistic, .name = "net_pairs"}).has_value());

  // ── store on miss: a definition that names the modes (connectome_flow) makes them first, as their own run, even when its other inputs are not there ──
  const std::string flowError = refusal([&] {
    nxrtest::run_definition(ds, sub, "connectome_flow", Json{{"spectra", "none_stored"}, {"package", "none"}, {"channel_map", "none"}, {"atoms", "none"}, {"kernel", "none"}, {"tiles", "cortex tiles"}, {"network", "net"}});
  });
  CHECK(!flowError.empty());                                                         // the flow itself is refused: no spectra
  const auto vectors = ts.find({.cls = tk::Cls::operator_, .name = "hodgeDirac__net_eigenvectors"});
  CHECK(vectors.has_value());
  if (!vectors) return nxrtest::finish("test_graph_network");
  const auto line = ts.find({.cls = tk::Cls::manifold, .name = "hodgeDirac__net", .type = "points"});
  CHECK(line.has_value() && ts.find({.cls = tk::Cls::field, .name = "hodgeDirac__net_eigenvalues"}).has_value() && ts.find({.cls = tk::Cls::selection, .name = "hodgeDirac__net_kinds"}).has_value());
  CHECK(vectors->to_manifold_id == graph->id && vectors->kind == std::optional<std::string>("eigenvectors"));
  // the four-class form: [(V + E) x K], K = (V - components) exact + (E - V + components) cycles = E when the network is connected; orthonormal under diag(*0, *1)
  ArrayReader vr = ArrayReader::open(sub.store(), *vectors->path);
  const std::vector<std::int64_t> shape = vr.shape();
  CHECK(shape.size() == 2 && shape[0] == V + E && shape[1] == E);
  if (shape.size() == 2 && area && conductance) {
    const std::vector<double> Phi = doubles(sub, *vectors->path), A = doubles(sub, *area->path), C = doubles(sub, *conductance->path);
    const std::int64_t K = shape[1];
    double worst = 0;
    for (std::int64_t i = 0; i < K; ++i)
      for (std::int64_t j = 0; j < K; ++j) {
        double g = 0;
        for (std::int64_t r = 0; r < V + E; ++r) g += Phi[static_cast<std::size_t>(r * K + i)] * (r < V ? A[static_cast<std::size_t>(r)] : C[static_cast<std::size_t>(r - V)]) * Phi[static_cast<std::size_t>(r * K + j)];
        worst = std::max(worst, std::abs(g - (i == j ? 1.0 : 0.0)));
      }
    CHECK(worst < 1e-8);
    CHECK(K == (V - comps) + (E - V + comps));
  }
  // a second read finds them stored: nothing is solved again
  const std::string first = vectors->id;
  CHECK(!refusal([&] { nxrtest::run_definition(ds, sub, "connectome_flow", Json{{"spectra", "none_stored"}, {"package", "none"}, {"channel_map", "none"}, {"atoms", "none"}, {"kernel", "none"}, {"tiles", "cortex tiles"}, {"network", "net"}}); }).empty());
  CHECK(ts.find({.cls = tk::Cls::operator_, .name = "hodgeDirac__net_eigenvectors"})->id == first);
  const auto again = nxrtest::run_definition(ds, sub, "eigenmodes", Json{{"graph", "net"}});
  CHECK(again.size() == 1 && again[0].value("hit", false));                         // eigenmodes --graph run again: a cache hit
  CHECK(!refusal([&] { nxrtest::run_definition(ds, sub, "eigenmodes", Json{{"graph", "net"}, {"surface", "cortex"}}); }).empty());   // one of surface or graph
  return nxrtest::finish("test_graph_network");
}
