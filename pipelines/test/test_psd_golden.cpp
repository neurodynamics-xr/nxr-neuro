// The `psd` and `rotary_psd` definitions against the REFERENCES (docs/specs/2026-11-17-definition-outputs.md, section 7): the synthetic signal of compute/test/golden/welch_golden.txt (3 channels, 2400 samples
// at 200 Hz, one bad span), run by the runner under each preset and parameter set, equals scipy.signal.welch, nxr's spectral chain and Brainstorm's bst_psd output (the goldens' own values) to float32
// storage; the rotary run equals rotary_welch_golden.txt (scipy on the complex pair). Synthetic signals only. The result does not depend on the chunk size.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>

#include "check.h"
#include "golden_io.h"
#include "nxr/bst/entities.h"
#include "nxr/io/statistic.h"
#include "nxr/io/task/task.h"
#include "psd_run.h"

using namespace nxr::io;
namespace fs = std::filesystem;
using Eigen::MatrixXd;

namespace {
const std::string kDir = NXR_TEST_GOLDEN_DIR;

// a subject with a recording `rest` of X [C x T] (float32, `chunk` samples a chunk), its bad span from the mask, and (rotary) the operator `plane` [2 x C] over every channel
void make_subject(DatasetWriter& ds, const std::string& name, const MatrixXd& X, double fs, std::int64_t chunk, const std::vector<std::uint8_t>* good, const MatrixXd* plane, const std::string& rec = "rest") {
  const std::int64_t C = X.rows(), T = X.cols();
  SubjectWriter sub = SubjectWriter::create(ds, name);
  sub.compose([&] {
    const std::string cid = uuid7();
    std::vector<float> pos(static_cast<std::size_t>(C * 3), 0.05f);
    ManifoldSpec m;
    m.id = cid; m.name = "sensors"; m.path = "sensor/sensors"; m.type = "points"; m.n_vertices = C;
    m.topology = {"none", 0, 1, std::nullopt, {}};
    GeometrySpec gs; gs.form = "stored"; gs.cs_key = "scs"; gs.positions_path = "sensor/sensors"; m.geometry = gs;
    m.populate = [&](nxr::io::Store&, const std::string&) { sub.write_array("sensor/sensors", view(pos, {C, 3}), {}, std::nullopt, "manifold", cid); };
    sub.manifold(std::move(m));
    ManifoldSpec tm;
    tm.name = "rest_time"; tm.path = "time/rest_time"; tm.type = "line"; tm.n_vertices = T;
    tm.topology = {"lattice", 1, 1, std::nullopt, {}};
    tm.cols.set("n_edges", T - 1);
    GeometrySpec tg; tg.form = "closed_form"; tg.cs_key = "clock";
    DimensionSpec dm; dm.name = "time"; dm.n_vertices = T; dm.spacing = 1.0 / fs; dm.origin = 0.0; dm.unit = "s";
    tg.dimensions.push_back(dm); tm.geometry = tg;
    const std::string tid = sub.manifold(std::move(tm));
    const std::string prod = sub.product({cid, tid});
    std::vector<float> xf(static_cast<std::size_t>(C * T));
    for (int i = 0; i < C; ++i) for (int t = 0; t < T; ++t) xf[static_cast<std::size_t>(i * T + t)] = static_cast<float>(X(i, t));
    FieldSpec fr;
    fr.name = rec; fr.path = nxr::io::task::Subject::of(sub).node_path(prod, rec); fr.kind = "recording"; fr.manifold_id = prod;
    fr.data = view(xf, {C, T}); fr.array.chunks = Shape{C, chunk}; fr.array.compress = false; fr.cols.set("unit", "T");
    const std::string rid = sub.field(std::move(fr));
    if (good) {                                                   // the bad span of every run of zeros: sample i is inside [(i0 - 1/4)/fs, (i1 - 1/4)/fs)
      std::vector<nxr::bst::TimeSpan> spans;
      for (std::int64_t i = 0; i < T;) {
        if ((*good)[static_cast<std::size_t>(i)]) { ++i; continue; }
        std::int64_t j = i;
        while (j < T && !(*good)[static_cast<std::size_t>(j)]) ++j;
        spans.push_back({(static_cast<double>(i) - 0.25) / fs, (static_cast<double>(j) - 0.25) / fs, 0});
        i = j;
      }
      nxr::bst::TimeSpansOptions so;
      so.line_id = tid; so.of_field_id = rid; so.description = "bad segments";
      nxr::bst::time_spans(sub, "time/" + rec + "_bad_segments", spans, {"BAD"}, so);
    }
    if (plane) {
      std::vector<std::int32_t> chans(static_cast<std::size_t>(C));
      for (std::int64_t i = 0; i < C; ++i) chans[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
      SelectionSpec ss; ss.name = "plane_channels"; ss.path = ss.name; ss.type = "indices"; ss.manifold_id = cid; ss.data = view(chans, {C}); ss.picked = true;
      const std::string psel = sub.selection(std::move(ss));
      std::vector<float> pf(static_cast<std::size_t>(2 * C));
      for (int i = 0; i < 2; ++i) for (int c = 0; c < C; ++c) pf[static_cast<std::size_t>(i * C + c)] = static_cast<float>((*plane)(i, c));
      OperatorSpec op;
      op.name = "plane"; op.path = "plane"; op.kind = "rotation plane"; op.from_manifold_id = cid; op.layout = "dense"; op.n_rows = 2; op.n_cols = C;
      op.data = view(pf, {2, C}); op.array.compress = false;
      op.cols.set("to_manifold_id", cid).set("from_selection_id", psel);
      sub.operator_(std::move(op));
    }
  });
}

double row_rel(const std::vector<float>& got, const MatrixXd& want) {
  double worst = 0;
  for (Eigen::Index i = 0; i < want.rows(); ++i) {
    double d = 0, m = 0;
    for (Eigen::Index k = 0; k < want.cols(); ++k) { d = std::max(d, std::abs(static_cast<double>(got[static_cast<std::size_t>(i * want.cols() + k)]) - want(i, k))); m = std::max(m, std::abs(want(i, k))); }
    worst = std::max(worst, d / m);
  }
  return worst;
}
}  // namespace

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_psd_golden_" + std::to_string(::getpid()));
  fs::remove_all(root);
  Golden g(kDir + "/welch_golden.txt");
  CHECK(g.good());
  const MatrixXd x = g.mat("x"), gm = g.mat("good");
  const double fs = g.mat("fs")(0, 0);
  std::vector<std::uint8_t> good(static_cast<std::size_t>(gm.cols()));
  for (std::size_t i = 0; i < good.size(); ++i) good[i] = gm(0, static_cast<Eigen::Index>(i)) > 0.5;
  struct Entry { std::string name; MatrixXd psd; std::int64_t count = -1; };
  std::map<std::string, Entry> gold;                                                          // the goldens in file order: a matrix, and the segment count that follows some
  for (const char* n : {"scipy_hann_const_n400", "scipy_hann_const_n400_bad", "scipy_hann_const_n400_bad_segments", "scipy_hamming_linear_n301_ov100", "scipy_blackman_none_spectrum_n256_ov64", "nxr_qc_psd",
                        "nxr_chain", "nxr_chain_segments", "nxr_chain_bad", "nxr_chain_bad_segments", "bst_default_2s", "bst_default_2s_windows", "bst_bad_1", "bst_bad_1_windows", "bst_default_1s", "bst_default_1s_windows"})
    gold[n] = Entry{n, g.mat(n)};
  Golden rg(kDir + "/rotary_welch_golden.txt");
  const MatrixXd X4 = rg.mat("X"), A = rg.mat("A");
  rg.mat("fs");
  const MatrixXd rf400 = rg.mat("rotary_f_n400"), rwant400 = rg.mat("rotary_hann_const_n400"), rf301 = rg.mat("rotary_f_n301"), rwant301 = rg.mat("rotary_hamming_linear_n301_ov100");

  DatasetWriter ds = create_dataset(root, "demo");
  // a recording's frequency line is shared by every spectrum of its span length (one more length is another line): each case runs on a subject of its own
  make_subject(ds, "S2", X4, fs, 650, nullptr, &A);
  make_subject(ds, "S2b", X4, fs, 2400, nullptr, &A);
  SubjectWriter s2 = SubjectWriter::open(ds, "S2"), s2b = SubjectWriter::open(ds, "S2b");
  const double tol = 1e-5;                                                                     // float32 storage of the recording and of the result

  // each reference, by the preset that names it (the segment's seconds make the golden's length: 2 s = 400 samples at 200 Hz) and the parameters the golden states
  struct Case { const char* golden; const char* preset; Json set; bool bad; const char* count; };
  const std::vector<Case> cases = {
      {"scipy_hann_const_n400", "scipy", Json{{"segment_seconds", 2.0}}, false, nullptr},
      {"scipy_hann_const_n400_bad", "scipy", Json{{"segment_seconds", 2.0}}, true, "scipy_hann_const_n400_bad_segments"},
      {"scipy_hamming_linear_n301_ov100", "scipy", Json{{"segment_samples", 301}, {"overlap", 100.0 / 301}, {"window", "hamming"}, {"detrend", "linear"}}, false, nullptr},
      {"scipy_blackman_none_spectrum_n256_ov64", "scipy", Json{{"segment_samples", 256}, {"overlap", 0.25}, {"window", "blackman"}, {"detrend", "none"}, {"scaling", "spectrum"}}, false, nullptr},
      {"nxr_qc_psd", "scipy", Json{{"segment_seconds", 4.0}}, false, nullptr},                // nxr's qc.sensors.psd: scipy's with a 4 s window
      {"nxr_chain", "nxr-chain", Json{{"segment_seconds", 2.0}}, false, "nxr_chain_segments"},
      {"nxr_chain_bad", "nxr-chain", Json{{"segment_seconds", 2.0}}, true, "nxr_chain_bad_segments"},
      {"bst_default_2s", "brainstorm", Json{{"segment_seconds", 2.0}}, false, "bst_default_2s_windows"},
      {"bst_bad_1", "brainstorm", Json{{"segment_seconds", 2.0}}, true, "bst_bad_1_windows"},
      {"bst_default_1s", "brainstorm", Json{{"segment_seconds", 1.0}}, false, "bst_default_1s_windows"},
  };
  int n = 0;
  for (const Case& c : cases) {
    Json set = c.set;
    set["name"] = std::string("g") + std::to_string(n);
    set["bad_segments"] = c.bad ? "exclude" : "ignore";
    make_subject(ds, "G" + std::to_string(n), x, fs, 700, &good, nullptr);                  // 700-sample chunks: segments cross chunk edges
    make_subject(ds, "G" + std::to_string(n) + "b", x, fs, 1777, &good, nullptr);
    SubjectWriter s1 = SubjectWriter::open(ds, "G" + std::to_string(n)), s1b = SubjectWriter::open(ds, "G" + std::to_string(n) + "b");
    ++n;
    const nxrtest::PsdRun a = nxrtest::run_psd(ds, s1, "rest", set, c.preset);
    CHECK(!a.hit);
    const MatrixXd& want = gold.at(c.golden).psd;
    std::vector<std::int64_t> shape;
    const std::vector<float> got = nxrtest::psd_values(s1, a, &shape);
    CHECK(shape.size() == 2 && shape[0] == want.rows() && shape[1] == want.cols());
    const double e = row_rel(got, want);
    std::cout << "  " << c.golden << " (" << c.preset << (c.bad ? ", bad segments left out" : "") << "): " << e << "\n";
    CHECK(e < tol);
    if (c.count) CHECK_EQ(integer(read_statistic(s1, set["name"].get<std::string>()).row, "n_members"), static_cast<std::int64_t>(gold.at(c.count).psd(0, 0)));
    // the same samples in another chunk width: the same numbers to the last bit
    set["name"] = std::string("g") + std::to_string(n - 1);
    const nxrtest::PsdRun b = nxrtest::run_psd(ds, s1b, "rest", set, c.preset);
    CHECK(nxrtest::psd_values(s1b, b) == got);
  }
  // the preset's own defaults are what a caller gets with only the preset named (scipy: hann, periodic, constant, density; the stored options say so)
  make_subject(ds, "S1", x, fs, 700, &good, nullptr);
  SubjectWriter s1 = SubjectWriter::open(ds, "S1");
  {
    const nxrtest::PsdRun a = nxrtest::run_psd(ds, s1, "rest", Json{{"segment_seconds", 2.0}, {"name", "opts"}, {"bad_segments", "ignore"}}, "scipy");
    const Json o = read_statistic(s1, "opts").options;
    CHECK(o["window"] == "hann" && o["window_form"] == "periodic" && o["detrend"] == "constant" && o["scaling"] == "density" && o["segment_rule"] == "round");
    (void)a;
  }

  // THE ROTARY SPECTRUM against scipy on the complex pair (rotary_welch_golden.txt): the signed frequency line, +f the operator's x -> y, the two chunk widths the same to the last bit
  struct RCase { const MatrixXd* f; const MatrixXd* want; Json set; const char* name; };
  const std::vector<RCase> rcases = {{&rf400, &rwant400, Json{{"segment_seconds", 2.0}}, "r400"},
                                     {&rf301, &rwant301, Json{{"segment_samples", 301}, {"overlap", 100.0 / 301}, {"window", "hamming"}, {"detrend", "linear"}}, "r301"}};
  for (const RCase& c : rcases) {
    Json set = c.set;
    set["operator"] = "plane";
    set["name"] = c.name;
    const nxrtest::PsdRun a = nxrtest::run_definition("rotary_psd", ds, s2, "rest", set, "scipy");
    CHECK(!a.hit);
    const std::vector<float> got = nxrtest::psd_values(s2, a);
    const double e = row_rel(got, *c.want);
    std::cout << "  rotary " << c.name << ": " << e << "\n";
    CHECK(e < tol);
    const StoredStatistic st = read_statistic(s2, c.name);
    const auto line = nxr::io::task::Subject::of(s2).by_id(nxr::io::task::Cls::manifold, st.axes[1]["manifold_id"].get<std::string>());
    CHECK(line.has_value());
    const auto ax = nxr::io::task::Subject::of(s2).geometry(line->id).axes.at(0);
    CHECK(std::abs(*ax.origin - (*c.f)(0, 0)) < 1e-9 && std::abs(*ax.spacing - ((*c.f)(0, 1) - (*c.f)(0, 0))) < 1e-9 && ax.n_vertices == c.f->cols());   // the signed frequencies
    CHECK(nxrtest::psd_values(s2b, nxrtest::run_definition("rotary_psd", ds, s2b, "rest", set, "scipy")) == got);
  }

  // a rerun is a hit, in the cache the runner keeps; the preset is part of what is asked
  CHECK(nxrtest::run_psd(ds, s1, "rest", Json{{"segment_seconds", 2.0}, {"name", "opts"}, {"bad_segments", "ignore"}}, "scipy").hit);
  CHECK(!nxrtest::run_psd(ds, s1, "rest", Json{{"segment_seconds", 2.0}, {"name", "opts2"}, {"bad_segments", "ignore"}}, "brainstorm").hit);   // another preset, another run
  fs::remove_all(root);
  return nxrtest::finish("test_psd_golden");
}
