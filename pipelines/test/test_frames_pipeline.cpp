// The cortical frames pipeline on a synthetic dataset: a template (finer, RAS) and a subject (SCS), each a two-hemisphere ellipsoid "brain" with its registered sphere.
// The template's frame is made from its own RAS axes; the subject's is carried from it through the registered spheres; the stored fields equal the library's, the canonical frames are written as the
// dataset writes them (matrix3 [V,3,3], singularities), a re-run is refused without --replace, and a subject without a sphere falls back to the signpost gauge.
#include <unistd.h>

#include <cmath>
#include <filesystem>
#include <iostream>

#include "brain_helpers.h"
#include "check.h"
#include "nxr/io/task/task.h"
#include "nxr/pipelines/frames.h"
#include "nxr/pipelines/input.h"
#include "run_definition.h"

using namespace nxr::io;
using namespace nxr::pipelines;
namespace fs = std::filesystem;

static std::vector<float> field_values(SubjectWriter& sub, const std::string& name, std::vector<std::int64_t>* shape = nullptr) {
  const auto f = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::field, .name = name, .any_status = true});
  if (!f) { std::printf("  missing field %s\n", name.c_str()); return {}; }
  ArrayReader r = ArrayReader::open(sub.store(), *f->path);
  if (shape) *shape = r.shape();
  const std::vector<std::uint8_t> raw = r.read();
  std::vector<float> out(raw.size() / r.itemsize());
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (r.dtype() == DType::Float32) std::memcpy(&out[i], raw.data() + i * 4, 4);
    else { std::int32_t v; std::memcpy(&v, raw.data() + i * 4, 4); out[i] = static_cast<float>(v); }
  }
  return out;
}

int main() {
  const char* tmp = std::getenv("TMPDIR");
  const fs::path root = fs::path(tmp ? tmp : fs::temp_directory_path().string()) / ("nxr_test_frames_pipeline_" + std::to_string(::getpid()));
  fs::remove_all(root);
  const Brain subj = brain(3, false), tmpl = brain(4, true);
  DatasetWriter ds = create_dataset(root, "demo");
  auto add = [&](const std::string& name, const Brain& b, bool sphere) {
    SubjectWriter sub = SubjectWriter::create(ds, name);
    sub.compose([&] {
      SurfaceOptions so;
      so.name = "cortex";
      if (sphere) so.sphere = b.Sph;
      add_surface(sub, so, b.P, b.F);
    });
  };
  add("T", tmpl, true);
  add("S1", subj, true);
  add("S2", subj, false);

  // the definition surface_frames, run through the runner (what `nxr-neuro run surface_frames` does); the stored fields are what the checks read
  auto frames_run = [&](SubjectWriter& sb, Json set, bool replace) { set["surface"] = "cortex"; if (set.contains("template")) set["template_surface"] = "cortex"; return nxrtest::run_definition(ds, sb, "surface_frames", set, replace); };
  auto node = [&](SubjectWriter& sb, const std::string& name, nxr::io::task::Cls cls) { return nxr::io::task::Subject::of(sb).find({.cls = cls, .name = name, .any_status = true}); };

  // the template: canonical frames computed from its sphere, its own RAS axes
  SubjectWriter t = SubjectWriter::open(ds, "T");
  frames_run(t, Json{{"convention", "ras"}}, false);
  std::vector<std::int64_t> shape;
  const std::vector<float> cf = field_values(t, "cortex_canonical_frames", &shape);
  CHECK_EQ(shape.size(), std::size_t(3));
  CHECK(shape.size() == 3 && shape[0] == tmpl.n && shape[1] == 3 && shape[2] == 3);
  const auto kind = node(t, "cortex_canonical_frames", nxr::io::task::Cls::field);
  CHECK(kind.has_value() && kind->kind == "canonical frames" && kind->value_type == "matrix3");
  CHECK(node(t, "cortex_frame_singularities", nxr::io::task::Cls::selection).has_value());
  CHECK(!field_values(t, "cortex_canonical_gauge_angle").empty());
  const auto rel = node(t, "cortex_anatomical_reliable", nxr::io::task::Cls::selection);
  CHECK(rel.has_value());
  const int reliable = rel && rel->n_elements ? static_cast<int>(*rel->n_elements) : 0;
  std::printf("  template: %d reliable of %d\n", reliable, tmpl.n);
  CHECK(reliable > tmpl.n / 2);
  const auto again = frames_run(t, Json{{"convention", "ras"}}, false);   // the same parameters: a hit (the stored frames are read, nothing is made twice)
  CHECK(!again.empty() && again[0].value("hit", false));
  bool refused = false;
  try { frames_run(t, Json{{"convention", "scs"}}, false); } catch (const std::exception&) { refused = true; }   // other parameters over the stored names: refused without replace
  CHECK(refused);
  frames_run(t, Json{{"convention", "ras"}}, true);                       // replaced: the stored canonical frames are read

  // the subject: carried from the template
  SubjectWriter s = SubjectWriter::open(ds, "S1");
  frames_run(s, Json{{"template", "T"}}, false);
  {
    const auto ang = node(s, "cortex_anatomical_angle", nxr::io::task::Cls::field);
    CHECK(ang.has_value() && ang->producer.dump().find("template:T") != std::string::npos);
  }
  // equal to the subject's own SCS-native frame, away from the poles
  const StoredFrames sf = read_stored_frames(s, "cortex", subj.n);
  CHECK(sf.canonical && sf.anatomical);
  const nxr::frames::Anatomical own = nxr::frames::anatomicalFrame(subj.P.data(), subj.n, subj.F.data(), subj.nF, subj.N, sf.e1, sf.e2, {});
  int both = 0, close = 0;
  for (int i = 0; i < subj.n; ++i) {
    if (!own.reliable[i] || !sf.anatomy.reliable[static_cast<std::size_t>(i)] || own.projection[i] < 0.6) continue;
    ++both;
    close += std::abs(std::remainder(own.angle[i] - sf.anatomy.angle[i], 2 * M_PI)) < 3 * M_PI / 180 && own.second[static_cast<std::size_t>(i)] == sf.anatomy.second[static_cast<std::size_t>(i)];
  }
  std::printf("  subject: %d vertices reliable in both, %d within 3 deg with the same label\n", both, close);
  CHECK(both > subj.n / 2 && close >= 0.97 * both);
  CHECK(!field_values(s, "cortex_anatomical_handed").empty());

  // a subject with no sphere: the SCS-native frame in the signpost gauge, recorded
  SubjectWriter s2 = SubjectWriter::open(ds, "S2");
  frames_run(s2, Json::object(), false);
  {
    const auto ang = node(s2, "cortex_anatomical_angle", nxr::io::task::Cls::field);
    CHECK(ang.has_value() && ang->producer.dump().find("signpost gauge") != std::string::npos);
  }
  bool nocarry = false;
  try { frames_run(s2, Json{{"template", "T"}}, true); } catch (const std::exception&) { nocarry = true; }
  CHECK(nocarry);                                           // carrying needs the registered sphere
  ds.close();
  fs::remove_all(root);
  return nxrtest::finish("test_frames_pipeline");
}
