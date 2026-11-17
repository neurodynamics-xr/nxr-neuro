// nxr-neuro eigen — the verbs of `nxr eigen` that nxr-compute can answer .
//
//   nxr-neuro eigen solve --dataset D/N --subject S [--modes K] [--surface NAME] [--replace]
//                         [--shift absolute|relative] [--sigma S] [--slice N] [--normalize] [--star1-floor F]
//                         [--mollify D] [--flip-eps E]
//
// The default is the REFERENCE method of nxr and Rheome (one shift-invert solve per ladder at the absolute sigma -1e-3, no re-orthonormalisation, no *1 floor).
// The options name the other methods: --shift relative (the scale-aware ladder), --slice N (spectrum slicing, windows of N), --normalize (M-orthonormalise
// afterwards), --star1-floor F (raise *1 to F; nxr has none, Rheome used 1e-6), --mollify D (mollification margin relative to the mean edge length, default 1e-6) and
// --flip-eps E (flip while the cotan weight is below -E, default 1e-6). They are recorded in the result's provenance and belong to the cache key.
//
// The same arguments as `nxr eigen solve`, the same JSON lines; the result line adds "cache": "hit" | "miss" (a hit reads the stored result, a miss solves and stores).
#include <cmath>
#include <iostream>

#include "cli.h"
#include "nxr/pipelines/eigen.h"

using namespace nxr::io;

int eigen_main(const std::vector<std::string>& argv) {
  if (argv.empty() || argv[0] != "solve") throw cli::Usage("nxr-neuro eigen: error: the following arguments are required: action (solve)");
  const std::vector<std::string> rest(argv.begin() + 1, argv.end());
  const cli::Args a = cli::parse("nxr-neuro eigen solve", rest, {"--dataset", "--subject", "--modes", "--surface", "--shift", "--sigma", "--slice", "--star1-floor", "--mollify", "--flip-eps"}, {"--replace", "--normalize"}, {"--dataset", "--subject"});
  nxr::pipelines::EigenSolveOptions opt;
  if (const auto m = a.maybe("--modes")) opt.modes = std::stoi(*m);
  opt.surface = a.maybe("--surface");
  if (const auto v = a.maybe("--shift")) {
    if (*v == "absolute") opt.method.solve.shift = nxr::manifold::solve::HodgeShift::Absolute;
    else if (*v == "relative") opt.method.solve.shift = nxr::manifold::solve::HodgeShift::RelativeLadder;
    else throw cli::Usage("nxr-neuro eigen solve: error: --shift must be absolute or relative");
  }
  if (const auto v = a.maybe("--sigma")) opt.method.solve.sigma = std::stod(*v);
  if (const auto v = a.maybe("--slice")) opt.method.solve.slice = std::stoi(*v);
  opt.method.solve.normalize = a.flag.count("--normalize") > 0;
  if (const auto v = a.maybe("--star1-floor")) { const double f = std::stod(*v); if (f > 0) opt.method.star1_floor = f; }
  // the intrinsic Delaunay construction's two numerical constants (the Studio's mollify and flipEps; nxr's --mollify / --flip-eps), recorded and in the cache key
  if (const auto v = a.maybe("--mollify")) opt.method.mollify = std::stod(*v);
  if (const auto v = a.maybe("--flip-eps")) opt.method.flip_eps = std::stod(*v);
  opt.replace = a.flag.count("--replace") > 0;
  DatasetWriter ds = open_dataset(a);
  OpenSubjectOptions oo;
  oo.created_by = cli::stamp();
  SubjectWriter sub = SubjectWriter::open(ds, a.get("--subject"), oo);
  const auto r = nxr::pipelines::eigen_solve(ds, sub, opt, [](const Json& j) { cli::emit(j); });
  Json out = Json{{"stage", "result"}};
  for (auto it = r.info.begin(); it != r.info.end(); ++it) out[it.key()] = it.value();
  out["solve"] = Json{{"exact", r.modes.n_exact}, {"coexact", r.modes.n_coexact}, {"seconds", std::round(r.seconds * 1000) / 1000}};
  cli::emit(out);
  return 0;
}
