// nxr-pipelines — `montage create`: the standard 10-05 electrode montage on a template's own scalp, stored in the template's package (spec: docs/audits/2026-11-17-sleep-events-and-montages.md,
// sections 4.2 and 4.3). This file only PREPARES the inputs and hands them on:
//
//   graph        head [V x 3, F x 3] and head_landmarks [NAS, LPA, RPA]  --inion (the recorded convention)-->  4 landmarks  --nxr::meeg::tenFive-->  positions [345 x 3]
//   type checks  `head` is a surface and `head_landmarks` a points manifold of three points named NAS, LPA, RPA, in one coordinate system; the montage does not exist yet (or `replace`)
//   inion        by default `nxr::meeg::inionFromLandmarks` (the farthest crossing of the line from the nasion through the midpoint of the preauricular points with the scalp); the
//                convention is recorded in the manifold's producer, with how far each landmark moved to reach the scalp. A caller who knows the inion gives it (`inion`).
//   stored       the points manifold `eeg_1005` (positions in the head's coordinate system, metres), its names as a labelling `eeg_1005_names` (one level per point), the old 10-20 names as
//                `aliases` in its producer (T3 = T7, T4 = T8, T5 = P7, T6 = P8), the `system` partition `eeg_1005_system` (levels `10-20`, `10-10`, `10-05`: the narrowest system a point belongs to;
//                the 10-10 system is the levels `10-20` and `10-10` together, the 10-05 all), and the origin record: derived from the scalp's and the landmarks' (their licences joined).
//
// What it cannot resolve: these are the SYSTEM's positions on this head model, not where a cap sits on a person; the inion is a stated convention, not the bump the inion is.
#pragma once
#include <array>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>

#include "nxr/io.h"

namespace nxr::pipelines {

struct MontageCreateOptions {
  std::filesystem::path package;                 // the template (or any) package that holds `head` and `head_landmarks`; the montage is added to it (a working copy, not a frozen release)
  std::string subject;                           // its subject; empty: the package's only subject
  std::string scalp = "head", landmarks = "head_landmarks";   // the stored names
  std::string name = "eeg_1005";                 // the montage's manifold name
  std::optional<std::array<double, 3>> inion;    // the inion in the head's coordinates (metres); unset: the default convention
  double tolerance = 1e-7;                       // Cz stops when it moves less than this fraction of the nasion - inion distance
  int max_iterations = 100;                      // the most Cz iterations before refusing
  bool replace = false;                          // replace an existing montage of that name
  std::string created_by;                        // the program's stamp
};

struct MontageCreateResult {
  nxr::io::Json info;   // {name, points, systems: {10-05, 10-10, 10-20}, inion: {convention, point}, landmark_shift, cz_iterations, cz_residual, cs_key, warnings}
};

MontageCreateResult montage_create(const MontageCreateOptions& opt, const std::function<void(const nxr::io::Json&)>& progress = {});

}  // namespace nxr::pipelines
