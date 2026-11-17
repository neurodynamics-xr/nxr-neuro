// nxr-io — a SYNTHETIC RECORDING (`create recording --synthetic`): sine waves plus noise from a seed, for trying the toolkit, the documentation's examples and the smoke test of a
// release. Written into a subject package through the recording importer (`import_recording`), so it is stored exactly as an imported recording is (the chunk and shard grid, the
// envelope, the channel flags, the sensor manifold, the origin record), and every item says it is synthetic.
//
//   channel c, sample i:  x = sum over k of  amplitude / (k + 1) * sin(2 pi f_k i / sfreq + phase(c, k))  +  noise * N(0, 1)        (volts)
//
// The phases and the noise are drawn from a counter-based generator keyed by (seed, channel, sample), so the same options give the same samples on every machine and any
// part of the recording can be read without the rest. Nothing here is real data: the origin is `synthetic:<the options>`, licence CC0-1.0, redistributable.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "nxr/bst/session.h"
#include "nxr/formats/recording.h"
#include "nxr/io/subject_package.h"
#include "nxr/io/writer.h"

namespace nxr::convert {

struct SyntheticRecording {
  std::string name = "synthetic";          // the recording's name in the subject
  std::string acquisition = "rest";        // the session it is stored under
  std::int64_t n_channels = 8;             // EEG 001 ... (1 to 4096)
  double sfreq = 250.0;                    // Hz
  double duration = 60.0;                  // seconds (the recording holds round(sfreq * duration) samples)
  std::uint64_t seed = 1;
  std::vector<double> frequencies = {10.0};   // Hz, each below sfreq / 2
  double amplitude = 2e-5;                 // volts, of the first sine (the k-th has amplitude / (k + 1))
  double noise = 5e-6;                     // volts, the standard deviation of the white noise
  nxr::bst::Emit progress;
};

// The reader of the synthetic recording (a pure generator: no file); refuses options that make no recording (a count, rate, duration, frequency or size out of range).
std::unique_ptr<nxr::formats::RecordingReader> synthetic_reader(const SyntheticRecording& opt);
// The options as text, as the origin records them: `seed=1;channels=8;sfreq=250;duration=60;frequencies=10;amplitude=2e-05;noise=5e-06`.
std::string synthetic_description(const SyntheticRecording& opt);
// Write the recording into the subject `sub` of the package `w` writes; returns what `import_recording` returns.
nxr::io::Json create_synthetic(nxr::io::DatasetWriter& w, nxr::io::SubjectWriter& sub, const SyntheticRecording& opt);

}  // namespace nxr::convert
