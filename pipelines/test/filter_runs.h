// Test helper: a temporal filter of a recording as the shipped DEFINITIONS `lowpass`, `highpass`, `bandpass`, `bandstop` and `notch` run through the runner (what `nxr-neuro run bandpass --all` does). A
// FIR filter that removes the record's mean reads the stored channel means: made first by `run measure --set channel_span_mean` when they are not stored. The hand-wired filter_compute is gone.
#pragma once
#include <string>

#include "nxr/signal/filters.h"
#include "nxr/io.h"
#include "nxr/pipelines/definition.h"
#include "nxr/pipelines/run_shipped.h"
#include "nxr/pipelines/runner.h"

namespace nxrtest {

struct FilterComputeOptions {
  std::string recording, name;
  nxr::signal::filters::FilterOptions filter;
  bool replace = false;
};
struct FilterComputeResult {
  bool hit = false;
  nxr::io::Json info;   // name, path, samples
};

inline FilterComputeResult filter_compute(nxr::io::DatasetWriter& ds, nxr::io::SubjectWriter& sub, const FilterComputeOptions& o) {
  using nxr::signal::filters::Kind;
  const Kind k = o.filter.kind;
  const std::string def = k == Kind::Lowpass ? "lowpass" : k == Kind::Highpass ? "highpass" : k == Kind::Bandpass ? "bandpass" : k == Kind::Bandstop ? "bandstop" : "notch";
  nxr::io::Json set = nxr::io::Json{{"recording", o.recording}};
  if (!o.name.empty()) set["name"] = o.name;
  if (k == Kind::Lowpass || k == Kind::Bandpass || k == Kind::Bandstop) set["upper"] = o.filter.upperEdge;
  if (k == Kind::Highpass || k == Kind::Bandpass || k == Kind::Bandstop) set["lower"] = o.filter.lowerEdge;
  if (k == Kind::Notch) set["frequencies"] = o.filter.frequencies;
  if (o.filter.notchBandwidth) set["notch_bandwidth"] = *o.filter.notchBandwidth;
  if (o.filter.method == nxr::signal::filters::Method::Iir) set["method"] = "iir";
  set["remove_mean"] = o.filter.removeMean;
  // the method unset is the reference's: FIR, but IIR for bandstop and notch
  const bool fir = o.filter.method ? *o.filter.method == nxr::signal::filters::Method::Fir : (k != Kind::Bandstop && k != Kind::Notch);
  const bool fir_mean = fir && o.filter.removeMean;
  if (fir_mean) {
    nxr::pipelines::graph::RunOptions mo;
    mo.chunks = "all";
    mo.preset = "channel_span_mean";
    try { nxr::pipelines::graph::run(ds, sub, nxr::pipelines::graph::load("measure"), nxr::io::Json{{"field", o.recording}, {"recording", o.recording}}, mo); } catch (const nxr::io::Error& e) { if (e.code() != nxr::io::code::already_exists) throw; }
  }
  nxr::pipelines::graph::RunOptions ro;
  ro.chunks = "all";
  ro.replace = o.replace;
  const auto runs = nxr::pipelines::graph::run(ds, sub, nxr::pipelines::graph::load(def), set, ro);
  FilterComputeResult r;
  r.hit = !runs.empty() && runs[0].value("hit", false);
  const std::string name = o.name.empty() ? o.recording + "_" + def : o.name;
  r.info["name"] = name;
  if (const auto f = nxr::io::task::Subject::of(sub).find({.cls = nxr::io::task::Cls::field, .name = name, .any_status = true})) {
    if (f->path) r.info["path"] = *f->path;
  }
  return r;
}

}  // namespace nxrtest
