// nxr-io — a Brainstorm channel file's PROJECTORS and CTF COMPENSATION as Operators on the channels (register D168): nxr.convert.projectors.
//
//   <ch>_projector_<k>     Operator, kind `projector`, dense [M x M] f64, k = 1.. in the file's order; params_json status · comment · method · kind (ssp | ica) · components · order
//   <ch>_ctf_compensation  Operator, kind `ctf compensation`, csr [M x M] f64: entry (MEG channel, reference channel) = MegRefCoef; params_json n_meg · n_ref · grade
//
// SSP: P = I - U U^T over the removed columns of Components. ICA: W = Components^T, A = pinv(W), P = I - A[:, removed] W[removed, :].
// The arithmetic is fixed (ascending sums, products not fused, pinv = W^T (W W^T)^-1 by Gauss-Jordan with partial pivoting) so the bytes equal nxr's.
// An import APPLIES the status-1 projectors to the samples it copies (in the file's order) and records on the recording what it did with each entry.
#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/bst/mat.h"
#include "nxr/io/writer.h"

namespace nxr::bst {

struct ProjectorEntry {
  int order = 0;                              // 1.. in the file
  std::string comment, method, kind;          // kind: "ssp" | "ica"
  int status = 0;                             // 0 inactive · 1 active · 2 already applied to the samples
  std::size_t m = 0, k = 0;                   // channels, components
  std::vector<double> components;             // [m x k] row-major
  std::vector<bool> mask;                     // [k] the removed components
};

std::vector<ProjectorEntry> projector_entries(const std::map<std::string, Value>& channel_mat);
std::vector<double> projector_matrix(const ProjectorEntry& e);                                  // [m x m] row-major
// the product of the status-1 projectors in the file's order (the first applied first); empty when none is active
std::vector<double> applied_matrix(const std::vector<ProjectorEntry>& entries);
// P x for x [m x n] float32 row-major -> float32 (float64 inside); a channel P leaves alone is copied
void project(const std::vector<double>& p, std::size_t m, const std::vector<float>& x, std::size_t n, std::vector<float>& y);

struct CtfGrade { int current = 0, destination = 0; };
std::optional<CtfGrade> ctf_grade(const Value* sfile);                                          // a raw link's currCtfComp / destCtfComp

// the operators; returns the projectors' names / the compensation's name (nullopt when the file has no MegRefCoef)
std::vector<std::string> export_projectors(nxr::io::SubjectWriter& sub, const std::vector<ProjectorEntry>& entries, const std::string& name, const std::string& channels_id,
                                           const std::string& acquisition);
std::optional<std::string> export_compensation(nxr::io::SubjectWriter& sub, const std::map<std::string, Value>& channel_mat, const std::string& name, const std::string& channels_id,
                                               const std::string& acquisition, const std::optional<CtfGrade>& grade);

// what a recording's provenance says: `projectors` and `ctf_compensation` (an object holding the keys that apply; empty when neither)
nxr::io::Json recording_record(const std::vector<ProjectorEntry>& entries, const std::map<std::string, Value>& channel_mat, const std::string& name, const std::optional<CtfGrade>& grade);

}  // namespace nxr::bst
