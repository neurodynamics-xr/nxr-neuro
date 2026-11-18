// nxr-io — a Brainstorm channel file's PROJECTORS and CTF COMPENSATION as Operators on the channels (register D168): nxr.convert.projectors.
//
//   <ch>_projector_<k>     Operator, kind `projector`, dense [M x M] f64, k = 1.. in the file's order; params_json status · comment · method · kind (ssp | ica) · components · order
//   <ch>_ctf_compensation  Operator, kind `ctf compensation`, csr [M x M] f64: entry (MEG channel, reference channel) = MegRefCoef; params_json n_meg · n_ref · grade
//
// SSP: P = I - U U^T over the removed columns of Components. ICA: W = Components^T, A = pinv(W), P = I - A[:, removed] W[removed, :].
// The arithmetic is fixed (ascending sums, products not fused, pinv = W^T (W W^T)^-1 by Gauss-Jordan with partial pivoting) so the bytes equal nxr's.
// An import stores the samples AS RECORDED: no projector is applied to them. The chunk service applies the recording's status-1 projectors to every block it loads
// (`joint_projector`, Brainstorm's in_fread), and the inverse applies them to the gain and the noise covariance as each reference does.
// An SSP projector's removed vectors are kept in its params_json (`vectors`, one array of M values each), so the projectors of a recording are composed as the
// references compose them: every selected SSP vector stacked and re-orthogonalised together (Brainstorm process_ssp2 BuildProjector, MNE-Python _make_projector).
#pragma once
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "nxr/bst/mat.h"
#include "nxr/io/database.h"
#include "nxr/io/store.h"
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

// A projector as stored: an Operator of kind `projector` on a channels manifold. `vectors`: an SSP's removed vectors (each [m]); empty for an ICA projector or one stored
// without them, which is then applied as its matrix [m x m].
struct StoredProjector {
  std::string name, kind, method;
  int order = 0, status = 0;
  std::size_t m = 0;
  std::vector<std::vector<double>> vectors;
  std::vector<double> matrix;
};
// The projector operators on a channels manifold, in the file's order (empty when it has none).
std::vector<StoredProjector> stored_projectors(const nxr::io::Database& db, const nxr::io::Store& store, const std::string& channels_manifold_id);
// How the selected projectors are composed into one [n x n] operator, the rule of each reference:
//   statuses     the entries used (Brainstorm: {1} when data are read (in_fread), {1, 2} for the inverse; MNE-Python: every projector of the info)
//   channels     the channels the operator acts on (empty: all m); each SSP vector is cut to them
//   renormalise  each cut vector rescaled to unit norm before the stacking (MNE-Python _make_projector); Brainstorm uses the vectors as they are
//   zero_tiny    entries of the orthonormal basis below 1e-15 set to zero (Brainstorm BuildProjector); MNE-Python keeps them
// The selected SSP vectors are stacked [n x k] and re-orthogonalised by their singular value decomposition; the vectors whose singular value is below 1% of the
// largest are dropped (both references), and the SSP operator I - U U^T takes the place of the first SSP entry; the others (ICA, or a projector stored as a matrix
// only) are applied as their matrices, the product taken in the file's order (the first applied first). Empty when nothing is selected or the result is I.
struct ProjectorRule {
  std::vector<int> statuses{1};
  std::vector<std::size_t> channels;
  bool renormalise = false;
  bool zero_tiny = true;
};
std::vector<double> joint_projector(const std::vector<StoredProjector>& projectors, const ProjectorRule& rule, std::vector<std::string>* used = nullptr);

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
