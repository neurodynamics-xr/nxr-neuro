// FIF objects: source spaces (-src.fif), noise covariances (-cov.fif), forward solutions (-fwd.fif), inverse operators (-inv.fif), event lists
// (-eve.fif), annotation files (-annot.fif) and coordinate transforms (-trans.fif); read and (for all but the forward and inverse) write.
//
// Ported from MNE-Python (mne/source_space/_source_space.py, mne/cov.py, mne/forward/forward.py, mne/minimum_norm/inverse.py, mne/event.py,
// mne/annotations.py, mne/transforms.py, mne/_fiff/matrix.py; BSD-3-Clause, the MNE-Python contributors; see THIRD_PARTY_NOTICES.md). The
// goldens come from MNE-Python writing the files (test/golden/gen_fif_objects.py).
//
//   fif::File f(std::make_shared<FileSource>("sub-01-src.fif"));
//   fif::SourceSpaces src = fif::read_source_spaces(f);
//
// Rules:
// - Every tag of every block an object reads is parsed. What has no field is kept in the object's `other_tags` as generic tags, so nothing is dropped.
// - Values are as the file stores them (SI: metres, tesla, ...); the file's 32-bit floats are widened to double. Matrices are row-major as the
//   file stores them: named matrices (forward and inverse) are NOT transposed (MNE transposes them on reading; here `rows` x `cols` is the file's).
// - Vertex and triangle indices are zero-based (the file stores triangles one-based; MNE reads them zero-based, as does this reader).
// - A file of a kind that is not the one asked for is refused with the reason (`unsupported` for a variant not read, `format_error` for bad bytes).
#pragma once
#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "nxr/formats/fif.h"
#include "nxr/formats/fif_write.h"

namespace nxr::formats::fif {

namespace block {
constexpr std::int32_t kMne = 350, kMneSourceSpace = 351, kMneForwardSolution = 352, kMneParentMriFile = 353, kMneParentMeasFile = 354, kMneCov = 355,
                       kMneInverseSolution = 356, kMneEnv = 358, kMneEvents = 361;
}  // namespace block

namespace tag {
constexpr std::int32_t kParentFileId = 109, kMriWidth = 2010, kMriHeight = 2012, kMriDepth = 2014, kBemSurfNtri = 3104,
                       kBemSurfTriangles = 3106, kMneFileName = 3508, kMneSourceSpacePoints = 3510, kMneSourceSpaceNormals = 3511,
                       kMneSourceSpaceNpoints = 3512, kMneSourceSpaceSelection = 3513, kMneSourceSpaceNuse = 3514, kMneSourceSpaceNearest = 3515,
                       kMneSourceSpaceNearestDist = 3516, kMneSourceSpaceId = 3517, kMneSourceSpaceType = 3518, kMneSourceSpaceVertices = 3519,
                       kMneForwardSolution = 3520, kMneSourceOrientation = 3521, kMneIncludedMethods = 3522, kMneForwardSolutionGrad = 3523,
                       kMneCovKind = 3530, kMneCovDim = 3531, kMneCov = 3532, kMneCovDiag = 3533, kMneCovEigenvalues = 3534,
                       kMneCovEigenvectors = 3535, kMneCovNfree = 3536, kMneCovMethod = 3537, kMneCovScore = 3538, kMneInverseLeads = 3540,
                       kMneInverseFields = 3541, kMneInverseSing = 3542, kMneInverseLeadsWeighted = 3546, kMneInverseSourceOrientations = 3545,
                       kMneInverseSourceUnit = 3547, kMneEnvWorkingDir = 3550, kMneEnvCommandLine = 3551, kMneEventTriggerMask = 3565,
                       kMneEventComments = 3566, kMneSourceSpaceNtri = 3590, kMneSourceSpaceTriangles = 3591, kMneSourceSpaceNuseTri = 3592,
                       kMneSourceSpaceUseTriangles = 3593, kMneSourceSpaceNneighbors = 3594, kMneSourceSpaceNeighbors = 3595,
                       kMneSourceSpaceVoxelDims = 3596, kMneSourceSpaceInterpolator = 3597, kMneSourceSpaceMriFile = 3598,
                       kMneSourceSpaceDist = 3599, kMneSourceSpaceDistLimit = 3600, kMneEventList = 3561;
}  // namespace tag

// FIFFV_MNE_*: the source space types, source orientations, included methods and covariance kinds
namespace mne {
constexpr std::int32_t kSpaceSurface = 1, kSpaceVolume = 2, kSpaceDiscrete = 3;
constexpr std::int32_t kFixedOri = 1, kFreeOri = 2;
constexpr std::int32_t kMeg = 1, kEeg = 2, kMegEeg = 3;
constexpr std::int32_t kNoiseCov = 1, kSourceCov = 2, kFmriPriorCov = 3, kSignalCov = 4, kDepthPriorCov = 5, kOrientPriorCov = 6;
constexpr std::int32_t kSurfLeftHemi = 101, kSurfRightHemi = 102;
constexpr std::int32_t kCoordMriVoxel = 2001, kCoordRas = 2002;
}  // namespace mne

// A named matrix (a FIFFB_MNE_NAMED_MATRIX block): row-major `nrow` x `ncol` as stored (see the rules above); names empty when the file has none.
struct NamedMatrix {
  std::int32_t kind = 0;               // the tag kind of the data
  std::int32_t nrow = 0, ncol = 0;
  std::vector<std::string> row_names, col_names;
  std::vector<double> data;
};

// ---- Source spaces ------------------------------------------------------------------------------------------------------------------------

struct SourceSpace {
  std::string type;                    // "surf", "vol" or "discrete"
  std::int32_t id = -1;                // FIFFV_MNE_SURF_*_HEMI (101, 102) or -1 when the file has none
  std::int32_t coord_frame = 0;        // FIFFV_COORD_* (MRI 5, head 4)
  std::int32_t np = 0;                 // vertices
  std::vector<double> rr, nn;          // np x 3, metres, row-major; unit normals
  std::int32_t ntri = 0;
  std::vector<std::int32_t> tris;      // ntri x 3, zero-based
  std::int32_t nuse = 0;
  std::vector<std::int32_t> inuse;     // np entries, 1 where the vertex is used
  std::vector<std::int32_t> vertno;    // the used vertices (where `inuse` is set), as MNE derives them
  std::int32_t nuse_tri = 0;
  std::vector<std::int32_t> use_tris;  // nuse_tri x 3, zero-based
  std::vector<std::int32_t> nearest;   // np entries: the nearest used vertex (patch information); empty when absent
  std::vector<double> nearest_dist;    // np entries
  std::optional<SparseMatrix> dist;    // the upper triangle of the vertex distances, as stored (MNE adds its transpose)
  std::optional<double> dist_limit;
  std::optional<std::string> subject_his_id;
  // volume source spaces
  std::vector<std::int32_t> shape;                 // voxel grid dimensions (3) when the file has them
  std::optional<CoordTrans> src_mri_t, vox_mri_t, mri_ras_t;
  std::optional<SparseMatrix> interpolator;
  std::optional<std::string> mri_file, mri_volume_name, seg_name;
  std::optional<std::int32_t> mri_width, mri_height, mri_depth;
  std::vector<std::vector<std::int32_t>> neighbor_vert;   // per vertex, as MNE splits NEIGHBORS by NNEIGHBORS; empty when absent
  std::vector<GenericTag> other_tags;                     // tags of the source-space block no field takes
  std::vector<GenericTag> other_mri_tags;                 // the same for its parent-MRI block
};

struct SourceSpaces {
  std::vector<SourceSpace> spaces;
  std::optional<Id> env_id;                             // the file's FIFFB_MNE_ENV block
  std::optional<std::string> working_dir, command_line;
};

SourceSpaces read_source_spaces(const File& f);                                           // refuses a file with none
void write_source_spaces(TagWriter& w, const SourceSpaces& s);                            // the FIFFB_MNE block of a -src.fif file
void write_source_spaces_file(ByteSink& out, const SourceSpaces& s, const Id& file_id);   // a whole -src.fif file

// ---- Covariance ---------------------------------------------------------------------------------------------------------------------------

struct Covariance {
  std::int32_t kind = mne::kNoiseCov;
  bool diag = false;
  std::int32_t dim = 0;
  std::int32_t nfree = -1;             // -1 when the file has none
  std::vector<std::string> names;
  std::vector<double> data;            // diagonal (dim) or the full symmetric matrix (dim x dim, row-major; the file stores its lower triangle)
  std::vector<double> eig;             // eigenvalues and eigenvectors when the file has them (eigvec: eigvec_rows x eigvec_cols, row-major)
  std::vector<double> eigvec;
  std::int32_t eigvec_rows = 0, eigvec_cols = 0;
  std::vector<Projector> projs;
  std::vector<std::string> bads;
  std::optional<std::string> method;
  std::optional<double> score;         // the negative log-likelihood
  std::vector<GenericTag> other_tags;
};

// The covariance blocks anywhere in the file, in file order (a -cov.fif has one; an inverse operator has several).
std::vector<Covariance> read_covariances(const File& f);
// The first of a kind (default: the noise covariance); refuses when there is none.
Covariance read_covariance(const File& f, std::int32_t kind = mne::kNoiseCov);
void write_covariance(TagWriter& w, const Covariance& c);                         // a FIFFB_MNE_COV block
void write_covariance_file(ByteSink& out, const Covariance& c, const Id& file_id);

// ---- Forward solutions and inverse operators (read) ---------------------------------------------------------------------------------------

// The "parent" information the forward and inverse files carry: the MRI and measurement files they came from, the channels, the transforms.
struct ParentInfo {
  std::optional<std::string> mri_file, meas_file;
  std::optional<Id> mri_id, meas_id;   // the files' ids (FIFF_PARENT_FILE_ID)
  std::optional<Id> meas_block_id;     // FIFF_PARENT_BLOCK_ID of the measurement file, where MNE writes the measurement id
  std::vector<ChannelInfo> channels;
  std::vector<std::string> bads;
  std::optional<CoordTrans> mri_head_t, dev_head_t, ctf_head_t;
  std::int32_t custom_ref_applied = 0;
  std::vector<GenericTag> other_mri_tags, other_meas_tags;
};

struct ForwardSolution {   // one FIFFB_MNE_FORWARD_SOLUTION block (a file has one for MEG, one for EEG, or both)
  std::int32_t methods = 0;            // FIFFV_MNE_MEG 1, EEG 2
  std::int32_t source_ori = 0;         // FIFFV_MNE_FIXED_ORI 1, FREE_ORI 2
  std::int32_t coord_frame = 0;
  std::int32_t nsource = 0, nchan = 0;
  NamedMatrix sol;                     // as stored: rows are source components (nsource or 3 nsource), columns the channels
  std::optional<NamedMatrix> sol_grad;
  std::vector<GenericTag> other_tags;
};

struct ForwardFile {
  std::vector<ForwardSolution> solutions;
  SourceSpaces src;
  ParentInfo parent;
  std::optional<std::string> working_dir, command_line;
};
ForwardFile read_forward(const File& f);

struct InverseOperator {
  std::int32_t methods = 0, source_ori = 0, nsource = 0, nchan = 0, coord_frame = 0;
  std::optional<std::int32_t> units;   // FIFF_UNIT_AM 202, AM_M2 203, AM_M3 204
  std::vector<double> source_nn;       // the orientation of each source (rows x 3, row-major, rows stored in source_nn_rows)
  std::int32_t source_nn_rows = 0;
  std::vector<double> sing;            // singular values
  bool eigen_leads_weighted = false;
  NamedMatrix eigen_leads;             // as stored (MNE transposes)
  NamedMatrix eigen_fields;
  std::optional<Covariance> noise_cov, source_cov, orient_prior, depth_prior, fmri_prior;
  std::vector<Projector> projs;
  SourceSpaces src;
  ParentInfo parent;
  std::vector<GenericTag> other_tags;
};
InverseOperator read_inverse(const File& f);

// ---- Events, annotations, transforms ------------------------------------------------------------------------------------------------------

struct EventFile {
  std::vector<std::array<std::int32_t, 3>> events;          // (sample, previous value, new value), as MNE's (N, 3) array
  std::vector<std::pair<std::string, std::int32_t>> event_id;   // names and their values, from the description or comments tag; may be empty
  std::vector<GenericTag> other_tags;
};
EventFile read_events(const File& f);
void write_events(TagWriter& w, const EventFile& e);       // the FIFFB_MNE_EVENTS block
void write_events_file(ByteSink& out, const EventFile& e, const Id& file_id);

// An annotation file (-annot.fif): the annotation block of a raw file stored alone.
AnnotationSet read_annotations_file(const File& f);        // refuses a file with none
void write_annotations(TagWriter& w, const AnnotationSet& a);   // the FIFFB_MNE_ANNOTATIONS block (as inside a raw file)
void write_annotations_file(ByteSink& out, const AnnotationSet& a, const Id& file_id);

// A -trans.fif file: the first coordinate transform in the file (MNE's read_trans). Refuses a file with none.
CoordTrans read_transform(const File& f);
void write_transform_file(ByteSink& out, const CoordTrans& t, const Id& file_id);

// SSP projectors and bad channels as MNE writes them into the files' blocks (a FIFFB_PROJ block; a FIFFB_MNE_BAD_CHANNELS block, nothing for none).
void write_projectors(TagWriter& w, const std::vector<Projector>& projs);
void write_bad_channels(TagWriter& w, const std::vector<std::string>& bads);

// The sparse matrix of a tag (`File::value` does not decode sparse matrices).
SparseMatrix read_sparse(const File& f, const TagHeader& t);

}  // namespace nxr::formats::fif
