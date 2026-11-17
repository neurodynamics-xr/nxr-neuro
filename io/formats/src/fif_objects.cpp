// FIF objects: source spaces, covariances, forward solutions, inverse operators, events, annotation files, transforms. Ported from MNE-Python
// (mne/source_space/_source_space.py, mne/cov.py, mne/forward/forward.py, mne/minimum_norm/inverse.py, mne/event.py, mne/annotations.py,
// mne/_fiff/{matrix,proj}.py; BSD-3-Clause, Copyright the MNE-Python contributors; see THIRD_PARTY_NOTICES.md).
#include "nxr/formats/fif_objects.h"

#include <algorithm>
#include <cmath>
#include <set>

#include "fif_info.h"
#include "fif_internal.h"
#include "nlohmann/json.hpp"
#include "nxr/formats/endian.h"
#include "nxr/formats/errors.h"

namespace nxr::formats::fif {

namespace {

std::int64_t one_int(const Value& v, const char* what) {
  if (!v.ints.empty()) return v.ints[0];
  if (!v.reals.empty()) return static_cast<std::int64_t>(v.reals[0]);
  throw format_error(std::string("FIF: the tag for ") + what + " is empty");
}
double one_real(const Value& v, const char* what) {
  if (!v.reals.empty()) return v.reals[0];
  if (!v.ints.empty()) return static_cast<double>(v.ints[0]);
  throw format_error(std::string("FIF: the tag for ") + what + " is empty");
}
std::vector<std::int32_t> ints32(const Value& v) {
  std::vector<std::int32_t> o;
  if (!v.ints.empty()) for (auto x : v.ints) o.push_back(static_cast<std::int32_t>(x));
  else for (double x : v.reals) o.push_back(static_cast<std::int32_t>(x));
  return o;
}
std::vector<double> reals_of(const Value& v) {
  if (!v.reals.empty()) return v.reals;
  std::vector<double> o;
  for (auto x : v.ints) o.push_back(static_cast<double>(x));
  return o;
}
bool is_sparse(const TagHeader& t) { return (t.type & (dtype::kSparseCcs | dtype::kSparseRcs)) != 0; }

GenericTag generic_of(const File& f, const TagHeader& t) {
  GenericTag g;
  g.kind = t.kind;
  g.type = t.type;
  try {
    g.value = f.value(t);
  } catch (const nxr::io::Error&) {
    g.value.bytes = f.bytes(t);
  }
  return g;
}

bool skip_tag(const TagHeader& t) { return t.kind == tag::kNop || t.kind == tag::kBlockId || t.kind == tag::kParentBlockId; }

NamedMatrix read_named(const File& f, const Block& scope, std::int32_t matkind, bool* found) {
  NamedMatrix m;
  *found = false;
  const Block* node = nullptr;
  if (scope.kind == block::kMneNamedMatrix) {
    if (find_tag(scope, matkind)) node = &scope;
  } else {
    for (const Block& c : scope.children)
      if (c.kind == block::kMneNamedMatrix && find_tag(c, matkind)) { node = &c; break; }
  }
  if (!node) return m;
  *found = true;
  const Value v = f.value(*find_tag(*node, matkind));
  if (v.dims.size() != 2) throw format_error("FIF: a named matrix (tag kind " + std::to_string(matkind) + ") is not two-dimensional");
  m.kind = matkind;
  m.nrow = v.dims[0];
  m.ncol = v.dims[1];
  m.data = reals_of(v);
  if (const TagHeader* t = find_tag(*node, tag::kMneNrow))
    if (one_int(f.value(*t), "the row count") != m.nrow) throw format_error("FIF: the number of rows in a matrix and its FIFF_MNE_NROW tag do not match");
  if (const TagHeader* t = find_tag(*node, tag::kMneNcol))
    if (one_int(f.value(*t), "the column count") != m.ncol) throw format_error("FIF: the number of columns in a matrix and its FIFF_MNE_NCOL tag do not match");
  auto split = [](const std::string& s) {   // MNE splits these lists on ':' without the {COLON} escape
    std::vector<std::string> o;
    if (s.empty()) return o;
    std::size_t st = 0;
    for (;;) {
      const std::size_t at = s.find(':', st);
      o.push_back(s.substr(st, at == std::string::npos ? std::string::npos : at - st));
      if (at == std::string::npos) break;
      st = at + 1;
    }
    return o;
  };
  if (const TagHeader* t = find_tag(*node, tag::kMneRowNames)) m.row_names = split(f.value(*t).text);
  if (const TagHeader* t = find_tag(*node, tag::kMneColNames)) m.col_names = split(f.value(*t).text);
  return m;
}

std::vector<std::string> read_bad_channels(const File& f, const Block& scope) {
  for (const Block* bb : find_blocks(scope, block::kMneBadChannels))
    if (const TagHeader* t = find_tag(*bb, tag::kMneChNameList)) return detail::split_name_list(f.value(*t).text);
  return {};
}

void read_env(const File& f, const Block& root, std::optional<Id>* id, std::optional<std::string>* wd, std::optional<std::string>* cmd) {
  const auto envs = find_blocks(root, block::kMneEnv);
  if (envs.empty()) return;
  if (envs[0]->id) *id = envs[0]->id;
  if (const TagHeader* t = find_tag(*envs[0], tag::kMneEnvWorkingDir)) *wd = f.value(*t).text;
  if (const TagHeader* t = find_tag(*envs[0], tag::kMneEnvCommandLine)) *cmd = f.value(*t).text;
}

void write_env(TagWriter& w, const std::optional<Id>& id, const std::optional<std::string>& wd, const std::optional<std::string>& cmd) {
  if (!id && !wd && !cmd) return;
  w.start_block(block::kMneEnv);
  if (id) w.write_id(tag::kBlockId, *id);
  if (wd) w.write_string(tag::kMneEnvWorkingDir, *wd);
  if (cmd) w.write_string(tag::kMneEnvCommandLine, *cmd);
  w.end_block(block::kMneEnv);
}

// JSON text as Python's json.dumps writes it (", " and ": " separators, ASCII only): the annotation tags hold JSON MNE-Python wrote, and what we write
// matches it byte for byte
std::string py_dumps(const nlohmann::ordered_json& j) {
  if (j.is_array()) {
    std::string s = "[";
    for (std::size_t i = 0; i < j.size(); ++i) { if (i) s += ", "; s += py_dumps(j[i]); }
    return s + "]";
  }
  if (j.is_object()) {
    std::string s = "{";
    bool first = true;
    for (auto it = j.begin(); it != j.end(); ++it) { if (!first) s += ", "; first = false; s += nlohmann::ordered_json(it.key()).dump(-1, ' ', true) + ": " + py_dumps(it.value()); }
    return s + "}";
  }
  return j.dump(-1, ' ', true);
}

void write_file(ByteSink& out, const Id& id, const std::function<void(TagWriter&)>& body) {
  TagWriter w(out);
  w.start_file(id);
  body(w);
  w.end_file();
}

}  // namespace

// ---- sparse -------------------------------------------------------------------------------------------------------------------------------

SparseMatrix read_sparse(const File& f, const TagHeader& t) {
  const std::uint32_t coding = t.type & 0x00F00000u;
  if ((t.type & dtype::kMatrix) == 0 || (coding != dtype::kSparseCcs && coding != dtype::kSparseRcs))
    throw format_error("FIF: the tag (kind " + std::to_string(t.kind) + ") is not a sparse matrix");
  const std::uint32_t et = t.type & 0xFFFFu;
  if (et != dtype::kFloat && et != dtype::kDouble) throw refused("FIF: a sparse matrix of element type " + std::to_string(et) + " is not read");
  const std::size_t es = et == dtype::kFloat ? 4 : 8;
  const std::vector<unsigned char> b = f.bytes(t);
  if (b.size() < 16) throw format_error("FIF: a sparse matrix tag is too short");
  const unsigned char* p = b.data();
  const std::int32_t ndim = load_i32(p + b.size() - 4, false);
  if (ndim != 2) throw refused("FIF: only two-dimensional sparse matrices are read");
  const std::int32_t nnz = load_i32(p + b.size() - 16, false), nrow = load_i32(p + b.size() - 12, false), ncol = load_i32(p + b.size() - 8, false);
  SparseMatrix m;
  m.row_compressed = coding == dtype::kSparseRcs;
  m.nrow = nrow;
  m.ncol = ncol;
  if (nnz < 0 || nrow < 0 || ncol < 0) throw format_error("FIF: a sparse matrix has negative dimensions");
  const std::size_t ptr = static_cast<std::size_t>(m.row_compressed ? nrow : ncol) + 1;
  if (static_cast<std::size_t>(nnz) * es + static_cast<std::size_t>(nnz) * 4 + ptr * 4 + 16 > b.size()) throw format_error("FIF: a sparse matrix tag is shorter than its dimensions say");
  for (std::int32_t i = 0; i < nnz; ++i) m.data.push_back(es == 4 ? load_f32(p + 4 * i, false) : load_f64(p + 8 * i, false));
  const std::size_t o1 = static_cast<std::size_t>(nnz) * es;
  for (std::int32_t i = 0; i < nnz; ++i) m.indices.push_back(load_i32(p + o1 + 4 * i, false));
  const std::size_t o2 = o1 + static_cast<std::size_t>(nnz) * 4;
  for (std::size_t i = 0; i < ptr; ++i) m.indptr.push_back(load_i32(p + o2 + 4 * i, false));
  if (m.indptr.back() > nnz || std::any_of(m.indptr.begin(), m.indptr.end(), [](std::int32_t x) { return x < 0; }))
    throw refused("FIF: a sparse matrix with inconsistent index arrays (MNE-C wrote some unswapped) is not read");
  return m;
}

// ---- projectors, bad channels -------------------------------------------------------------------------------------------------------------

void write_projectors(TagWriter& w, const std::vector<Projector>& projs) {
  if (projs.empty()) return;
  w.start_block(block::kProj);
  for (const Projector& p : projs) {
    w.start_block(block::kProjItem);
    w.write_name_list(tag::kProjItemChNameList, p.channels);
    w.write_string(tag::kName, p.desc);
    w.write_int(tag::kNchan, static_cast<std::int32_t>(p.channels.size()));
    w.write_int(tag::kProjItemKind, p.kind);
    if (p.kind == 1) w.write_float(3412, 0.0);   // FIFF_PROJ_ITEM_TIME of a field projector
    w.write_int(tag::kProjItemNvec, p.nvec);
    w.write_int(tag::kMneProjItemActive, p.active ? 1 : 0);
    w.write_float_matrix(tag::kProjItemVectors, p.vectors, p.nvec, static_cast<std::int32_t>(p.channels.size()));
    if (p.explained_var) w.write_float(tag::kMneIcaPcaExplainedVar, *p.explained_var);
    w.end_block(block::kProjItem);
  }
  w.end_block(block::kProj);
}

void write_bad_channels(TagWriter& w, const std::vector<std::string>& bads) {
  if (bads.empty()) return;
  w.start_block(block::kMneBadChannels);
  w.write_name_list(tag::kMneChNameList, bads);
  w.end_block(block::kMneBadChannels);
}

// ---- source spaces ------------------------------------------------------------------------------------------------------------------------

namespace {

SourceSpace read_one_source_space(const File& f, const Block& b) {
  SourceSpace s;
  std::set<std::int32_t> seen;
  std::vector<std::int32_t> nneigh, neigh;
  bool have_type = false, have_npoints = false, have_frame = false, have_points = false, have_normals = false, have_nuse = false;
  std::optional<std::int32_t> nuse_tri, ntri_tag;
  std::vector<std::int32_t> nearest;
  std::vector<double> nearest_dist;
  bool have_nearest = false, have_nearest_dist = false;
  std::vector<std::int32_t> tris_raw, use_tris_raw, selection;
  bool have_tris = false, have_use_tris = false, have_selection = false;
  for (const TagHeader& t : b.tags) {
    if (skip_tag(t)) continue;
    switch (t.kind) {
      case tag::kMneSourceSpaceId: s.id = static_cast<std::int32_t>(one_int(f.value(t), "the source space id")); break;
      case tag::kMneSourceSpaceType: {
        const auto ty = one_int(f.value(t), "the source space type");
        have_type = true;
        if (ty == mne::kSpaceSurface) s.type = "surf";
        else if (ty == mne::kSpaceVolume) s.type = "vol";
        else if (ty == mne::kSpaceDiscrete) s.type = "discrete";
        else throw format_error("FIF: unknown source space type " + std::to_string(ty));
        break;
      }
      case tag::kMneSourceSpaceVoxelDims: s.shape = ints32(f.value(t)); break;
      case tag::kCoordTrans: s.src_mri_t = detail::decode_coord_trans(f.bytes(t)); break;
      case tag::kMneSourceSpaceNneighbors: nneigh = ints32(f.value(t)); break;
      case tag::kMneSourceSpaceNeighbors: neigh = ints32(f.value(t)); break;
      case tag::kComment: s.seg_name = f.value(t).text; break;
      case tag::kMneSourceSpaceNpoints: s.np = static_cast<std::int32_t>(one_int(f.value(t), "the vertex count")); have_npoints = true; break;
      case tag::kBemSurfNtri: ntri_tag = static_cast<std::int32_t>(one_int(f.value(t), "the triangle count")); break;
      case tag::kMneSourceSpaceNtri: if (!ntri_tag) ntri_tag = static_cast<std::int32_t>(one_int(f.value(t), "the triangle count")); break;
      case tag::kMneCoordFrame: s.coord_frame = static_cast<std::int32_t>(one_int(f.value(t), "the coordinate frame")); have_frame = true; break;
      case tag::kMneSourceSpacePoints: { const Value v = f.value(t); s.rr = reals_of(v); have_points = true; if (v.dims.size() != 2 || v.dims[1] != 3) throw format_error("FIF: the source space vertices are not an n x 3 matrix"); break; }
      case tag::kMneSourceSpaceNormals: { const Value v = f.value(t); s.nn = reals_of(v); have_normals = true; if (v.dims.size() != 2 || v.dims[1] != 3) throw format_error("FIF: the source space normals are not an n x 3 matrix"); break; }
      case tag::kBemSurfTriangles: tris_raw = ints32(f.value(t)); have_tris = true; break;
      case tag::kMneSourceSpaceTriangles: if (!have_tris) { tris_raw = ints32(f.value(t)); have_tris = true; } break;
      case tag::kMneSourceSpaceNuse: s.nuse = static_cast<std::int32_t>(one_int(f.value(t), "the used vertex count")); have_nuse = true; break;
      case tag::kMneSourceSpaceSelection: selection = ints32(f.value(t)); have_selection = true; break;
      case tag::kMneSourceSpaceNuseTri: nuse_tri = static_cast<std::int32_t>(one_int(f.value(t), "the used triangle count")); break;
      case tag::kMneSourceSpaceUseTriangles: use_tris_raw = ints32(f.value(t)); have_use_tris = true; break;
      case tag::kMneSourceSpaceNearest: nearest = ints32(f.value(t)); have_nearest = true; break;
      case tag::kMneSourceSpaceNearestDist: nearest_dist = reals_of(f.value(t)); have_nearest_dist = true; break;
      case tag::kMneSourceSpaceDist: s.dist = read_sparse(f, t); break;
      case tag::kMneSourceSpaceDistLimit: s.dist_limit = one_real(f.value(t), "the distance limit"); break;
      case tag::kSubjHisId: s.subject_his_id = f.value(t).text; break;
      default: s.other_tags.push_back(generic_of(f, t)); break;
    }
  }
  if (!have_type) throw format_error("FIF: unknown source space type (no type tag)");
  if (!have_npoints) throw format_error("FIF: the number of source space vertices is not found");
  if (!have_frame) throw format_error("FIF: the source space's coordinate frame is not found");
  if (!have_points) throw format_error("FIF: the source space's vertices are not found");
  if (!have_normals) throw format_error("FIF: the source space's vertex normals are not found");
  if (s.rr.size() != static_cast<std::size_t>(s.np) * 3) throw format_error("FIF: the vertex information is incorrect (" + std::to_string(s.rr.size() / 3) + " vertices for " + std::to_string(s.np) + ")");
  if (s.nn.size() != static_cast<std::size_t>(s.np) * 3) throw format_error("FIF: the vertex normal information is incorrect");
  s.ntri = ntri_tag.value_or(0);
  if (s.ntri > 0) {
    if (!have_tris) throw format_error("FIF: the triangulation is not found");
    if (tris_raw.size() != static_cast<std::size_t>(s.ntri) * 3) throw format_error("FIF: the triangulation information is incorrect");
    s.tris = tris_raw;
    for (auto& x : s.tris) --x;
  }
  if (have_nuse) {
    if (!have_selection) throw format_error("FIF: the source selection information is missing");
    if (selection.size() != static_cast<std::size_t>(s.np)) throw format_error("FIF: the number of entries in the source space selection is incorrect");
    s.inuse = selection;
    for (std::int32_t i = 0; i < s.np; ++i) if (s.inuse[static_cast<std::size_t>(i)]) s.vertno.push_back(i);
  }
  if (nuse_tri && have_use_tris) {
    s.nuse_tri = *nuse_tri;
    s.use_tris = use_tris_raw;
    for (auto& x : s.use_tris) --x;
  }
  if (have_nearest && have_nearest_dist) { s.nearest = nearest; s.nearest_dist = nearest_dist; }
  if (!nneigh.empty() && !neigh.empty()) {
    std::size_t off = 0;
    for (std::int32_t n : nneigh) {
      if (n < 0 || off + static_cast<std::size_t>(n) > neigh.size()) throw format_error("FIF: the neighbour counts exceed the neighbour list");
      s.neighbor_vert.emplace_back(neigh.begin() + static_cast<std::ptrdiff_t>(off), neigh.begin() + static_cast<std::ptrdiff_t>(off + static_cast<std::size_t>(n)));
      off += static_cast<std::size_t>(n);
    }
  }
  // volume source spaces: the parent MRI block holds the voxel transforms and the interpolator
  if (s.type == "vol") {
    const auto mris = find_blocks(b, block::kMneParentMriFile);
    if (mris.empty()) throw format_error("FIF: no parent MRI information in a volume source space (written by an MNE too old to store it)");
    for (const TagHeader& t : mris[0]->tags) {
      if (skip_tag(t)) continue;
      switch (t.kind) {
        case tag::kCoordTrans: {
          const CoordTrans c = detail::decode_coord_trans(f.bytes(t));
          if (c.from == mne::kCoordMriVoxel) s.vox_mri_t = c;
          else if (c.to == mne::kCoordRas) s.mri_ras_t = c;
          else { GenericTag g = generic_of(f, t); s.other_mri_tags.push_back(std::move(g)); }
          break;
        }
        case tag::kMneSourceSpaceInterpolator: {
          SparseMatrix m = read_sparse(f, t);
          if (!m.data.empty() || m.nrow * static_cast<std::int64_t>(m.ncol) != 0) { if (!m.data.empty()) s.interpolator = std::move(m); }
          break;
        }
        case tag::kMneSourceSpaceMriFile: s.mri_file = f.value(t).text; break;
        case tag::kMriWidth: s.mri_width = static_cast<std::int32_t>(one_int(f.value(t), "the MRI width")); break;
        case tag::kMriHeight: s.mri_height = static_cast<std::int32_t>(one_int(f.value(t), "the MRI height")); break;
        case tag::kMriDepth: s.mri_depth = static_cast<std::int32_t>(one_int(f.value(t), "the MRI depth")); break;
        case tag::kMneFileName: s.mri_volume_name = f.value(t).text; break;
        default: s.other_mri_tags.push_back(generic_of(f, t)); break;
      }
    }
  }
  return s;
}

void write_one_source_space(TagWriter& w, const SourceSpace& s) {
  std::int32_t ty = 0;
  if (s.type == "surf") ty = mne::kSpaceSurface;
  else if (s.type == "vol") ty = mne::kSpaceVolume;
  else if (s.type == "discrete") ty = mne::kSpaceDiscrete;
  else throw format_error("FIF: unknown source space type '" + s.type + "'");
  w.write_int(tag::kMneSourceSpaceType, ty);
  if (s.id >= 0) w.write_int(tag::kMneSourceSpaceId, s.id);
  if (s.subject_his_id && !s.subject_his_id->empty()) w.write_string(tag::kSubjHisId, *s.subject_his_id);
  w.write_int(tag::kMneCoordFrame, s.coord_frame);
  w.write_int(tag::kMneSourceSpaceNpoints, s.np);
  w.write_float_matrix(tag::kMneSourceSpacePoints, s.rr, s.np, 3);
  w.write_float_matrix(tag::kMneSourceSpaceNormals, s.nn, s.np, 3);
  w.write_int(tag::kMneSourceSpaceSelection, s.inuse);
  w.write_int(tag::kMneSourceSpaceNuse, s.nuse);
  if (s.ntri > 0) {
    w.write_int(tag::kMneSourceSpaceNtri, s.ntri);
    std::vector<std::int32_t> t1 = s.tris;
    for (auto& x : t1) ++x;
    w.write_int_matrix(tag::kMneSourceSpaceTriangles, t1, s.ntri, 3);
  }
  if (s.type != "vol" && !s.use_tris.empty()) {
    w.write_int(tag::kMneSourceSpaceNuseTri, s.nuse_tri);
    std::vector<std::int32_t> t1 = s.use_tris;
    for (auto& x : t1) ++x;
    w.write_int_matrix(tag::kMneSourceSpaceUseTriangles, t1, s.nuse_tri, 3);
  }
  if (s.type == "vol") {
    if (!s.neighbor_vert.empty()) {
      std::vector<std::int32_t> nn, all;
      for (const auto& n : s.neighbor_vert) { nn.push_back(static_cast<std::int32_t>(n.size())); all.insert(all.end(), n.begin(), n.end()); }
      w.write_int(tag::kMneSourceSpaceNneighbors, nn);
      w.write_int(tag::kMneSourceSpaceNeighbors, all);
    }
    if (s.src_mri_t) w.write_coord_trans(*s.src_mri_t);
    if (!s.shape.empty()) w.write_int(tag::kMneSourceSpaceVoxelDims, s.shape);
    w.start_block(block::kMneParentMriFile);
    if (s.mri_ras_t) w.write_coord_trans(*s.mri_ras_t);
    if (s.vox_mri_t) w.write_coord_trans(*s.vox_mri_t);
    if (s.mri_volume_name) w.write_string(tag::kMneFileName, *s.mri_volume_name);
    if (s.interpolator) w.write_sparse(tag::kMneSourceSpaceInterpolator, *s.interpolator);
    if (s.mri_file) w.write_string(tag::kMneSourceSpaceMriFile, *s.mri_file);
    if (s.mri_width) w.write_int(tag::kMriWidth, *s.mri_width);
    if (s.mri_height) w.write_int(tag::kMriHeight, *s.mri_height);
    if (s.mri_depth) w.write_int(tag::kMriDepth, *s.mri_depth);
    for (const GenericTag& g : s.other_mri_tags) w.write_generic(g);
    w.end_block(block::kMneParentMriFile);
  }
  if (!s.nearest.empty()) {
    w.write_int(tag::kMneSourceSpaceNearest, s.nearest);
    w.write_float_array(tag::kMneSourceSpaceNearestDist, s.nearest_dist);
  }
  if (s.dist) {
    w.write_sparse(tag::kMneSourceSpaceDist, *s.dist);
    w.write_float_array(tag::kMneSourceSpaceDistLimit, {s.dist_limit.value_or(0.0)});
  }
  if (s.type == "vol" && s.seg_name) w.write_string(tag::kComment, *s.seg_name);
  for (const GenericTag& g : s.other_tags) w.write_generic(g);
}

}  // namespace

SourceSpaces read_source_spaces(const File& f) {
  SourceSpaces out;
  const auto blocks = find_blocks(f.tree(), block::kMneSourceSpace);
  if (blocks.empty()) throw refused("FIF: no source spaces in the file");
  for (const Block* b : blocks) out.spaces.push_back(read_one_source_space(f, *b));
  read_env(f, f.tree(), &out.env_id, &out.working_dir, &out.command_line);
  return out;
}

void write_source_spaces(TagWriter& w, const SourceSpaces& s) {
  w.start_block(block::kMne);
  write_env(w, s.env_id, s.working_dir, s.command_line);
  for (const SourceSpace& sp : s.spaces) {
    w.start_block(block::kMneSourceSpace);
    write_one_source_space(w, sp);
    w.end_block(block::kMneSourceSpace);
  }
  w.end_block(block::kMne);
}

void write_source_spaces_file(ByteSink& out, const SourceSpaces& s, const Id& id) {
  write_file(out, id, [&](TagWriter& w) { write_source_spaces(w, s); });
}

// ---- covariance ---------------------------------------------------------------------------------------------------------------------------

namespace {

Covariance read_one_cov(const File& f, const Block& b) {
  Covariance c;
  bool have_dim = false, have_data = false;
  std::vector<double> lower;
  bool lower_given = false;
  for (const TagHeader& t : b.tags) {
    if (skip_tag(t)) continue;
    switch (t.kind) {
      case tag::kMneCovKind: c.kind = static_cast<std::int32_t>(one_int(f.value(t), "the covariance kind")); break;
      case tag::kMneCovDim: c.dim = static_cast<std::int32_t>(one_int(f.value(t), "the covariance dimension")); have_dim = true; break;
      case tag::kMneCovNfree: c.nfree = static_cast<std::int32_t>(one_int(f.value(t), "the degrees of freedom")); break;
      case tag::kMneCovMethod: c.method = f.value(t).text; break;
      case tag::kMneCovScore: c.score = one_real(f.value(t), "the covariance score"); break;
      case tag::kMneRowNames: c.names = detail::split_name_list(f.value(t).text); break;
      case tag::kMneCov:
        if (is_sparse(t)) throw refused("FIF: a sparse covariance matrix is not read");
        lower = reals_of(f.value(t));
        lower_given = true;
        break;
      case tag::kMneCovDiag:
        if (!lower_given) { c.data = reals_of(f.value(t)); c.diag = true; have_data = true; }
        break;
      case tag::kMneCovEigenvalues: c.eig = reals_of(f.value(t)); break;
      case tag::kMneCovEigenvectors: {
        const Value v = f.value(t);
        c.eigvec = reals_of(v);
        c.eigvec_rows = v.dims.size() > 0 ? v.dims[0] : 0;
        c.eigvec_cols = v.dims.size() > 1 ? v.dims[1] : 1;
        break;
      }
      case tag::kProjItemKind: break;
      default: c.other_tags.push_back(generic_of(f, t)); break;
    }
  }
  if (!have_dim) throw format_error("FIF: the covariance matrix dimension is not found");
  if (!c.names.empty() && static_cast<std::int32_t>(c.names.size()) != c.dim) throw format_error("FIF: the number of names does not match the covariance matrix dimension");
  if (lower_given) {
    const std::size_t d = static_cast<std::size_t>(c.dim);
    if (lower.size() != d * (d + 1) / 2) throw format_error("FIF: a packed covariance matrix of " + std::to_string(lower.size()) + " values for dimension " + std::to_string(c.dim));
    c.data.assign(d * d, 0.0);
    c.diag = false;
    std::size_t k = 0;
    for (std::size_t i = 0; i < d; ++i)
      for (std::size_t j = 0; j <= i; ++j) { c.data[i * d + j] = lower[k]; c.data[j * d + i] = lower[k]; ++k; }
    have_data = true;
  }
  if (!have_data) throw format_error("FIF: no covariance matrix data found");
  if (c.diag && c.data.size() != static_cast<std::size_t>(c.dim)) throw format_error("FIF: a diagonal covariance of the wrong length");
  c.projs = read_projectors(f, b);
  c.bads = read_bad_channels(f, b);
  return c;
}

}  // namespace

std::vector<Covariance> read_covariances(const File& f) {
  std::vector<Covariance> out;
  for (const Block* b : find_blocks(f.tree(), block::kMneCov)) out.push_back(read_one_cov(f, *b));
  return out;
}

Covariance read_covariance(const File& f, std::int32_t kind) {
  const auto all = read_covariances(f);
  if (all.empty()) throw refused("FIF: no covariance matrices in the file");
  for (const Covariance& c : all) if (c.kind == kind) return c;
  throw refused("FIF: no covariance matrix of kind " + std::to_string(kind) + " in the file");
}

void write_covariance(TagWriter& w, const Covariance& c) {
  w.start_block(block::kMneCov);
  w.write_int(tag::kMneCovKind, c.kind);
  w.write_int(tag::kMneCovDim, c.dim);
  if (c.nfree > 0) w.write_int(tag::kMneCovNfree, c.nfree);
  if (!c.names.empty()) w.write_name_list(tag::kMneRowNames, c.names);
  if (c.diag) {
    w.write_double(tag::kMneCovDiag, c.data);
  } else {
    std::vector<double> lower;
    const std::size_t d = static_cast<std::size_t>(c.dim);
    for (std::size_t i = 0; i < d; ++i) for (std::size_t j = 0; j <= i; ++j) lower.push_back(c.data[i * d + j]);
    w.write_double(tag::kMneCov, lower);
  }
  if (!c.eig.empty() && !c.eigvec.empty()) {
    w.write_float_matrix(tag::kMneCovEigenvectors, c.eigvec, c.eigvec_rows, c.eigvec_cols);
    w.write_double(tag::kMneCovEigenvalues, c.eig);
  }
  write_projectors(w, c.projs);
  write_bad_channels(w, c.bads);
  if (c.method) w.write_string(tag::kMneCovMethod, *c.method);
  if (c.score) w.write_double(tag::kMneCovScore, *c.score);
  for (const GenericTag& g : c.other_tags) w.write_generic(g);
  w.end_block(block::kMneCov);
}

void write_covariance_file(ByteSink& out, const Covariance& c, const Id& id) {
  write_file(out, id, [&](TagWriter& w) { write_covariance(w, c); });
}

// ---- forward and inverse ------------------------------------------------------------------------------------------------------------------

namespace {

ParentInfo read_parent(const File& f) {
  ParentInfo p;
  const auto mris = find_blocks(f.tree(), block::kMneParentMriFile);
  if (mris.empty()) throw format_error("FIF: no parent MRI information in the file");
  for (const TagHeader& t : mris[0]->tags) {
    if (skip_tag(t)) continue;
    switch (t.kind) {
      case tag::kMneFileName: p.mri_file = f.value(t).text; break;
      case tag::kParentFileId: p.mri_id = detail::decode_id(f.bytes(t)); break;
      case tag::kCoordTrans: {
        CoordTrans c = detail::decode_coord_trans(f.bytes(t));
        if (c.from == 5 && c.to == 4) p.mri_head_t = c;
        else if (c.from == 4 && c.to == 5) p.mri_head_t = detail::invert(c);
        else p.other_mri_tags.push_back(generic_of(f, t));
        break;
      }
      case tag::kMneCustomRef: case 236: p.custom_ref_applied = static_cast<std::int32_t>(one_int(f.value(t), "the custom reference flag")); break;
      default: p.other_mri_tags.push_back(generic_of(f, t)); break;
    }
  }
  if (!p.mri_head_t) throw format_error("FIF: the MRI/head coordinate transformation is not found");
  const auto megs = find_blocks(f.tree(), block::kMneParentMeasFile);
  if (megs.empty()) throw format_error("FIF: no parent MEG information in the file");
  const Block& m = *megs[0];
  for (const TagHeader& t : m.tags) {
    if (skip_tag(t)) { if (t.kind == tag::kParentBlockId && t.size == 20) p.meas_block_id = detail::decode_id(f.bytes(t)); continue; }
    switch (t.kind) {
      case tag::kMneFileName: p.meas_file = f.value(t).text; break;
      case tag::kParentFileId: p.meas_id = detail::decode_id(f.bytes(t)); break;
      case tag::kChInfo: p.channels.push_back(detail::decode_ch_info(f.bytes(t))); break;
      case tag::kNchan: break;
      case tag::kCoordTrans: {
        const CoordTrans c = detail::decode_coord_trans(f.bytes(t));
        if (c.from == 1 && c.to == 4) p.dev_head_t = c;
        else if (c.from == 1004 && c.to == 4) p.ctf_head_t = c;
        else p.other_meas_tags.push_back(generic_of(f, t));
        break;
      }
      default: p.other_meas_tags.push_back(generic_of(f, t)); break;
    }
  }
  apply_extended_channel_info(f, m, p.channels);
  for (ChannelInfo& ch : p.channels) ch.coord_frame = channel_coord_frame(ch.kind);
  p.bads = read_bad_channels(f, m);
  return p;
}

}  // namespace

ForwardFile read_forward(const File& f) {
  ForwardFile out;
  const auto fwds = find_blocks(f.tree(), block::kMneForwardSolution);
  if (fwds.empty()) throw refused("FIF: no forward solutions in the file");
  for (const Block* b : fwds) {
    ForwardSolution s;
    bool have_methods = false, have_ori = false, have_frame = false, have_nsource = false, have_nchan = false;
    for (const TagHeader& t : b->tags) {
      if (skip_tag(t)) continue;
      switch (t.kind) {
        case tag::kMneIncludedMethods: s.methods = static_cast<std::int32_t>(one_int(f.value(t), "the included methods")); have_methods = true; break;
        case tag::kMneSourceOrientation: s.source_ori = static_cast<std::int32_t>(one_int(f.value(t), "the source orientation")); have_ori = true; break;
        case tag::kMneCoordFrame: s.coord_frame = static_cast<std::int32_t>(one_int(f.value(t), "the coordinate frame")); have_frame = true; break;
        case tag::kMneSourceSpaceNpoints: s.nsource = static_cast<std::int32_t>(one_int(f.value(t), "the source count")); have_nsource = true; break;
        case tag::kNchan: s.nchan = static_cast<std::int32_t>(one_int(f.value(t), "the channel count")); have_nchan = true; break;
        default: s.other_tags.push_back(generic_of(f, t)); break;
      }
    }
    if (!have_methods) throw format_error("FIF: methods not listed for one of the forward solutions");
    if (!have_ori) throw format_error("FIF: the source orientation of a forward solution is not found");
    if (!have_frame) throw format_error("FIF: the coordinate frame of a forward solution is not found");
    if (!have_nsource) throw format_error("FIF: the number of sources of a forward solution is not found");
    if (!have_nchan) throw format_error("FIF: the number of channels of a forward solution is not found");
    bool found = false;
    s.sol = read_named(f, *b, tag::kMneForwardSolution, &found);
    if (!found) throw format_error("FIF: the forward solution data are not found");
    bool gfound = false;
    NamedMatrix g = read_named(f, *b, tag::kMneForwardSolutionGrad, &gfound);
    if (gfound) s.sol_grad = std::move(g);
    // the stored matrix is sources x channels (MNE transposes it): the channel count is its columns
    if (s.sol.ncol != s.nchan || (s.sol.nrow != s.nsource && s.sol.nrow != 3 * s.nsource)) throw format_error("FIF: the forward solution matrix has the wrong dimensions");
    out.solutions.push_back(std::move(s));
  }
  out.src = read_source_spaces(f);
  out.parent = read_parent(f);
  read_env(f, f.tree(), &out.src.env_id, &out.working_dir, &out.command_line);
  return out;
}

InverseOperator read_inverse(const File& f) {
  InverseOperator inv;
  const auto invs = find_blocks(f.tree(), block::kMneInverseSolution);
  if (invs.empty()) throw refused("FIF: no inverse solutions in the file");
  const Block& b = *invs[0];
  bool have_methods = false, have_ori = false, have_nsource = false, have_frame = false, have_nn = false, have_sing = false;
  for (const TagHeader& t : b.tags) {
    if (skip_tag(t)) continue;
    switch (t.kind) {
      case tag::kMneIncludedMethods: inv.methods = static_cast<std::int32_t>(one_int(f.value(t), "the included methods")); have_methods = true; break;
      case tag::kMneSourceOrientation: inv.source_ori = static_cast<std::int32_t>(one_int(f.value(t), "the source orientation constraints")); have_ori = true; break;
      case tag::kMneSourceSpaceNpoints: inv.nsource = static_cast<std::int32_t>(one_int(f.value(t), "the number of sources")); have_nsource = true; break;
      case tag::kMneCoordFrame: inv.coord_frame = static_cast<std::int32_t>(one_int(f.value(t), "the coordinate frame")); have_frame = true; break;
      case tag::kMneInverseSourceUnit: inv.units = static_cast<std::int32_t>(one_int(f.value(t), "the source unit")); break;
      case tag::kMneInverseSourceOrientations: {
        const Value v = f.value(t);
        inv.source_nn = reals_of(v);
        inv.source_nn_rows = v.dims.empty() ? 0 : v.dims[0];
        have_nn = true;
        break;
      }
      case tag::kMneInverseSing: inv.sing = reals_of(f.value(t)); have_sing = true; break;
      case tag::kNchan: inv.nchan = static_cast<std::int32_t>(one_int(f.value(t), "the channel count")); break;
      default: inv.other_tags.push_back(generic_of(f, t)); break;
    }
  }
  if (!have_methods) throw format_error("FIF: modalities not found in the inverse operator");
  if (!have_ori) throw format_error("FIF: source orientation constraints not found");
  if (!have_nsource) throw format_error("FIF: the number of sources not found");
  if (!have_frame) throw format_error("FIF: the coordinate frame tag not found");
  if (!have_nn) throw format_error("FIF: the source orientation information not found");
  if (!have_sing) throw format_error("FIF: the singular values not found");
  bool found = false;
  inv.eigen_leads = read_named(f, b, tag::kMneInverseLeads, &found);
  if (!found) {
    inv.eigen_leads_weighted = true;
    inv.eigen_leads = read_named(f, b, tag::kMneInverseLeadsWeighted, &found);
  }
  if (!found) throw format_error("FIF: the eigenleads are not found in the inverse operator");
  inv.eigen_fields = read_named(f, b, tag::kMneInverseFields, &found);
  if (!found) throw format_error("FIF: the eigenfields are not found in the inverse operator");
  for (const Block& c : b.children) {
    if (c.kind != block::kMneCov) continue;
    Covariance cov = read_one_cov(f, c);
    switch (cov.kind) {
      case mne::kNoiseCov: inv.noise_cov = std::move(cov); break;
      case mne::kSourceCov: inv.source_cov = std::move(cov); break;
      case mne::kOrientPriorCov: inv.orient_prior = std::move(cov); break;
      case mne::kDepthPriorCov: inv.depth_prior = std::move(cov); break;
      case mne::kFmriPriorCov: inv.fmri_prior = std::move(cov); break;
      default: break;
    }
  }
  inv.projs = read_projectors(f, f.tree());
  inv.src = read_source_spaces(f);
  inv.parent = read_parent(f);
  read_env(f, f.tree(), &inv.src.env_id, &inv.src.working_dir, &inv.src.command_line);
  return inv;
}

// ---- events, annotations, transforms ------------------------------------------------------------------------------------------------------

EventFile read_events(const File& f) {
  const auto blocks = find_blocks(f.tree(), block::kMneEvents);
  if (blocks.empty()) throw refused("FIF: could not find event data in the file");
  const Block& b = *blocks[0];
  EventFile e;
  bool have = false;
  const TagHeader* desc = nullptr;
  const TagHeader* comments = nullptr;
  for (const TagHeader& t : b.tags) {
    if (skip_tag(t)) continue;
    if (t.kind == tag::kMneEventList && !have) {
      const std::vector<std::int32_t> v = ints32(f.value(t));
      if (v.size() % 3 != 0) throw format_error("FIF: the event list has " + std::to_string(v.size()) + " values, not a multiple of 3");
      for (std::size_t i = 0; i + 2 < v.size(); i += 3) e.events.push_back({v[i], v[i + 1], v[i + 2]});
      have = true;
    } else if (t.kind == tag::kDescription && !desc) {
      desc = &t;
    } else if (t.kind == tag::kMneEventComments && !comments) {
      comments = &t;
    } else {
      e.other_tags.push_back(generic_of(f, t));
    }
  }
  if (!have) throw format_error("FIF: could not find any events");
  if (desc) {
    const std::string s = f.value(*desc).text;
    std::size_t st = 0;
    while (st <= s.size()) {
      const std::size_t semi = s.find(';', st);
      const std::string part = s.substr(st, semi == std::string::npos ? std::string::npos : semi - st);
      const std::size_t colon = part.rfind(':');   // MNE splits at the last colon: names may contain colons
      if (colon == std::string::npos) throw format_error("FIF: an event description without a value: '" + part + "'");
      e.event_id.emplace_back(part.substr(0, colon), std::stoi(part.substr(colon + 1)));
      if (semi == std::string::npos) break;
      st = semi + 1;
    }
  } else if (comments) {
    const std::vector<unsigned char> raw = f.bytes(*comments);
    std::vector<std::string> names;
    std::string cur;
    for (unsigned char c : raw) { if (c == 0) { names.push_back(cur); cur.clear(); } else cur += static_cast<char>(c); }
    if (names.size() != e.events.size()) throw format_error("FIF: the number of event comments does not match the number of events");
    for (std::size_t i = 0; i < names.size(); ++i) e.event_id.emplace_back(names[i], e.events[i][2]);
  }
  return e;
}

void write_events(TagWriter& w, const EventFile& e) {
  w.start_block(block::kMneEvents);
  std::vector<std::int32_t> flat;
  for (const auto& ev : e.events) flat.insert(flat.end(), ev.begin(), ev.end());
  w.write_int(tag::kMneEventList, flat);
  if (!e.event_id.empty()) {
    std::string s;
    for (std::size_t i = 0; i < e.event_id.size(); ++i) { if (i) s += ';'; s += e.event_id[i].first + ":" + std::to_string(e.event_id[i].second); }
    w.write_string(tag::kDescription, s);
  }
  for (const GenericTag& g : e.other_tags) w.write_generic(g);
  w.end_block(block::kMneEvents);
}

void write_events_file(ByteSink& out, const EventFile& e, const Id& id) {
  write_file(out, id, [&](TagWriter& w) { write_events(w, e); });
}

AnnotationSet read_annotations_file(const File& f) {
  auto a = read_annotations(f, f.tree());
  if (!a) throw refused("FIF: no annotations in the file");
  return *a;
}

void write_annotations(TagWriter& w, const AnnotationSet& a) {
  w.start_block(block::kMneAnnotations);
  std::vector<double> onset, end;
  std::vector<std::string> desc;
  nlohmann::ordered_json drop = nlohmann::ordered_json::array(), extras = nlohmann::ordered_json::array();
  bool any_ch = false, any_extra = false;
  for (const AnnotationRecord& r : a.items) {
    onset.push_back(r.onset);
    end.push_back(r.onset + r.duration);
    desc.push_back(r.description);
    nlohmann::ordered_json ch = nlohmann::ordered_json::array();
    for (const std::string& c : r.channels) ch.push_back(c);
    any_ch = any_ch || !r.channels.empty();
    drop.push_back(ch);
    if (r.extras_json.empty()) extras.push_back(nlohmann::ordered_json::object());
    else { extras.push_back(nlohmann::ordered_json::parse(r.extras_json)); any_extra = true; }
  }
  w.write_float(tag::kMneBaselineMin, onset);
  w.write_float(tag::kMneBaselineMax, end);
  w.write_name_list(tag::kComment, desc);
  if (a.orig_time) {
    const double secs = std::floor(*a.orig_time);
    w.write_double(tag::kMeasDate, std::vector<double>{secs, std::round((*a.orig_time - secs) * 1e6)});
  }
  if (any_ch) w.write_string(tag::kMneEpochsDropLog, py_dumps(drop));
  if (any_extra) w.write_string(tag::kFreeList, py_dumps(extras));
  w.end_block(block::kMneAnnotations);
}

void write_annotations_file(ByteSink& out, const AnnotationSet& a, const Id& id) {
  write_file(out, id, [&](TagWriter& w) { write_annotations(w, a); });
}

CoordTrans read_transform(const File& f) {
  for (const TagHeader& t : f.directory())
    if (t.kind == tag::kCoordTrans) return detail::decode_coord_trans(f.bytes(t));
  throw refused("FIF: no coordinate transform in the file");
}

void write_transform_file(ByteSink& out, const CoordTrans& t, const Id& id) {
  write_file(out, id, [&](TagWriter& w) { w.write_coord_trans(t); });
}

}  // namespace nxr::formats::fif
