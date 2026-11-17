// CIFTI-2: the XML of a NIfTI-2 extension, and the matrix by rows and columns. Written from the CIFTI-2 specification (Glasser et al. 2013,
// NeuroImage; the format description at https://www.nitrc.org/projects/cifti), with pugixml; nibabel (MIT) is the reference the goldens come from.
#include "nxr/formats/cifti.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "nxr/formats/endian.h"
#include "nxr/formats/errors.h"
#include "pugixml.hpp"

namespace nxr::formats::cifti {

namespace {

std::vector<std::string> split_ws(const char* text) {
  std::vector<std::string> out;
  std::istringstream is(text ? text : "");
  std::string w;
  while (is >> w) out.push_back(w);
  return out;
}

std::int64_t to_int(const std::string& s, const char* what) {
  char* end = nullptr;
  const long long v = std::strtoll(s.c_str(), &end, 10);
  if (s.empty() || *end != '\0') throw format_error(std::string("CIFTI: '") + s + "' in " + what + " is not an integer");
  return v;
}
double to_real(const std::string& s, const char* what) {
  char* end = nullptr;
  const double v = std::strtod(s.c_str(), &end);
  if (s.empty() || *end != '\0') throw format_error(std::string("CIFTI: '") + s + "' in " + what + " is not a number");
  return v;
}

std::vector<std::pair<std::string, std::string>> read_metadata(const pugi::xml_node& parent) {
  std::vector<std::pair<std::string, std::string>> md;
  const pugi::xml_node m = parent.child("MetaData");
  for (pugi::xml_node e : m.children("MD")) md.emplace_back(e.child("Name").text().get(), e.child("Value").text().get());
  return md;
}

std::vector<std::array<std::int32_t, 3>> read_voxels(const pugi::xml_node& n, const char* what) {
  std::vector<std::array<std::int32_t, 3>> out;
  const auto w = split_ws(n.text().get());
  if (w.size() % 3 != 0) throw format_error(std::string("CIFTI: the ") + what + " voxel indices are not a multiple of three numbers");
  for (std::size_t i = 0; i < w.size(); i += 3) out.push_back({static_cast<std::int32_t>(to_int(w[i], what)), static_cast<std::int32_t>(to_int(w[i + 1], what)), static_cast<std::int32_t>(to_int(w[i + 2], what))});
  return out;
}

std::vector<std::uint32_t> read_vertices(const pugi::xml_node& n, const char* what) {
  std::vector<std::uint32_t> out;
  for (const std::string& s : split_ws(n.text().get())) {
    const std::int64_t v = to_int(s, what);
    if (v < 0) throw format_error(std::string("CIFTI: a negative vertex index in ") + what);
    out.push_back(static_cast<std::uint32_t>(v));
  }
  return out;
}

}  // namespace

std::uint64_t Mapping::length() const {
  if (type == "CIFTI_INDEX_TYPE_BRAIN_MODELS") {
    std::uint64_t n = 0;
    for (const BrainModel& b : brain_models) n = std::max(n, b.index_offset + b.index_count);
    return n;
  }
  if (type == "CIFTI_INDEX_TYPE_PARCELS") return parcels.size();
  if (type == "CIFTI_INDEX_TYPE_SERIES") return series ? series->points : 0;
  return named_maps.size();   // scalars, labels
}

void parse_xml(const std::string& xml, std::string* version, std::vector<std::pair<std::string, std::string>>* metadata, std::vector<Mapping>* mappings) {
  pugi::xml_document doc;
  const pugi::xml_parse_result res = doc.load_buffer(xml.data(), xml.size());
  if (!res) throw format_error(std::string("CIFTI: the XML extension does not parse: ") + res.description());
  const pugi::xml_node root = doc.child("CIFTI");
  if (!root) throw format_error("CIFTI: the XML extension has no CIFTI element");
  *version = root.attribute("Version").as_string();
  if (*version != "2" && *version != "2.0") throw refused("CIFTI: version '" + *version + "' is not read (only CIFTI-2)");
  const pugi::xml_node matrix = root.child("Matrix");
  if (!matrix) throw format_error("CIFTI: the XML has no Matrix element");
  *metadata = read_metadata(matrix);
  for (pugi::xml_node mm : matrix.children("MatrixIndicesMap")) {
    Mapping m;
    for (const std::string& d : [&] { std::vector<std::string> v; std::string s = mm.attribute("AppliesToMatrixDimension").as_string(), cur; std::istringstream is(s); while (std::getline(is, cur, ',')) v.push_back(cur); return v; }())
      m.applies_to.push_back(static_cast<int>(to_int(d, "AppliesToMatrixDimension")));
    m.type = mm.attribute("IndicesMapToDataType").as_string();
    if (m.type.empty()) throw format_error("CIFTI: a MatrixIndicesMap without IndicesMapToDataType");
    if (m.type != "CIFTI_INDEX_TYPE_BRAIN_MODELS" && m.type != "CIFTI_INDEX_TYPE_PARCELS" && m.type != "CIFTI_INDEX_TYPE_SCALARS" && m.type != "CIFTI_INDEX_TYPE_LABELS" &&
        m.type != "CIFTI_INDEX_TYPE_SERIES")
      throw refused("CIFTI: the index type '" + m.type + "' is not read");
    if (m.type == "CIFTI_INDEX_TYPE_SERIES") {
      Series s;
      s.points = static_cast<std::uint64_t>(to_int(mm.attribute("NumberOfSeriesPoints").as_string(), "NumberOfSeriesPoints"));
      s.start = to_real(mm.attribute("SeriesStart").as_string("0"), "SeriesStart");
      s.step = to_real(mm.attribute("SeriesStep").as_string("0"), "SeriesStep");
      s.exponent = static_cast<int>(to_int(mm.attribute("SeriesExponent").as_string("0"), "SeriesExponent"));
      s.unit = mm.attribute("SeriesUnit").as_string();
      m.series = s;
    }
    for (pugi::xml_node nm : mm.children("NamedMap")) {
      NamedMap n;
      n.name = nm.child("MapName").text().get();
      n.metadata = read_metadata(nm);
      for (pugi::xml_node lab : nm.child("LabelTable").children("Label")) {
        LabelEntry e;
        e.key = static_cast<std::int32_t>(lab.attribute("Key").as_llong());
        e.name = lab.text().get();
        e.red = lab.attribute("Red").as_float();
        e.green = lab.attribute("Green").as_float();
        e.blue = lab.attribute("Blue").as_float();
        e.alpha = lab.attribute("Alpha").as_float();
        n.labels.push_back(std::move(e));
      }
      m.named_maps.push_back(std::move(n));
    }
    for (pugi::xml_node bm : mm.children("BrainModel")) {
      BrainModel b;
      b.model_type = bm.attribute("ModelType").as_string();
      b.structure = bm.attribute("BrainStructure").as_string();
      b.index_offset = static_cast<std::uint64_t>(bm.attribute("IndexOffset").as_ullong());
      b.index_count = static_cast<std::uint64_t>(bm.attribute("IndexCount").as_ullong());
      if (bm.attribute("SurfaceNumberOfVertices")) b.surface_vertex_count = static_cast<std::uint64_t>(bm.attribute("SurfaceNumberOfVertices").as_ullong());
      if (b.model_type != "CIFTI_MODEL_TYPE_SURFACE" && b.model_type != "CIFTI_MODEL_TYPE_VOXELS") throw format_error("CIFTI: the brain model type '" + b.model_type + "' is not a CIFTI-2 type");
      if (b.model_type == "CIFTI_MODEL_TYPE_SURFACE") {
        b.vertex_indices = read_vertices(bm.child("VertexIndices"), "VertexIndices");
        if (b.vertex_indices.size() != b.index_count) throw format_error("CIFTI: a surface brain model of " + std::to_string(b.index_count) + " indices lists " + std::to_string(b.vertex_indices.size()) + " vertices");
      } else {
        b.voxel_ijk = read_voxels(bm.child("VoxelIndicesIJK"), "VoxelIndicesIJK");
        if (b.voxel_ijk.size() != b.index_count) throw format_error("CIFTI: a voxel brain model of " + std::to_string(b.index_count) + " indices lists " + std::to_string(b.voxel_ijk.size()) + " voxels");
      }
      m.brain_models.push_back(std::move(b));
    }
    for (pugi::xml_node sf : mm.children("Surface")) {
      SurfaceInfo s;
      s.structure = sf.attribute("BrainStructure").as_string();
      s.vertex_count = static_cast<std::uint64_t>(sf.attribute("SurfaceNumberOfVertices").as_ullong());
      m.surfaces.push_back(std::move(s));
    }
    for (pugi::xml_node pc : mm.children("Parcel")) {
      Parcel p;
      p.name = pc.attribute("Name").as_string();
      for (pugi::xml_node v : pc.children("Vertices")) p.vertices.push_back({v.attribute("BrainStructure").as_string(), read_vertices(v, "a parcel's Vertices")});
      if (pc.child("VoxelIndicesIJK")) p.voxel_ijk = read_voxels(pc.child("VoxelIndicesIJK"), "a parcel's VoxelIndicesIJK");
      m.parcels.push_back(std::move(p));
    }
    if (const pugi::xml_node vol = mm.child("Volume")) {
      VolumeSpace v;
      std::string dims = vol.attribute("VolumeDimensions").as_string();
      for (char& c : dims) if (c == ',') c = ' ';
      const auto dd = split_ws(dims.c_str());
      if (dd.size() != 3) throw format_error("CIFTI: VolumeDimensions needs three numbers");
      for (int i = 0; i < 3; ++i) v.dims[static_cast<std::size_t>(i)] = to_int(dd[static_cast<std::size_t>(i)], "VolumeDimensions");
      const pugi::xml_node tm = vol.child("TransformationMatrixVoxelIndicesIJKtoXYZ");
      if (!tm) throw format_error("CIFTI: a Volume without its TransformationMatrixVoxelIndicesIJKtoXYZ");
      v.meter_exponent = static_cast<int>(to_int(tm.attribute("MeterExponent").as_string("-3"), "MeterExponent"));
      const auto w = split_ws(tm.text().get());
      if (w.size() != 16) throw format_error("CIFTI: the voxel-to-world matrix has " + std::to_string(w.size()) + " numbers, not 16");
      for (std::size_t i = 0; i < 16; ++i) v.ijk_to_xyz[i] = to_real(w[i], "the voxel-to-world matrix");
      m.volume = v;
    }
    mappings->push_back(std::move(m));
  }
  if (mappings->empty()) throw format_error("CIFTI: the XML has no MatrixIndicesMap");
}

const Mapping& Reader::map(int dimension) const {
  for (const Mapping& m : mappings_)
    if (std::find(m.applies_to.begin(), m.applies_to.end(), dimension) != m.applies_to.end()) return m;
  throw range_error("CIFTI: no mapping for matrix dimension " + std::to_string(dimension));
}

std::unique_ptr<Reader> open(std::shared_ptr<const ByteSource> file) {
  auto rd = std::unique_ptr<Reader>(new Reader());
  Reader& r = *rd;
  r.src_ = file;
  r.nifti_ = nifti::read_header(*file);
  const nifti::Header& h = r.nifti_;
  if (h.version != 2) throw refused("CIFTI: a NIfTI-" + std::to_string(h.version) + " file is not a CIFTI-2 file (CIFTI-2 is NIfTI-2)");
  const nifti::Extension* ext = nullptr;
  for (const nifti::Extension& e : h.extensions) if (e.code == 32) { ext = &e; break; }
  if (!ext) throw format_error("CIFTI: the file has no extension of code 32 (not a CIFTI file)");
  std::string xml(ext->bytes.begin(), ext->bytes.end());
  while (!xml.empty() && xml.back() == '\0') xml.pop_back();
  r.xml_ = xml;
  parse_xml(xml, &r.version_, &r.metadata_, &r.mappings_);
  if (h.shape.size() != 6 || h.shape[0] != 1 || h.shape[1] != 1 || h.shape[2] != 1 || h.shape[3] != 1)
    throw refused("CIFTI: the data are not a two-dimensional matrix (NIfTI dim of " + std::to_string(h.shape.size()) + " entries)");
  r.rows_ = static_cast<std::uint64_t>(h.shape[4]);   // matrix dimension 0 is NIfTI dim[5], the fastest in the file
  r.cols_ = static_cast<std::uint64_t>(h.shape[5]);
  if (h.datatype == 16) r.precision_ = Precision::float32;
  else if (h.datatype == 64) r.precision_ = Precision::float64;
  else throw refused("CIFTI: data of NIfTI datatype " + std::to_string(h.datatype) + " are not read (float32 and float64 are)");
  for (int d = 0; d < 2; ++d) {
    const std::uint64_t want = d == 0 ? r.rows_ : r.cols_;
    const Mapping& m = r.map(d);
    if (m.length() != want) throw format_error("CIFTI: the mapping of dimension " + std::to_string(d) + " covers " + std::to_string(m.length()) + " indices, the data have " + std::to_string(want));
  }
  const std::uint64_t es = r.precision_ == Precision::float32 ? 4 : 8;
  if (h.vox_offset + r.rows_ * r.cols_ * es > file->size())
    throw format_error("CIFTI: the file is truncated: " + std::to_string(r.rows_) + " x " + std::to_string(r.cols_) + " values need " + std::to_string(h.vox_offset + r.rows_ * r.cols_ * es) + " bytes, the file has " + std::to_string(file->size()));
  return rd;
}

std::unique_ptr<Reader> open_file(const std::string& path) {
  if (path.size() > 3 && path.compare(path.size() - 3, 3, ".gz") == 0) throw refused("CIFTI: a gzipped file is not read (CIFTI files are used uncompressed; their data are large)");
  return open(std::make_shared<FileSource>(path));
}

template <class T>
void Reader::read_values(std::uint64_t r0, std::uint64_t n, std::uint64_t c0, std::uint64_t c1, T* out) const {
  if (c0 > c1 || c1 > cols_) throw range_error("columns [" + std::to_string(c0) + ", " + std::to_string(c1) + ") of " + std::to_string(cols_));
  if (r0 > rows_ || n > rows_ - r0) throw range_error("rows [" + std::to_string(r0) + ", " + std::to_string(r0) + " + " + std::to_string(n) + ") of " + std::to_string(rows_));
  const std::uint64_t es = precision_ == Precision::float32 ? 4 : 8, w = c1 - c0;
  std::vector<unsigned char> buf(static_cast<std::size_t>(n * es));
  for (std::uint64_t k = 0; k < w; ++k) {   // one contiguous run of rows per column (dimension 0 varies fastest in the file)
    if (n) src_->read(nifti_.vox_offset + ((c0 + k) * rows_ + r0) * es, buf.size(), buf.data());
    for (std::uint64_t i = 0; i < n; ++i)
      out[i * w + k] = static_cast<T>(es == 4 ? load_f32(&buf[static_cast<std::size_t>(i * 4)], nifti_.little_endian) : load_f64(&buf[static_cast<std::size_t>(i * 8)], nifti_.little_endian));
  }
}

void Reader::read(std::uint64_t r0, std::uint64_t n, std::uint64_t c0, std::uint64_t c1, float* out) const {
  if (precision_ != Precision::float32) throw refused("this CIFTI file holds 64-bit values: use the double read");
  read_values(r0, n, c0, c1, out);
}
void Reader::read(std::uint64_t r0, std::uint64_t n, std::uint64_t c0, std::uint64_t c1, double* out) const { read_values(r0, n, c0, c1, out); }

}  // namespace nxr::formats::cifti
