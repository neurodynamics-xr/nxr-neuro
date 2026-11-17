#include "nxr/formats/gifti.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

#include "nxr/formats/deflate.h"
#include "nxr/formats/endian.h"
#include "nxr/formats/errors.h"
#include "pugixml.hpp"

namespace nxr::formats::gifti {

namespace {

const char* kB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::vector<unsigned char> b64_decode(const char* s) {
  static int table[256];
  static bool init = false;
  if (!init) {
    for (int& t : table) t = -1;
    for (int i = 0; i < 64; ++i) table[static_cast<unsigned char>(kB64[i])] = i;
    init = true;
  }
  std::vector<unsigned char> out;
  unsigned int acc = 0;
  int bits = 0;
  for (; *s; ++s) {
    const unsigned char c = static_cast<unsigned char>(*s);
    if (c == '=') break;
    if (c == ' ' || c == '\n' || c == '\r' || c == '\t') continue;
    const int v = table[c];
    if (v < 0) throw format_error("GIFTI: a character that is not base64 in a Data element");
    acc = (acc << 6) | static_cast<unsigned int>(v);
    bits += 6;
    if (bits >= 8) { bits -= 8; out.push_back(static_cast<unsigned char>((acc >> bits) & 0xFF)); }
  }
  return out;
}

std::string b64_encode(const unsigned char* p, std::size_t n) {
  std::string out;
  out.reserve((n + 2) / 3 * 4);
  for (std::size_t i = 0; i < n; i += 3) {
    const unsigned int a = p[i], b = i + 1 < n ? p[i + 1] : 0, c = i + 2 < n ? p[i + 2] : 0;
    const unsigned int v = (a << 16) | (b << 8) | c;
    out.push_back(kB64[(v >> 18) & 63]);
    out.push_back(kB64[(v >> 12) & 63]);
    out.push_back(i + 1 < n ? kB64[(v >> 6) & 63] : '=');
    out.push_back(i + 2 < n ? kB64[v & 63] : '=');
  }
  return out;
}

int element_bytes(const std::string& dt) {
  if (dt == "NIFTI_TYPE_FLOAT32" || dt == "NIFTI_TYPE_INT32") return 4;
  if (dt == "NIFTI_TYPE_UINT8") return 1;
  throw refused("GIFTI: the datatype " + dt + " is not read (GIFTI defines NIFTI_TYPE_FLOAT32, NIFTI_TYPE_INT32 and NIFTI_TYPE_UINT8)");
}

double parse_double(const char*& p, const char* what) {
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(p, &end);
  if (end == p) throw format_error(std::string("GIFTI: a number was expected in ") + what);
  p = end;
  return v;
}

std::vector<std::pair<std::string, std::string>> read_meta(const pugi::xml_node& parent) {
  std::vector<std::pair<std::string, std::string>> m;
  for (pugi::xml_node md : parent.child("MetaData").children("MD"))
    m.emplace_back(md.child_value("Name"), md.child_value("Value"));
  return m;
}

void permute_column_to_row(std::vector<unsigned char>& raw, const std::vector<std::int64_t>& dims, int eb) {
  // `raw` holds the elements with the FIRST index fastest; make it last-index-fastest.
  const std::size_t nd = dims.size();
  std::uint64_t n = 1;
  for (auto d : dims) n *= static_cast<std::uint64_t>(d);
  std::vector<unsigned char> out(raw.size());
  std::vector<std::int64_t> idx(nd, 0);
  for (std::uint64_t k = 0; k < n; ++k) {   // k runs in the file's (first index fastest) order
    std::uint64_t dst = 0;
    for (std::size_t a = 0; a < nd; ++a) dst = dst * static_cast<std::uint64_t>(dims[a]) + static_cast<std::uint64_t>(idx[a]);
    std::memcpy(out.data() + dst * eb, raw.data() + k * eb, eb);
    for (std::size_t a = 0; a < nd; ++a) { if (++idx[a] < dims[a]) break; idx[a] = 0; }
  }
  raw.swap(out);
}

}  // namespace

std::uint64_t DataArray::count() const {
  std::uint64_t n = 1;
  for (std::int64_t d : dims) {
    if (d < 0) throw format_error("GIFTI: a negative dimension");
    if (d != 0 && n > UINT64_MAX / static_cast<std::uint64_t>(d)) throw format_error("GIFTI: the dimensions overflow");
    n *= static_cast<std::uint64_t>(d);
  }
  return n;
}

double DataArray::at(std::size_t i) const {
  if (!f32.empty()) return f32[i];
  if (!i32.empty()) return i32[i];
  return u8[i];
}

const std::string* DataArray::meta(const std::string& name) const {
  for (const auto& kv : metadata) if (kv.first == name) return &kv.second;
  return nullptr;
}
const std::string* Image::meta(const std::string& name) const {
  for (const auto& kv : metadata) if (kv.first == name) return &kv.second;
  return nullptr;
}

Image read(const ByteSource& src) {
  const std::vector<unsigned char> xml = src.bytes(0, static_cast<std::size_t>(src.size()));
  pugi::xml_document doc;
  const pugi::xml_parse_result res = doc.load_buffer(xml.data(), xml.size(), pugi::parse_default | pugi::parse_declaration | pugi::parse_doctype | pugi::parse_ws_pcdata_single);
  if (!res) throw format_error(std::string("GIFTI: the XML does not parse: ") + res.description());
  const pugi::xml_node root = doc.child("GIFTI");
  if (!root) throw format_error("GIFTI: the root element is not <GIFTI>");

  Image img;
  img.version = root.attribute("Version").as_string("1.0");
  img.metadata = read_meta(root);
  for (pugi::xml_node l : root.child("LabelTable").children("Label")) {
    Label lab;
    lab.key = l.attribute("Key").as_int();
    lab.name = l.text().as_string();
    if (l.attribute("Red") || l.attribute("Green") || l.attribute("Blue") || l.attribute("Alpha")) {
      lab.has_colour = true;
      lab.rgba = {l.attribute("Red").as_float(), l.attribute("Green").as_float(), l.attribute("Blue").as_float(), l.attribute("Alpha").as_float(1.0f)};
    }
    img.labels.push_back(std::move(lab));
  }
  const int declared = root.attribute("NumberOfDataArrays").as_int(-1);

  for (pugi::xml_node da : root.children("DataArray")) {
    DataArray a;
    a.intent = da.attribute("Intent").as_string();
    a.datatype = da.attribute("DataType").as_string();
    const int eb = element_bytes(a.datatype);
    const std::string order = da.attribute("ArrayIndexingOrder").as_string("RowMajorOrder");
    if (order == "ColumnMajorOrder") a.column_major_in_file = true;
    else if (order != "RowMajorOrder") throw format_error("GIFTI: ArrayIndexingOrder \"" + order + "\"");
    const std::string endian = da.attribute("Endian").as_string("LittleEndian");
    if (endian == "BigEndian") a.big_endian_in_file = true;
    else if (endian != "LittleEndian") throw format_error("GIFTI: Endian \"" + endian + "\"");
    const int nd = da.attribute("Dimensionality").as_int(-1);
    if (nd < 0 || nd > 16) throw format_error("GIFTI: Dimensionality " + std::to_string(nd) + " is missing or out of range");
    for (int i = 0; i < nd; ++i) {
      const pugi::xml_attribute at = da.attribute(("Dim" + std::to_string(i)).c_str());
      if (!at) throw format_error("GIFTI: Dim" + std::to_string(i) + " is missing");
      a.dims.push_back(at.as_llong());
    }
    a.metadata = read_meta(da);
    for (pugi::xml_node t : da.children("CoordinateSystemTransformMatrix")) {
      Transform tr;
      tr.dataspace = t.child_value("DataSpace");
      tr.xformspace = t.child_value("TransformedSpace");
      const char* p = t.child_value("MatrixData");
      for (int i = 0; i < 16; ++i) tr.matrix[i] = parse_double(p, "MatrixData");
      a.transforms.push_back(tr);
    }
    const std::string enc = da.attribute("Encoding").as_string();
    const std::uint64_t n = a.count();
    if (n > (std::uint64_t(1) << 40) / static_cast<std::uint64_t>(eb)) throw format_error("GIFTI: an array of " + std::to_string(n) + " values is not credible");
    const char* text = da.child_value("Data");
    std::vector<unsigned char> raw;
    bool little = !a.big_endian_in_file;
    if (enc == "ASCII") {
      a.encoding = Encoding::ascii;
      raw.resize(static_cast<std::size_t>(n) * eb);
      const char* p = text;
      for (std::uint64_t i = 0; i < n; ++i) {
        const double v = parse_double(p, "an ASCII Data element");
        if (a.datatype == "NIFTI_TYPE_FLOAT32") store_f32(raw.data() + i * 4, static_cast<float>(v), true);
        else if (a.datatype == "NIFTI_TYPE_INT32") store_u32(raw.data() + i * 4, static_cast<std::uint32_t>(static_cast<std::int32_t>(v)), true);
        else raw[i] = static_cast<unsigned char>(v);
      }
      while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') ++p;
      if (*p) throw format_error("GIFTI: more ASCII values than the dimensions give");
      little = true;   // the bytes were just made little-endian
    } else if (enc == "Base64Binary" || enc == "GZipBase64Binary") {
      a.encoding = enc == "Base64Binary" ? Encoding::base64 : Encoding::gzip_base64;
      raw = b64_decode(text);
      if (a.encoding == Encoding::gzip_base64) raw = inflate_bytes(raw.data(), raw.size(), DeflateWrapper::detect, static_cast<std::size_t>(n) * eb);
    } else if (enc == "ExternalFileBinary") {
      throw refused("GIFTI: an ExternalFileBinary array (data in " + std::string(da.attribute("ExternalFileName").as_string()) + ") is not read: the data must be in the .gii file");
    } else {
      throw format_error("GIFTI: Encoding \"" + enc + "\"");
    }
    if (raw.size() != n * static_cast<std::uint64_t>(eb))
      throw format_error("GIFTI: a " + a.datatype + " array of " + std::to_string(n) + " values has " + std::to_string(raw.size()) + " bytes of data");
    if (a.column_major_in_file && a.dims.size() > 1) permute_column_to_row(raw, a.dims, eb);   // the permutation moves whole elements; swap after
    if (a.datatype == "NIFTI_TYPE_FLOAT32") {
      a.f32.resize(n);
      for (std::uint64_t i = 0; i < n; ++i) a.f32[i] = load_f32(raw.data() + i * 4, little);
    } else if (a.datatype == "NIFTI_TYPE_INT32") {
      a.i32.resize(n);
      for (std::uint64_t i = 0; i < n; ++i) a.i32[i] = load_i32(raw.data() + i * 4, little);
    } else {
      a.u8.assign(raw.begin(), raw.end());
    }
    img.arrays.push_back(std::move(a));
  }
  if (declared >= 0 && static_cast<std::size_t>(declared) != img.arrays.size())
    throw format_error("GIFTI: NumberOfDataArrays is " + std::to_string(declared) + " but the file holds " + std::to_string(img.arrays.size()));
  return img;
}

Image read_file(const std::string& path) {
  FileSource f(path);
  return read(f);
}

namespace {

void write_meta(pugi::xml_node parent, const std::vector<std::pair<std::string, std::string>>& meta) {
  pugi::xml_node m = parent.append_child("MetaData");
  for (const auto& kv : meta) {
    pugi::xml_node md = m.append_child("MD");
    md.append_child("Name").append_child(pugi::node_pcdata).set_value(kv.first.c_str());
    md.append_child("Value").append_child(pugi::node_pcdata).set_value(kv.second.c_str());
  }
}

std::string fmt(const char* f, double v) {
  char b[64];
  std::snprintf(b, sizeof b, f, v);
  return b;
}

}  // namespace

DataArray make_float32(const std::string& intent, std::vector<std::int64_t> dims, std::vector<float> values) {
  DataArray a;
  a.intent = intent;
  a.datatype = "NIFTI_TYPE_FLOAT32";
  a.dims = std::move(dims);
  a.f32 = std::move(values);
  return a;
}
DataArray make_int32(const std::string& intent, std::vector<std::int64_t> dims, std::vector<std::int32_t> values) {
  DataArray a;
  a.intent = intent;
  a.datatype = "NIFTI_TYPE_INT32";
  a.dims = std::move(dims);
  a.i32 = std::move(values);
  return a;
}

void write(ByteSink& sink, const Image& img) {
  pugi::xml_document doc;
  pugi::xml_node decl = doc.append_child(pugi::node_declaration);
  decl.append_attribute("version") = "1.0";
  decl.append_attribute("encoding") = "UTF-8";
  doc.append_child(pugi::node_doctype).set_value("GIFTI SYSTEM \"http://www.nitrc.org/frs/download.php/115/gifti.dtd\"");
  pugi::xml_node root = doc.append_child("GIFTI");
  root.append_attribute("Version") = img.version.c_str();
  root.append_attribute("NumberOfDataArrays") = static_cast<unsigned int>(img.arrays.size());
  write_meta(root, img.metadata);
  pugi::xml_node lt = root.append_child("LabelTable");
  for (const Label& l : img.labels) {
    pugi::xml_node n = lt.append_child("Label");
    n.append_attribute("Key") = l.key;
    if (l.has_colour) {
      n.append_attribute("Red") = fmt("%.9g", l.rgba[0]).c_str();
      n.append_attribute("Green") = fmt("%.9g", l.rgba[1]).c_str();
      n.append_attribute("Blue") = fmt("%.9g", l.rgba[2]).c_str();
      n.append_attribute("Alpha") = fmt("%.9g", l.rgba[3]).c_str();
    }
    n.append_child(pugi::node_pcdata).set_value(l.name.c_str());
  }
  for (const DataArray& a : img.arrays) {
    const int eb = element_bytes(a.datatype);
    const std::uint64_t n = a.count();
    const std::size_t have = a.datatype == "NIFTI_TYPE_FLOAT32" ? a.f32.size() : a.datatype == "NIFTI_TYPE_INT32" ? a.i32.size() : a.u8.size();
    if (have != n) throw range_error("GIFTI writer: " + std::to_string(have) + " values for dims of " + std::to_string(n));
    pugi::xml_node d = root.append_child("DataArray");
    d.append_attribute("Intent") = a.intent.c_str();
    d.append_attribute("DataType") = a.datatype.c_str();
    d.append_attribute("ArrayIndexingOrder") = "RowMajorOrder";
    d.append_attribute("Dimensionality") = static_cast<unsigned int>(a.dims.size());
    const char* enc = a.encoding == Encoding::ascii ? "ASCII" : a.encoding == Encoding::base64 ? "Base64Binary" : "GZipBase64Binary";
    d.append_attribute("Encoding") = enc;
    d.append_attribute("Endian") = "LittleEndian";
    d.append_attribute("ExternalFileName") = "";
    d.append_attribute("ExternalFileOffset") = "0";
    for (std::size_t i = 0; i < a.dims.size(); ++i) d.append_attribute(("Dim" + std::to_string(i)).c_str()) = std::to_string(a.dims[i]).c_str();
    write_meta(d, a.metadata);
    for (const Transform& t : a.transforms) {
      pugi::xml_node c = d.append_child("CoordinateSystemTransformMatrix");
      c.append_child("DataSpace").append_child(pugi::node_pcdata).set_value(t.dataspace.c_str());
      c.append_child("TransformedSpace").append_child(pugi::node_pcdata).set_value(t.xformspace.c_str());
      std::string m;
      for (int i = 0; i < 16; ++i) { m += fmt("%.17g", t.matrix[i]); m += (i % 4 == 3) ? '\n' : ' '; }
      c.append_child("MatrixData").append_child(pugi::node_pcdata).set_value(m.c_str());
    }
    std::string text;
    if (a.encoding == Encoding::ascii) {
      for (std::uint64_t i = 0; i < n; ++i) {
        if (a.datatype == "NIFTI_TYPE_FLOAT32") text += fmt("%.9g", a.f32[i]);
        else if (a.datatype == "NIFTI_TYPE_INT32") text += std::to_string(a.i32[i]);
        else text += std::to_string(a.u8[i]);
        text += '\n';
      }
    } else {
      std::vector<unsigned char> raw(static_cast<std::size_t>(n) * eb);
      for (std::uint64_t i = 0; i < n; ++i) {
        if (a.datatype == "NIFTI_TYPE_FLOAT32") store_f32(raw.data() + i * 4, a.f32[i], true);
        else if (a.datatype == "NIFTI_TYPE_INT32") store_u32(raw.data() + i * 4, static_cast<std::uint32_t>(a.i32[i]), true);
        else raw[i] = a.u8[i];
      }
      if (a.encoding == Encoding::gzip_base64) raw = deflate_bytes(raw.data(), raw.size(), DeflateWrapper::zlib);
      text = b64_encode(raw.data(), raw.size());
    }
    d.append_child("Data").append_child(pugi::node_pcdata).set_value(text.c_str());
  }
  std::ostringstream os;
  doc.save(os, "  ", pugi::format_default);
  sink.write(os.str());
}

void write_file(const std::string& path, const Image& image) {
  FileSink f(path);
  write(f, image);
  f.close();
}

}  // namespace nxr::formats::gifti
