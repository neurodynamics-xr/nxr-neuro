// The projectors and the CTF compensation of an imported recording: see nxr/convert/recording_operators.h.
#include "nxr/convert/recording_operators.h"

#include <algorithm>
#include <map>

#include "nxr/formats/ctf.h"
#include "nxr/formats/fif.h"
#include "nxr/io/array_writer.h"
#include "nxr/io/errors.h"

namespace nxr::convert {

namespace nf = nxr::formats;
using nxr::io::Json;
namespace code = nxr::io::code;

namespace {
[[noreturn]] void refuse(const std::string& m) { throw nxr::io::Error(code::invalid_argument, m); }

// CTF's res4 names carry the sensor's serial after a dash ("MLC11-2802"); the reader's channel names do not
std::string base_name(const std::string& s) { return s.substr(0, s.find('-')); }

std::map<std::string, std::size_t> index_of(const nf::RecordingHeader& h) {
  std::map<std::string, std::size_t> m;
  for (std::size_t i = 0; i < h.channels.size(); ++i) m.emplace(base_name(h.channels[i].name), i);
  return m;
}

// rows -> columns coefficients into a csr [M x M]
CtfCompensation csr_of(const nf::RecordingHeader& h, int grade, bool calibrated, const std::vector<std::string>& rows, const std::vector<std::vector<std::string>>& cols,
                       const std::vector<std::vector<double>>& coeffs) {
  const auto at = index_of(h);
  const std::size_t n = h.channels.size();
  std::map<std::size_t, std::vector<std::pair<std::size_t, double>>> by_row;
  std::vector<bool> is_ref(n, false);
  for (std::size_t r = 0; r < rows.size(); ++r) {
    const auto ri = at.find(base_name(rows[r]));
    if (ri == at.end()) refuse("the CTF compensation names a channel the recording does not have: " + rows[r]);
    for (std::size_t c = 0; c < cols[r].size(); ++c) {
      const auto ci = at.find(base_name(cols[r][c]));
      if (ci == at.end()) refuse("the CTF compensation names a reference channel the recording does not have: " + cols[r][c]);
      is_ref[ci->second] = true;
      if (coeffs[r][c] != 0.0) by_row[ri->second].emplace_back(ci->second, coeffs[r][c]);
    }
  }
  CtfCompensation out;
  out.grade = grade;
  out.calibrated = calibrated;
  out.n_meg = static_cast<std::int64_t>(rows.size());
  out.n_ref = static_cast<std::int64_t>(std::count(is_ref.begin(), is_ref.end(), true));
  out.indptr.push_back(0);
  for (std::size_t i = 0; i < n; ++i) {
    auto it = by_row.find(i);
    if (it != by_row.end()) {
      std::sort(it->second.begin(), it->second.end());
      for (const auto& [c, v] : it->second) { out.indices.push_back(static_cast<std::int32_t>(c)); out.values.push_back(v); }
    }
    out.indptr.push_back(static_cast<std::int32_t>(out.values.size()));
  }
  return out;
}
}  // namespace

RecordingOperators operators_of(const nf::RecordingReader& reader) {
  const nf::RecordingHeader& h = reader.header();
  RecordingOperators ops;
  const std::size_t n = h.channels.size();
  if (const auto* fif = dynamic_cast<const nf::fif::RawReader*>(&reader)) {
    const nf::fif::Info& in = fif->info();
    std::map<std::string, std::size_t> at;
    for (std::size_t i = 0; i < n; ++i) at.emplace(h.channels[i].name, i);
    int order = 0;
    for (const auto& p : in.projectors) {
      nxr::bst::ProjectorEntry e;
      e.order = ++order;
      e.comment = p.desc;
      e.method = "SSP";
      e.kind = "ssp";
      e.status = p.active ? 1 : 0;
      e.m = n;
      e.k = static_cast<std::size_t>(std::max(p.nvec, 0));
      e.components.assign(e.m * e.k, 0.0);
      e.mask.assign(e.k, true);
      const std::size_t nc = p.channels.size();
      if (p.vectors.size() != e.k * nc) refuse("projector " + std::to_string(order) + ": its vectors do not match its channels");
      for (std::size_t c = 0; c < nc; ++c) {
        const auto it = at.find(p.channels[c]);
        if (it == at.end()) continue;                      // a channel the recording does not have: its weight is dropped (MNE does the same)
        for (std::size_t j = 0; j < e.k; ++j) e.components[it->second * e.k + j] = p.vectors[j * nc + c];
      }
      ops.projectors.push_back(std::move(e));
    }
    if (in.compensation_grade > 0) {
      for (const auto& c : in.compensations) {
        if (c.grade != in.compensation_grade) continue;
        if (!c.calibrated) refuse("the file stores its CTF compensation uncalibrated: not imported");
        std::vector<std::string> rows;
        std::vector<std::vector<std::string>> cols;
        std::vector<std::vector<double>> co;
        for (int r = 0; r < c.nrow; ++r) {
          rows.push_back(c.row_names.at(static_cast<std::size_t>(r)));
          cols.push_back(c.col_names);
          co.emplace_back(c.data.begin() + r * c.ncol, c.data.begin() + (r + 1) * c.ncol);
        }
        ops.compensation = csr_of(h, c.grade, c.calibrated, rows, cols, co);
        break;
      }
    }
  } else if (const auto* ctf = dynamic_cast<const nf::ctf::Reader*>(&reader)) {
    const int grade = ctf->compensation_grade();
    if (grade > 0) {
      const std::int32_t want = 0x47304252 + (grade << 16);     // 'G1BR', 'G2BR', 'G3BR'
      std::vector<std::string> rows;
      std::vector<std::vector<std::string>> cols;
      std::vector<std::vector<double>> co;
      for (const auto& r : ctf->res4().compensation) {
        if (r.coeff_type != want) continue;
        rows.push_back(r.sensor_name);
        cols.push_back(r.sensors);
        co.push_back(r.coeffs);
      }
      if (!rows.empty()) ops.compensation = csr_of(h, grade, true, rows, cols, co);
    }
  }
  if (const auto w = h.matrices.find("icaweights"), sph = h.matrices.find("icasphere"), win = h.matrices.find("icawinv"); w != h.matrices.end() && sph != h.matrices.end()) {
    // the channels the ICA ran on: EEGLAB's `icachansind` (1-based), else all
    std::vector<std::size_t> at;
    if (const auto ci = h.matrices.find("icachansind"); ci != h.matrices.end()) for (double x : ci->second.data) at.push_back(static_cast<std::size_t>(x) - 1);
    else for (std::size_t i = 0; i < n; ++i) at.push_back(i);
    for (std::size_t c : at) if (c >= n) refuse("EEGLAB: icachansind names a channel the dataset does not have");
    const std::size_t nc = w->second.rows, ni = w->second.cols;
    if (sph->second.rows != ni || sph->second.cols != ni || at.size() != ni) refuse("EEGLAB: icaweights, icasphere and icachansind do not agree on the channels the ICA ran on");
    std::vector<double> un(nc * n, 0.0);
    for (std::size_t k = 0; k < nc; ++k)
      for (std::size_t j = 0; j < ni; ++j) {
        double x = 0;
        for (std::size_t i = 0; i < ni; ++i) x += w->second.data[k * ni + i] * sph->second.data[i * ni + j];
        un[k * n + at[j]] = x;
      }
    ops.ica_unmixing = {static_cast<std::int64_t>(nc), std::move(un)};
    if (win != h.matrices.end()) {
      if (win->second.rows != ni || win->second.cols != nc) refuse("EEGLAB: icawinv is not channels x components");
      std::vector<double> mx(n * nc, 0.0);
      for (std::size_t j = 0; j < ni; ++j)
        for (std::size_t k = 0; k < nc; ++k) mx[at[j] * nc + k] = win->second.data[j * nc + k];
      ops.ica_mixing = {static_cast<std::int64_t>(nc), std::move(mx)};
    }
  }
  return ops;
}

std::vector<double> applied_projection(const RecordingOperators& ops) { return nxr::bst::applied_matrix(ops.projectors); }

void write_operators(nxr::io::SubjectWriter& sub, RecordingOperators& ops, const std::string& channels_name, const std::string& channels_id, const std::string& acquisition) {
  ops.record = Json::object();
  if (!ops.projectors.empty()) {
    nxr::bst::export_projectors(sub, ops.projectors, channels_name, channels_id, acquisition);
    Json rec = Json::array();
    for (const auto& e : ops.projectors) {
      rec.push_back(Json{{"order", e.order}, {"comment", e.comment}, {"method", e.method}, {"status", e.status}, {"active", e.status == 1},
                         {"applied", e.status == 1 ? "applied by the import" : "not applied"}, {"operator", channels_name + "_projector_" + std::to_string(e.order)}});
    }
    ops.record["projectors"] = rec;
  }
  for (const bool mixing : {false, true}) {
    const auto& m = mixing ? ops.ica_mixing : ops.ica_unmixing;
    if (!m) continue;
    const std::int64_t nc = m->first, M = static_cast<std::int64_t>(m->second.size()) / nc;
    const std::string nm = channels_name + (mixing ? "_ica_mixing" : "_ica_unmixing");
    nxr::io::OperatorSpec op;
    op.name = nm;
    op.path = nm;
    op.kind = mixing ? "ica mixing" : "ica unmixing";
    op.from_manifold_id = channels_id;
    op.n_rows = mixing ? M : nc;
    op.n_cols = mixing ? nc : M;
    op.acquisition = acquisition;
    op.data = nxr::io::view(m->second, {op.n_rows, op.n_cols});
    op.cols.set("description", mixing ? "EEGLAB ICA mixing matrix (icawinv): component activations to channels, on the channels the ICA ran on; not applied"
                                      : "EEGLAB ICA unmixing matrix (icaweights times icasphere): channels to component activations, on the channels the ICA ran on; not applied");
    sub.operator_(std::move(op));
    ops.record[mixing ? "ica_mixing" : "ica_unmixing"] = Json{{"operator", nm}, {"components", nc}, {"applied", false}};
  }
  if (ops.compensation) {
    const CtfCompensation& c = *ops.compensation;
    const std::string nm = channels_name + "_ctf_compensation";
    const std::int64_t N = static_cast<std::int64_t>(c.indptr.size()) - 1, nnz = static_cast<std::int64_t>(c.values.size());
    nxr::io::OperatorSpec op;
    op.name = nm;
    op.path = nm;
    op.kind = "ctf compensation";
    op.from_manifold_id = channels_id;
    op.layout = "csr";
    op.n_rows = N;
    op.n_cols = N;
    op.acquisition = acquisition;
    op.cols.set("to_manifold_id", channels_id).set("nnz", nnz);
    op.cols.json("params_json", Json{{"n_meg", c.n_meg}, {"n_ref", c.n_ref}, {"grade", Json{{"current", c.grade}, {"destination", c.grade}}}});
    op.cols.set("description", "CTF compensation at grade " + std::to_string(c.grade) +
                                   ": entry (MEG channel, reference channel) = the file's coefficient, so a compensated gain is G - C G (rows MEG, columns reference); not applied to the samples");
    op.sparse = nxr::io::SparseArrays{{{"indptr", nxr::io::view(c.indptr, {N + 1})}, {"indices", nxr::io::view(c.indices, {nnz})}, {"data", nxr::io::view(c.values, {nnz})}}};
    sub.operator_(std::move(op));
    ops.record["ctf_compensation"] = Json{{"operator", nm}, {"grade", Json{{"current", c.grade}, {"destination", c.grade}}}, {"applied", false}};
  }
}

}  // namespace nxr::convert
