// FIF measurement info: every field of the header. Ported from MNE-Python's mne/_fiff/meas_info.py (read_meas_info), proj.py (_read_proj),
// ctf_comp.py (_read_ctf_comp), matrix.py (_read_named_matrix), _digitization.py (_read_dig_fif) and annotations.py (_read_annotations_fif)
// (BSD-3-Clause, Copyright the MNE-Python contributors; see THIRD_PARTY_NOTICES.md).
#include <cmath>
#include <map>
#include <set>

#include "fif_info.h"
#include "fif_internal.h"
#include "nlohmann/json.hpp"
#include "nxr/formats/errors.h"

namespace nxr::formats::fif {

namespace {

using detail::split_name_list;

constexpr std::int32_t kDateNoneUsecs = 2147483647;   // MNE's DATE_NONE = (0, 2147483647)

std::int64_t as_int(const Value& v, const char* what) {
  if (!v.ints.empty()) return v.ints[0];
  if (!v.reals.empty()) return static_cast<std::int64_t>(v.reals[0]);
  throw format_error(std::string("FIF: the tag for ") + what + " is empty");
}
double as_real(const Value& v, const char* what) {
  if (!v.reals.empty()) return v.reals[0];
  if (!v.ints.empty()) return static_cast<double>(v.ints[0]);
  throw format_error(std::string("FIF: the tag for ") + what + " is empty");
}
std::vector<std::int64_t> as_ints(const Value& v) {
  if (!v.ints.empty()) return v.ints;
  std::vector<std::int64_t> o;
  for (double d : v.reals) o.push_back(static_cast<std::int64_t>(d));
  return o;
}
std::vector<double> as_reals(const Value& v) {
  if (!v.reals.empty()) return v.reals;
  std::vector<double> o;
  for (std::int64_t i : v.ints) o.push_back(static_cast<double>(i));
  return o;
}

bool is_nan(double d) { return std::isnan(d); }

std::int32_t coord_frame_of_kind(std::int32_t kind) {
  switch (kind) {
    case 1: case 301: return 1;                              // MEG, reference MEG: device
    case 2: case 902: case 802: case 803: case 1100: return 4;   // EEG, ECoG, sEEG, DBS, fNIRS: head
    default: return 0;
  }
}

class Reader {
 public:
  Reader(const File& f) : f_(f) {}
  Value val(const TagHeader& t) const { return f_.value(t); }

  GenericTag generic(const TagHeader& t) const {
    GenericTag g;
    g.kind = t.kind;
    g.type = t.type;
    try {
      g.value = f_.value(t);
    } catch (const nxr::io::Error&) {
      g.value.bytes = f_.bytes(t);   // a type this reader does not decode (a sparse matrix): kept as stored
    }
    return g;
  }
  GenericBlock generic_block(const Block& b) const {
    GenericBlock g;
    g.kind = b.kind;
    for (const TagHeader& t : b.tags)
      if (t.kind != tag::kNop) g.tags.push_back(generic(t));
    for (const Block& c : b.children) g.children.push_back(generic_block(c));
    return g;
  }

  std::vector<DigPoint> dig_in(const Block& iso, std::vector<std::string>* names, std::int32_t* frame) const {
    std::vector<DigPoint> dig;
    *frame = 4;
    for (const TagHeader& t : iso.tags) {
      if (t.kind == tag::kDigPoint) {
        const auto b = f_.bytes(t);
        if (b.size() < 20) throw format_error("FIF: a digitisation point is truncated");
        dig.push_back(detail::decode_dig_point(b.data()));
      } else if (t.kind == tag::kDigString) {
        for (DigPoint& d : detail::decode_dig_string(f_.bytes(t))) dig.push_back(d);
      } else if (t.kind == tag::kMneCoordFrame) {
        *frame = static_cast<std::int32_t>(as_int(val(t), "the digitisation coordinate frame"));
      } else if (t.kind == tag::kMneChNameList) {
        *names = split_name_list(val(t).text);
      }
    }
    for (DigPoint& d : dig) d.coord_frame = *frame;
    return dig;
  }

  std::optional<CoordTrans> coord_trans(const TagHeader& t) const { return detail::decode_coord_trans(f_.bytes(t)); }

 private:
  const File& f_;
};

std::string rename_one(const std::string& n, const std::map<std::string, std::string>& m) {
  const auto it = m.find(n);
  return it == m.end() ? n : it->second;
}

}  // namespace

std::int32_t channel_coord_frame(std::int32_t kind) { return coord_frame_of_kind(kind); }

std::string julian_to_iso_date(std::int32_t jd) {
  // MNE: 1970-01-01 12:00 is julian day 2440588; the date of noon + (jd - 2440588) days. Civil-from-days (Howard Hinnant's algorithm).
  std::int64_t z = static_cast<std::int64_t>(jd) - 2440588 + 719468;
  const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
  const std::int64_t doe = z - era * 146097;
  const std::int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  std::int64_t y = yoe + era * 400;
  const std::int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const std::int64_t mp = (5 * doy + 2) / 153;
  const std::int64_t d = doy - (153 * mp + 2) / 5 + 1;
  const std::int64_t m = mp < 10 ? mp + 3 : mp - 9;
  if (m <= 2) ++y;
  char buf[32];
  std::snprintf(buf, sizeof buf, "%04lld-%02lld-%02lld", static_cast<long long>(y), static_cast<long long>(m), static_cast<long long>(d));
  return buf;
}

std::vector<Projector> read_projectors(const File& f, const Block& scope, const std::map<std::string, std::string>& renamed) {
  Reader R(f);
  std::vector<Projector> out;
  const auto pb = find_blocks(scope, block::kProj);
  if (pb.empty()) return out;
  for (const Block* item : find_blocks(*pb[0], block::kProjItem)) {
    Projector p;
    const TagHeader* d = find_tag(*item, tag::kDescription);
    if (!d) d = find_tag(*item, tag::kName);
    if (!d) throw format_error("FIF: a projection item has no description");
    p.desc = R.val(*d).text;
    const TagHeader* tk = find_tag(*item, tag::kProjItemKind);
    if (!tk) throw format_error("FIF: a projection item has no kind");
    p.kind = static_cast<std::int32_t>(as_int(R.val(*tk), "a projection kind"));
    const TagHeader* tn = find_tag(*item, tag::kProjItemNvec);
    if (!tn) throw format_error("FIF: a projection item has no vector count");
    p.nvec = static_cast<std::int32_t>(as_int(R.val(*tn), "a projection vector count"));
    const TagHeader* tc = find_tag(*item, tag::kProjItemChNameList);
    if (!tc) throw format_error("FIF: a projection item has no channel list");
    p.channels = split_name_list(R.val(*tc).text);
    const TagHeader* tv = find_tag(*item, tag::kProjItemVectors);
    if (!tv) throw format_error("FIF: a projection item has no vectors");
    const Value vv = R.val(*tv);
    if (vv.dims.size() != 2) throw format_error("FIF: projection vectors are not a matrix");
    std::vector<double> data = as_reals(vv);
    std::int32_t rows = vv.dims[0], cols = vv.dims[1];
    if (rows == static_cast<std::int32_t>(p.channels.size()) && cols == p.nvec && rows != p.nvec) {   // stored transposed
      std::vector<double> tr(data.size());
      for (std::int32_t r = 0; r < rows; ++r)
        for (std::int32_t c = 0; c < cols; ++c) tr[static_cast<std::size_t>(c) * rows + r] = data[static_cast<std::size_t>(r) * cols + c];
      data = tr;
      std::swap(rows, cols);
    }
    if (cols != static_cast<std::int32_t>(p.channels.size())) throw format_error("FIF: the number of channel names does not match the projection matrix");
    p.vectors = std::move(data);
    const TagHeader* ta = find_tag(*item, tag::kMneProjItemActive);
    p.active = ta ? as_int(R.val(*ta), "a projection's active flag") != 0 : false;
    if (const TagHeader* te = find_tag(*item, tag::kMneIcaPcaExplainedVar)) p.explained_var = as_real(R.val(*te), "explained variance");
    for (std::string& n : p.channels) n = rename_one(n, renamed);
    out.push_back(std::move(p));
  }
  return out;
}

std::map<std::string, std::string> apply_extended_channel_info(const File& f, const Block& scope, std::vector<ChannelInfo>& channels) {
  Reader R(f);
  std::map<std::string, std::string> renamed;
  {
    const auto ext = find_blocks(scope, block::kChInfo);
    if (!ext.empty()) {
      if (ext.size() != channels.size())
        throw format_error("FIF: " + std::to_string(ext.size()) + " extended channel information blocks for " + std::to_string(channels.size()) + " channels");
      for (std::size_t i = 0; i < ext.size(); ++i) {
        ChannelInfo& ch = channels[i];
        const std::string old_name = ch.name;
        for (const TagHeader& t : ext[i]->tags) {
          if (t.kind == tag::kNop) continue;
          const Value v = R.val(t);
          switch (t.kind) {
            case tag::kChScanNo: ch.scanno = static_cast<std::int32_t>(as_int(v, "a channel scan number")); break;
            case tag::kChLogicalNo: ch.logno = static_cast<std::int32_t>(as_int(v, "a channel logical number")); break;
            case tag::kChKind: ch.kind = static_cast<std::int32_t>(as_int(v, "a channel kind")); break;
            case tag::kChRange: ch.range = as_real(v, "a channel range"); break;
            case tag::kChCal: ch.cal = as_real(v, "a channel cal"); break;
            case tag::kChCoilType: ch.coil_type = static_cast<std::int32_t>(as_int(v, "a coil type")); break;
            case tag::kChLoc: { const auto r = as_reals(v); for (std::size_t k = 0; k < 12 && k < r.size(); ++k) ch.loc[k] = r[k]; break; }
            case tag::kChUnit: ch.unit = static_cast<std::int32_t>(as_int(v, "a channel unit")); break;
            case tag::kChUnitMul: ch.unit_mul = static_cast<std::int32_t>(as_int(v, "a channel unit multiplier")); break;
            case tag::kChDacqName: ch.name = v.text; break;
            case tag::kChCoordFrame: break;   // MNE overwrites it from the channel kind straight after (_update_ch_info_named); so do we
            default: break;                    // MNE warns and discards unknown extended kinds
          }
        }
        if (ch.name != old_name) renamed[old_name] = ch.name;
      }
    }
  }
  return renamed;
}

Info read_info(const File& f, const Block& meas) {
  const Reader R(f);
  Info info;
  info.file_id = f.file_id();

  const auto infos = find_blocks(meas, block::kMeasInfo);
  if (infos.empty()) throw format_error("FIF: no measurement info block");
  if (infos.size() > 1) throw refused("FIF: more than one measurement info block (" + std::to_string(infos.size()) + ")");
  const Block& mi = *infos[0];

  std::optional<CoordTrans> dev_head, ctf_head, dev_ctf;
  std::optional<std::int32_t> nchan;
  std::optional<std::array<std::int32_t, 2>> meas_date_tag;
  std::set<std::int32_t> handled = {tag::kNchan, tag::kSfreq, tag::kChInfo, tag::kLowpass, tag::kHighpass, tag::kMeasDate, tag::kUtcOffset,
                                    tag::kExperimenter, tag::kDescription, tag::kProjId, tag::kProjName, tag::kLineFreq, tag::kGantryAngle,
                                    tag::kMneCustomRef, 236, tag::kXplotterLayout, tag::kMneKitSystemId, tag::kBlockId, tag::kParentBlockId,
                                    tag::kNop};
  for (const TagHeader& t : mi.tags) {
    switch (t.kind) {
      case tag::kNchan: nchan = static_cast<std::int32_t>(as_int(R.val(t), "the channel count")); break;
      case tag::kSfreq: info.sfreq = as_real(R.val(t), "the sampling frequency"); break;
      case tag::kChInfo: info.channels.push_back(detail::decode_ch_info(f.bytes(t))); break;
      case tag::kLowpass: { const double v = as_real(R.val(t), "the lowpass"); if (!is_nan(v)) info.lowpass = v; break; }
      case tag::kHighpass: { const double v = as_real(R.val(t), "the highpass"); if (!is_nan(v)) info.highpass = v; break; }
      case tag::kMeasDate: {
        const auto v = as_ints(R.val(t));
        if (v.empty()) break;
        meas_date_tag = std::array<std::int32_t, 2>{static_cast<std::int32_t>(v[0]), v.size() > 1 ? static_cast<std::int32_t>(v[1]) : 0};
        break;
      }
      case tag::kUtcOffset: info.utc_offset = R.val(t).text; break;
      case tag::kCoordTrans: {
        const CoordTrans c = *R.coord_trans(t);
        if (c.from == 1 && c.to == 4) dev_head = c;                 // device -> head
        else if (c.from == 4 && c.to == 1) dev_head = detail::invert(c);   // reversed (BabyMEG)
        else if (c.from == 1004 && c.to == 4) ctf_head = c;         // CTF head -> head
        else if (c.from == 1001 && c.to == 1004) dev_ctf = c;       // CTF device -> CTF head
        else info.other_tags.push_back(R.generic(t));
        break;
      }
      case tag::kExperimenter: info.experimenter = R.val(t).text; break;
      case tag::kDescription: info.description = R.val(t).text; break;
      case tag::kProjId: info.proj_id = static_cast<std::int32_t>(as_int(R.val(t), "the project id")); break;
      case tag::kProjName: info.proj_name = R.val(t).text; break;
      case tag::kLineFreq: info.line_freq = as_real(R.val(t), "the line frequency"); break;
      case tag::kGantryAngle: info.gantry_angle = static_cast<std::int32_t>(as_int(R.val(t), "the gantry angle")); break;
      case tag::kMneCustomRef: case 236: info.custom_ref_applied = static_cast<std::int32_t>(as_int(R.val(t), "the custom reference flag")); break;
      case tag::kXplotterLayout: info.xplotter_layout = R.val(t).text; break;
      case tag::kMneKitSystemId: info.kit_system_id = static_cast<std::int32_t>(as_int(R.val(t), "the KIT system id")); break;
      default:
        if (!handled.count(t.kind)) info.other_tags.push_back(R.generic(t));
        break;
    }
  }

  // extended channel information (long names): one block per channel, in the channels' order
  const std::map<std::string, std::string> renamed = apply_extended_channel_info(f, mi, info.channels);
  for (ChannelInfo& ch : info.channels) ch.coord_frame = coord_frame_of_kind(ch.kind);

  if (!nchan) throw format_error("FIF: the number of channels is not defined");
  if (info.sfreq <= 0.0) throw format_error("FIF: the sampling frequency is not defined");
  if (info.channels.empty()) throw format_error("FIF: no channel information");
  if (static_cast<std::size_t>(*nchan) != info.channels.size()) throw format_error("FIF: " + std::to_string(info.channels.size()) + " channel definitions for " + std::to_string(*nchan) + " channels");
  info.nchan = *nchan;

  // the HPI results can give the transforms the info does not
  const auto hpi_blocks = find_blocks(mi, block::kHpiResult);
  if (!dev_head || !ctf_head) {
    if (hpi_blocks.size() == 1) {
      for (const TagHeader& t : hpi_blocks[0]->tags) {
        if (t.kind != tag::kCoordTrans) continue;
        const CoordTrans c = *R.coord_trans(t);
        if (c.from == 1 && c.to == 4 && !dev_head) dev_head = c;
        else if (c.from == 1004 && c.to == 4 && !ctf_head) ctf_head = c;
      }
    }
  }

  // digitisation points (one isotrak block; MNE ignores several)
  {
    const auto iso = find_blocks(mi, block::kIsotrak);
    if (iso.size() == 1) {
      std::int32_t frame = 4;
      info.dig = R.dig_in(*iso[0], &info.dig_channel_names, &frame);
      info.has_dig = true;
    }
  }

  // acquisition parameters
  {
    const auto acq = find_blocks(mi, block::kDacqPars);
    if (acq.size() == 1) {
      for (const TagHeader& t : acq[0]->tags) {
        if (t.kind == tag::kDacqPars) info.acq_pars = R.val(t).text;
        else if (t.kind == tag::kDacqStim) info.acq_stim = R.val(t).text;
      }
    }
  }

  // SSP projectors
  info.projectors = read_projectors(f, mi, renamed);

  // CTF compensation
  for (const Block* cb : find_blocks(mi, block::kMneCtfCompData)) {
    const Block* mat = nullptr;
    for (const Block& c : cb->children)
      if (c.kind == block::kMneNamedMatrix && find_tag(c, tag::kMneCtfCompData)) { mat = &c; break; }
    if (!mat) throw format_error("FIF: a compensation block has no matrix");
    Compensation comp;
    const TagHeader* td = find_tag(*mat, tag::kMneCtfCompData);
    const Value dv = R.val(*td);
    if (dv.dims.size() != 2) throw format_error("FIF: a compensation matrix is not two-dimensional");
    comp.nrow = dv.dims[0];
    comp.ncol = dv.dims[1];
    comp.data = as_reals(dv);
    if (const TagHeader* t = find_tag(*mat, tag::kMneNrow))
      if (as_int(R.val(*t), "nrow") != comp.nrow) throw format_error("FIF: the compensation matrix row count does not match its tag");
    if (const TagHeader* t = find_tag(*mat, tag::kMneNcol))
      if (as_int(R.val(*t), "ncol") != comp.ncol) throw format_error("FIF: the compensation matrix column count does not match its tag");
    auto plain_split = [](const std::string& s) {
      std::vector<std::string> o;
      if (s.empty()) return o;
      std::size_t start = 0;
      for (;;) {
        const std::size_t at = s.find(':', start);
        o.push_back(s.substr(start, at == std::string::npos ? std::string::npos : at - start));
        if (at == std::string::npos) break;
        start = at + 1;
      }
      return o;
    };
    if (const TagHeader* t = find_tag(*mat, tag::kMneRowNames)) comp.row_names = plain_split(R.val(*t).text);
    if (const TagHeader* t = find_tag(*mat, tag::kMneColNames)) comp.col_names = plain_split(R.val(*t).text);
    const TagHeader* tk = find_tag(*cb, tag::kMneCtfCompKind);
    if (!tk) throw format_error("FIF: the compensation type is not found");
    comp.ctfkind = static_cast<std::int32_t>(as_int(R.val(*tk), "a compensation kind"));
    comp.grade = comp.ctfkind == 0x47314252 ? 1 : comp.ctfkind == 0x47324252 ? 2 : comp.ctfkind == 0x47334252 ? 3 : comp.ctfkind;
    if (const TagHeader* t = find_tag(*cb, tag::kMneCtfCompCalibrated)) comp.calibrated = as_int(R.val(*t), "calibrated") != 0;
    for (auto* names : {&comp.row_names, &comp.col_names})
      for (std::string& n : *names) n = rename_one(n, renamed);
    info.compensations.push_back(std::move(comp));
  }

  // bad channels
  for (const Block* bb : find_blocks(mi, block::kMneBadChannels)) {
    if (const TagHeader* t = find_tag(*bb, tag::kMneChNameList)) {
      info.bads = split_name_list(R.val(*t).text);
    }
  }
  for (std::string& n : info.bads) n = rename_one(n, renamed);

  // events
  for (const Block* eb : find_blocks(mi, block::kEvents)) {
    EventList ev;
    for (const TagHeader& t : eb->tags) {
      if (t.kind == tag::kEventChannels) ev.channels = as_ints(R.val(t));
      else if (t.kind == tag::kEventList) ev.list = as_ints(R.val(t));
    }
    info.events.push_back(std::move(ev));
  }

  // HPI results
  for (const Block* hb : hpi_blocks) {
    HpiResult h;
    for (const TagHeader& t : hb->tags) {
      switch (t.kind) {
        case tag::kDigPoint: { const auto b = f.bytes(t); if (b.size() >= 20) h.dig_points.push_back(detail::decode_dig_point(b.data())); break; }
        case tag::kHpiDigitizationOrder: h.order = as_ints(R.val(t)); break;
        case tag::kHpiCoilsUsed: h.used = as_ints(R.val(t)); break;
        case tag::kHpiCoilMoments: h.moments = as_reals(R.val(t)); break;
        case tag::kHpiFitGoodness: h.goodness = as_reals(R.val(t)); break;
        case tag::kHpiFitGoodLimit: h.good_limit = as_real(R.val(t), "the HPI good limit"); break;
        case tag::kHpiFitDistLimit: h.dist_limit = as_real(R.val(t), "the HPI distance limit"); break;
        case tag::kHpiFitAccept: h.accept = static_cast<std::int32_t>(as_int(R.val(t), "the HPI accept flag")); break;
        case tag::kCoordTrans: h.coord_trans = R.coord_trans(t); break;
        default: break;
      }
    }
    info.hpi_results.push_back(std::move(h));
  }

  // HPI measurements
  for (const Block* hm : find_blocks(mi, block::kHpiMeas)) {
    HpiMeasurement m;
    if (hm->id) m.block_id = hm->id;
    if (hm->parent_id) m.parent_id = hm->parent_id;
    for (const TagHeader& t : hm->tags) {
      switch (t.kind) {
        case tag::kCreator: m.creator = R.val(t).text; break;
        case tag::kSfreq: m.sfreq = as_real(R.val(t), "an HPI measurement's sampling frequency"); break;
        case tag::kNchan: m.nchan = static_cast<std::int32_t>(as_int(R.val(t), "an HPI measurement's channel count")); break;
        case tag::kNave: m.nave = static_cast<std::int32_t>(as_int(R.val(t), "an HPI measurement's average count")); break;
        case tag::kHpiNcoil: m.ncoil = static_cast<std::int32_t>(as_int(R.val(t), "the HPI coil count")); break;
        case tag::kFirstSample: m.first_samp = static_cast<std::int32_t>(as_int(R.val(t), "the first sample")); break;
        case tag::kLastSample: m.last_samp = static_cast<std::int32_t>(as_int(R.val(t), "the last sample")); break;
        default: break;
      }
    }
    for (const Block* cb : find_blocks(*hm, block::kHpiCoil)) {
      HpiCoil c;
      for (const TagHeader& t : cb->tags) {
        switch (t.kind) {
          case tag::kHpiCoilNo: c.number = static_cast<std::int32_t>(as_int(R.val(t), "an HPI coil number")); break;
          case tag::kEpoch: c.epoch = as_reals(R.val(t)); break;
          case tag::kHpiSlopes: c.slopes = as_reals(R.val(t)); break;
          case tag::kHpiCorrCoeff: c.corr_coeff = as_reals(R.val(t)); break;
          case tag::kHpiCoilFreq: c.coil_freq = as_real(R.val(t), "an HPI coil frequency"); break;
          default: break;
        }
      }
      m.coils.push_back(std::move(c));
    }
    info.hpi_measurements.push_back(std::move(m));
  }

  // subject
  {
    const auto sb = find_blocks(mi, block::kSubject);
    if (sb.size() == 1) {
      SubjectInfo s;
      for (const TagHeader& t : sb[0]->tags) {
        switch (t.kind) {
          case tag::kSubjId: s.id = static_cast<std::int32_t>(as_int(R.val(t), "the subject id")); break;
          case tag::kSubjHisId: s.his_id = R.val(t).text; break;
          case tag::kSubjLastName: s.last_name = R.val(t).text; break;
          case tag::kSubjFirstName: s.first_name = R.val(t).text; break;
          case tag::kSubjMiddleName: s.middle_name = R.val(t).text; break;
          case tag::kSubjBirthDay: {
            const std::int32_t jd = static_cast<std::int32_t>(as_int(R.val(t), "the subject's birthday"));
            s.birthday_julian = jd;
            s.birthday = julian_to_iso_date(jd);
            break;
          }
          case tag::kSubjSex: s.sex = static_cast<std::int32_t>(as_int(R.val(t), "the subject's sex")); break;
          case tag::kSubjHand: s.hand = static_cast<std::int32_t>(as_int(R.val(t), "the subject's handedness")); break;
          case tag::kSubjWeight: s.weight = as_real(R.val(t), "the subject's weight"); break;
          case tag::kSubjHeight: s.height = as_real(R.val(t), "the subject's height"); break;
          default: break;
        }
      }
      info.subject_info = std::move(s);
    }
  }
  {
    const auto db = find_blocks(mi, block::kDevice);
    if (db.size() == 1) {
      DeviceInfo d;
      for (const TagHeader& t : db[0]->tags) {
        if (t.kind == tag::kDeviceType) d.type = R.val(t).text;
        else if (t.kind == tag::kDeviceModel) d.model = R.val(t).text;
        else if (t.kind == tag::kDeviceSerial) d.serial = R.val(t).text;
        else if (t.kind == tag::kDeviceSite) d.site = R.val(t).text;
      }
      info.device_info = std::move(d);
    }
  }
  {
    const auto hb = find_blocks(mi, block::kHelium);
    if (hb.size() == 1) {
      HeliumInfo h;
      for (const TagHeader& t : hb[0]->tags) {
        if (t.kind == tag::kHeLevelRaw) h.he_level_raw = as_real(R.val(t), "the raw helium level");
        else if (t.kind == tag::kGantryAngle) h.gantry_angle = static_cast<std::int32_t>(as_int(R.val(t), "the gantry angle"));
        else if (t.kind == tag::kHeliumLevel) h.helium_level = as_real(R.val(t), "the helium level");
        else if (t.kind == tag::kOrigFileGuid) h.orig_file_guid = R.val(t).text;
        else if (t.kind == tag::kMeasDate) {
          const auto v = as_ints(R.val(t));
          if (v.size() >= 2) h.meas_date = std::array<std::int32_t, 2>{static_cast<std::int32_t>(v[0]), static_cast<std::int32_t>(v[1])};
        }
      }
      info.helium_info = std::move(h);
    }
  }
  {
    const auto hs = find_blocks(mi, block::kHpiSubsystem);
    if (hs.size() == 1) {
      HpiSubsystem s;
      for (const TagHeader& t : hs[0]->tags) {
        if (t.kind == tag::kHpiNcoil) s.ncoil = static_cast<std::int32_t>(as_int(R.val(t), "the HPI subsystem coil count"));
        else if (t.kind == tag::kEventChannel) s.event_channel = R.val(t).text;
      }
      for (const Block* cb : find_blocks(*hs[0], block::kHpiCoil))
        for (const TagHeader& t : cb->tags)
          if (t.kind == tag::kEventBits) s.coil_event_bits.push_back(as_ints(R.val(t)));
      info.hpi_subsystem = std::move(s);
    }
  }

  // the measurement id MNE picks
  {
    if (mi.parent_id) info.meas_id = mi.parent_id;
    else if (mi.id) info.meas_id = mi.id;
    else if (meas.id) info.meas_id = meas.id;
    else if (meas.parent_id) info.meas_id = meas.parent_id;
    else info.meas_id = f.file_id();
  }
  // the date: the tag, else the measurement id's time; (0, 2147483647) means no date
  {
    std::array<std::int32_t, 2> md{0, kDateNoneUsecs};
    bool have = false;
    if (meas_date_tag) { md = *meas_date_tag; have = true; }
    else if (info.meas_id) { md = {info.meas_id->secs, info.meas_id->usecs}; have = true; }
    if (have && !(md[0] == 0 && md[1] == kDateNoneUsecs)) info.meas_date = md;
  }

  info.dev_head_t = dev_head;
  info.ctf_head_t = ctf_head;
  info.dev_ctf_t = dev_ctf;
  if (dev_head && ctf_head && !dev_ctf) {   // MNE computes it: inv(ctf_head_t) x dev_head_t
    const CoordTrans inv = detail::invert(*ctf_head);
    CoordTrans o;
    o.from = 1001;
    o.to = 1004;
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c) {
        double s = 0;
        for (int k = 0; k < 4; ++k) s += inv.trans[r * 4 + k] * dev_head->trans[k * 4 + c];
        o.trans[r * 4 + c] = s;
      }
    info.dev_ctf_t = o;
  }

  // everything else in the info and measurement blocks, as generic tags and blocks
  {
    const std::set<std::int32_t> info_handled = {block::kSubject, block::kIsotrak, block::kHpiMeas, block::kHpiResult, block::kDacqPars, block::kProj,
                                                 block::kMneCtfComp, block::kMneBadChannels, block::kEvents, block::kDevice, block::kHelium,
                                                 block::kHpiSubsystem, block::kChInfo};
    for (const Block& c : mi.children)
      if (!info_handled.count(c.kind)) info.other_blocks.push_back(R.generic_block(c));
    const std::set<std::int32_t> meas_handled = {block::kMeasInfo, block::kRawData, block::kContinuousData, block::kIasRawData, block::kMneAnnotations, block::kRef};
    for (const Block& c : meas.children)
      if (!meas_handled.count(c.kind)) info.other_blocks.push_back(R.generic_block(c));
  }

  // grade in effect
  {
    std::int32_t grade = 0;
    bool first = true;
    for (const ChannelInfo& c : info.channels) {
      if (c.kind != 1) continue;
      const std::int32_t g = c.coil_type >> 16;
      if (first) { grade = g; first = false; }
      else if (g != grade) { grade = -1; break; }
    }
    info.compensation_grade = grade;
  }
  return info;
}

std::optional<AnnotationSet> read_annotations(const File& f, const Block& root) {
  const auto blocks = find_blocks(root, block::kMneAnnotations);
  if (blocks.empty()) return std::nullopt;
  const Block& b = *blocks[0];
  AnnotationSet set;
  std::vector<double> onset, end;
  std::vector<std::string> desc;
  nlohmann::json drop_log;
  nlohmann::ordered_json extras;   // ordered: the keys keep the file's order
  bool have_drop = false, have_extras = false;
  for (const TagHeader& t : b.tags) {
    if (t.kind == tag::kNop) continue;
    const Value v = f.value(t);
    switch (t.kind) {
      case tag::kMneBaselineMin: onset = as_reals(v); break;
      case tag::kMneBaselineMax: end = as_reals(v); break;
      case tag::kComment: desc = split_name_list(v.text); break;
      case tag::kMeasDate: {
        if (v.reals.size() == 1) set.orig_time = v.reals[0];
        else if (v.reals.size() >= 2) set.orig_time = v.reals[0] + v.reals[1] * 1e-6;   // MNE writes (seconds, microseconds) as two doubles
        else if (v.ints.size() >= 2) {
          if (!(v.ints[0] == 0 && v.ints[1] == kDateNoneUsecs)) set.orig_time = static_cast<double>(v.ints[0]) + static_cast<double>(v.ints[1]) * 1e-6;
        } else if (v.ints.size() == 1) set.orig_time = static_cast<double>(v.ints[0]);
        break;
      }
      case tag::kMneEpochsDropLog: drop_log = nlohmann::json::parse(v.text, nullptr, false); have_drop = !drop_log.is_discarded(); break;
      case tag::kFreeList: extras = nlohmann::ordered_json::parse(v.text, nullptr, false); have_extras = !extras.is_discarded(); break;
      default: break;
    }
  }
  if (onset.size() != end.size() || onset.size() != desc.size())
    throw format_error("FIF: the annotation onsets, ends and descriptions differ in number (" + std::to_string(onset.size()) + ", " + std::to_string(end.size()) + ", " + std::to_string(desc.size()) + ")");
  for (std::size_t i = 0; i < onset.size(); ++i) {
    AnnotationRecord a;
    a.onset = onset[i];
    a.duration = end[i] - onset[i];
    a.description = desc[i];
    if (have_drop && drop_log.is_array() && i < drop_log.size() && drop_log[i].is_array())
      for (const auto& c : drop_log[i])
        if (c.is_string()) a.channels.push_back(c.get<std::string>());
    if (have_extras && extras.is_array() && i < extras.size()) a.extras_json = extras[i].dump();
    set.items.push_back(std::move(a));
  }
  return set;
}

}  // namespace nxr::formats::fif
