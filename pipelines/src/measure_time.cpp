// nxr-pipelines — measurement over time (measure.h, MeasureOverTimeOptions): a time-resolved Field over the windows of a time partition, the Studio's
// reduceTiles folded across the chunk service's chunks, stored as one array-backed row a measure (D96, as the envelope).
#include <cmath>

#include "nxr/pipelines/task.h"
#include <limits>
#include <regex>

#include "nxr/io/constant_q.h"
#include "nxr/io/measures.h"
#include "nxr/partition.h"
#include "nxr/pipelines/measure.h"
#include "nxr/pipelines/reading.h"

namespace nxr::pipelines {

using namespace nxr::io;

namespace {
[[noreturn]] void invalid(const std::string& m) { throw Error(code::invalid_argument, m); }
[[noreturn]] void missing(const std::string& m) { throw Error(code::no_such_row, m); }
std::string sanitized(const std::string& s) { std::string o; for (char c : s) o += (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') ? c : '_'; return o; }
const std::vector<std::string> kDefault = {"mean", "mean_square", "max", "min"};
const std::vector<std::string> kKnown = {"mean", "mean_square", "max", "min", "abs_max"};
}  // namespace

MeasureComputeResult measure_over_time(DatasetWriter& ds, SubjectWriter& sub, const MeasureOverTimeOptions& opt, const std::function<void(const Json&)>& progress) {
  (void)ds;
  auto say = [&](const Json& j) { if (progress) progress(j); };
  // THE TIME PARTITION: a ladder's level, or a closed-form spans tiling
  const task::Subject subj = task::Subject::of(sub);
  const auto top = subj.find({.cls = task::Cls::selection, .name = opt.over});
  if (!top) missing(sub.name() + ": no Selection named '" + opt.over + "' (a time ladder: `constantq compute`; or a closed-form spans tiling such as <recording>_chunks)");
  task::Node member = *top;
  if (top->type.value_or(std::string()) == "ladder") {
    if (!opt.level) invalid(opt.over + " is a ladder: name its level");
    const auto m = subj.member(*top, *opt.level);
    if (!m) invalid(opt.over + " has no level " + std::to_string(*opt.level));
    member = *m;
  } else if (opt.level) {
    invalid(opt.over + " is not a ladder: it has no levels");
  }
  if (member.type.value_or(std::string()) != "spans") invalid(member.name + " is a " + member.type.value_or(std::string()) + ", not a tiling of time (spans)");
  const Json& mp = member.params;
  if (!mp.is_object() || !mp.contains("tiling")) invalid(member.name + " has no closed-form tiling");
  const std::string line = member.manifold_id.value_or(std::string());
  const auto lax = subj.geometry(line).axes;
  if (lax.empty() || !lax[0].spacing) invalid(member.name + ": its manifold is not a closed-form time line");
  const double sfreq = 1.0 / *lax[0].spacing, t0 = lax[0].origin.value_or(0.0);
  const std::int64_t samples = lax[0].n_vertices;
  const Json& tl = mp["tiling"];
  const std::int64_t width = static_cast<std::int64_t>(std::llround(tl["width"].get<double>() * sfreq)), first = static_cast<std::int64_t>(std::llround((tl["origin"].get<double>() - t0) * sfreq));
  const std::int64_t count = tl["count"].get<std::int64_t>();
  if (width < 1 || std::llround(tl["hop"].get<double>() * sfreq) != width || first != 0) invalid(member.name + ": only contiguous windows from the line's start (hop = width, origin = the line's) are measured");
  // THE MEASURES
  std::vector<std::string> measures = opt.measures.empty() ? kDefault : opt.measures;
  for (const auto& m : measures) if (std::find(kKnown.begin(), kKnown.end(), m) == kKnown.end()) invalid("measure '" + m + "': over time one of mean, mean_square, max, min, abs_max");
  std::vector<std::string> comps = measures;
  comps.push_back("count");
  const std::size_t C = comps.size();
  // THE FIELDS: time-resolved, on a product whose time factor (dimension 1) is the tiling's line
  std::regex re;
  try { re = std::regex(opt.fields, std::regex::ECMAScript); } catch (const std::regex_error&) { invalid("--fields is not a regex: " + opt.fields); }
  std::vector<task::Node> fields;
  for (const task::Node& f : subj.fields_on_axis(line, 1))
    if (std::regex_match(f.name, re)) fields.push_back(f);
  if (fields.empty()) invalid("no time-resolved Field on " + member.name + "'s time line matches '" + opt.fields + "'");
  const std::string by = std::string("measure over time \xc2\xb7 nxr-neuro ") + NXR_NEURO_VERSION;
  Json done = Json::array();
  for (const task::Node& f : fields) {
    const std::string fid = f.id, fname = f.name;
    // refused unless replace: rows already there for this (window member, Field)
    const bool have = subj.has_measures(member.id, fid);
    if (have && !opt.replace) invalid(fname + " is already measured over " + member.name + " (replace to measure it again)");
    // the span (the chunk service: one chunk by default) and the windows it touches
    // the span snaps to the tiling's windows when they are tiles (a power of two: a ladder's level, the chunk grid;), so its windows are whole
    const Span span = span_of(f, samples, sfreq, t0, opt.chunks, opt.start, opt.stop, std::max(0, nxr::io::constantq::levelOfWidth(width)));
    const ChunkSource src = source_of(sub, f, sfreq, t0);
    const std::int64_t R = src.rows();
    const std::int64_t k0 = span.s0 / width, k1 = std::min<std::int64_t>(count - 1, (span.s1 - 1) / width);
    const std::size_t W = static_cast<std::size_t>(count), RR = static_cast<std::size_t>(R);
    // the window minima and maxima as the Studio keeps them (float32); the sums folded across pieces in double, narrowed once at the end
    std::vector<float> mn(RR * W, std::numeric_limits<float>::infinity()), mx(RR * W, -std::numeric_limits<float>::infinity());
    std::vector<double> sm(RR * W, 0.0), sq(RR * W, 0.0), ct(RR * W, 0.0);
    say(Json{{"stage", "measure"}, {"field", fname}, {"over", member.name}, {"windows touched", Json::array({k0, k1})}, {"rows", R}, {"span", span.what}});
    // FOLD: each chunk split at window edges, each piece reduced by the Studio's reduceTiles (a row at a time), folded into its window
    chunk_loop(src, span, [&](const Chunk& c) {
      for (std::int64_t a = c.first; a < c.first + c.n;) {
        const std::int64_t k = a / width, b = std::min(c.first + c.n, (k + 1) * width), len = b - a;
        std::vector<float> piece(static_cast<std::size_t>(len));
        for (std::int64_t r = 0; r < R; ++r) {
          // the Studio's arrays are float32: the chunk service widened the stored float32 to double, so narrowing back is exact for a float32 Field
          const double* src_row = c.data.data() + r * c.n + (a - c.first);
          for (std::int64_t j = 0; j < len; ++j) piece[static_cast<std::size_t>(j)] = static_cast<float>(src_row[j]);
          const auto t = nxr::partition::reduceTiles(piece.data(), 1, static_cast<std::size_t>(len), static_cast<std::size_t>(len), 1);
          const std::size_t i = static_cast<std::size_t>(r) * W + static_cast<std::size_t>(k);
          mn[i] = std::min(mn[i], t.min[0]);
          mx[i] = std::max(mx[i], t.max[0]);
          sm[i] += t.sum[0];
          sq[i] += t.sumSq[0];
          ct[i] += t.count[0];
        }
        a = b;
      }
    });
    // THE ARRAY [rows x windows x components]: a window is measured only when the span covers it whole (a window cut by the span's edge is not the tiling's
    // window): NaN outside the span and for a window it cuts
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<float> arr(RR * W * C, nan);
    std::int64_t w0 = -1, w1 = -1;                                     // the windows measured whole
    for (std::size_t r = 0; r < RR; ++r)
      for (std::int64_t k = k0; k <= k1; ++k) {
        const std::size_t i = r * W + static_cast<std::size_t>(k);
        const double n = ct[i];
        if (!(n > 0) || n != static_cast<double>(std::min(width, samples - k * width))) continue;
        if (w0 < 0) w0 = k;
        w1 = k;
        for (std::size_t c = 0; c < C; ++c) {
          const std::string& m = comps[c];
          const float v = m == "mean" ? static_cast<float>(sm[i] / n) : m == "mean_square" ? static_cast<float>(sq[i] / n)
                        : m == "max" ? mx[i] : m == "min" ? mn[i] : m == "abs_max" ? std::max(std::fabs(mn[i]), std::fabs(mx[i])) : static_cast<float>(n);
          arr[(i)*C + c] = v;
        }
      }
    const std::string path = f.path.value_or(std::string()) + "_" + sanitized(member.name);
    const auto unit = f.unit;
    sub.compose([&] {
      if (have) subj.remove_measurements(member.id, fid);
      sub.write_array(path, view(arr, {R, static_cast<std::int64_t>(W), static_cast<std::int64_t>(C)}));
      Json rows = Json::array();
      for (std::size_t c = 0; c < C; ++c) {
        Json row{{"selection_id", member.id}, {"of_field_id", fid}, {"measure", comps[c]}, {"array_path", path}, {"component", static_cast<std::int64_t>(c)}, {"computed_by", by}};
        if (comps[c] == "mean_square" && unit) row["unit"] = "(" + *unit + ")^2";
        else if (comps[c] != "count" && unit) row["unit"] = *unit;
        rows.push_back(row);
      }
      subj.write_measurements(rows, by);
    });
    done.push_back(Json{{"field", fname}, {"array", path}, {"windows", w0 < 0 ? Json(nullptr) : Json::array({w0, w1})}, {"of", count}, {"components", comps}, {"span", span.what}, {"replaced", have}});
  }
  return {Json{{"over", member.name}, {"measured", done}}};
}

}  // namespace nxr::pipelines
