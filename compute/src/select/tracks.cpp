// nxr::select (tracks.h): nxr `_track_stats`'s linking.
#include "nxr/select/tracks.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace nxr::select {

Linker::Linker(double link) : link_(link) {
  if (!(link > 0) || !std::isfinite(link)) throw std::invalid_argument("Linker: the link distance is above 0");
}

void Linker::feed(const std::vector<std::int64_t>& frame, const std::vector<std::int8_t>& sense, const Eigen::Ref<const Eigen::MatrixXd>& centre) {
  if (finished_) throw std::invalid_argument("Linker: feed after finish");
  const std::size_t n = frame.size();
  if (sense.size() != n || static_cast<std::size_t>(centre.rows()) != n || (n > 0 && centre.cols() != 3)) throw std::invalid_argument("Linker: frame, sense and centre are for different numbers of cores");
  for (std::size_t i = 0; i < n; ++i) {
    if (sense[i] != 1 && sense[i] != -1) throw std::invalid_argument("Linker: a sense is +1 or -1");
    if (!centre.row(static_cast<Eigen::Index>(i)).allFinite()) throw std::invalid_argument("Linker: a centre is not finite");
    const std::int64_t before = frame_.empty() ? (i > 0 ? frame[i - 1] : std::numeric_limits<std::int64_t>::min()) : (i > 0 ? frame[i - 1] : frame_.back());
    if (frame[i] < before) throw std::invalid_argument("Linker: the frames go back (cores come by frame)");
  }
  for (std::size_t i = 0; i < n; ++i) {
    frame_.push_back(frame[i]); sense_.push_back(sense[i]);
    centre_.push_back(centre.row(static_cast<Eigen::Index>(i)).transpose());
  }
  track_.resize(frame_.size(), -1); prev_.resize(frame_.size(), -1);
  // every frame but the last one fed is complete (the next core has a later frame, or finish() says there is none)
  std::size_t from = linked_;
  while (from < frame_.size()) {
    std::size_t to = from;
    while (to < frame_.size() && frame_[to] == frame_[from]) ++to;
    if (to == frame_.size()) break;
    link_frame(from, to);
    from = to;
  }
}

void Linker::finish() {
  if (finished_) return;
  std::size_t from = linked_;
  if (from < frame_.size()) link_frame(from, frame_.size());
  finished_ = true;
}

void Linker::link_frame(std::size_t from, std::size_t to) {
  const std::int64_t k = frame_[from];
  if (!(have_prev_ && k == prev_frame_ + 1 && !active_.empty())) active_.clear();       // a gap breaks every track
  std::vector<Active> next;
  std::vector<std::uint8_t> used(active_.size(), 0);
  for (std::size_t q = from; q < to; ++q) {
    // the nearest unused active track of the same sense within `link` (the array's order breaks a tie, as a stable sort)
    std::vector<std::pair<double, std::size_t>> d;
    for (std::size_t u = 0; u < active_.size(); ++u) d.emplace_back(active_[u].sense == sense_[q] ? (active_[u].at - centre_[q]).norm() : std::numeric_limits<double>::infinity(), u);
    std::stable_sort(d.begin(), d.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    std::int64_t match = -1;
    for (const auto& [dist, u] : d) {
      if (dist > link_) break;
      if (!used[u]) { match = static_cast<std::int64_t>(u); break; }
    }
    if (match < 0) {
      track_[q] = static_cast<std::int32_t>(n_tracks_++);
      next.push_back(Active{track_[q], static_cast<std::int32_t>(q), sense_[q], centre_[q]});
    } else {
      used[static_cast<std::size_t>(match)] = 1;
      track_[q] = active_[static_cast<std::size_t>(match)].id;
      prev_[q] = active_[static_cast<std::size_t>(match)].core;
      next.push_back(Active{track_[q], static_cast<std::int32_t>(q), sense_[q], centre_[q]});
    }
  }
  active_ = std::move(next);
  prev_frame_ = k;
  have_prev_ = true;
  linked_ = to;
}

Linked link(const std::vector<std::int64_t>& frame, const std::vector<std::int8_t>& sense, const Eigen::Ref<const Eigen::MatrixXd>& centre, double linkDistance) {
  Linker l(linkDistance);
  l.feed(frame, sense, centre);
  l.finish();
  return Linked{l.track(), l.previous(), l.tracks()};
}

TrackStats trackStats(const std::vector<std::int64_t>& frame, const std::vector<std::int8_t>& sense, const Eigen::Ref<const Eigen::MatrixXd>& centre, const std::vector<std::int32_t>& track, const std::vector<std::int32_t>& previous) {
  const std::size_t n = frame.size();
  if (sense.size() != n || track.size() != n || previous.size() != n || static_cast<std::size_t>(centre.rows()) != n) throw std::invalid_argument("trackStats: the arrays are for different numbers of cores");
  std::int32_t nt = 0;
  for (const auto t : track) { if (t < 0) throw std::invalid_argument("trackStats: a core has no track"); nt = std::max<std::int32_t>(nt, t + 1); }
  TrackStats s;
  const std::size_t T = static_cast<std::size_t>(nt);
  s.first_core.assign(T, -1); s.last_core.assign(T, -1); s.first_frame.assign(T, 0); s.last_frame.assign(T, 0); s.n_frames.assign(T, 0);
  s.sense.assign(T, 0); s.path.assign(T, 0.0); s.displacement.assign(T, 0.0);
  for (std::size_t i = 0; i < n; ++i) {                                                   // the cores are in order, so a track's first core is its first met
    const std::size_t t = static_cast<std::size_t>(track[i]);
    if (s.first_core[t] < 0) { s.first_core[t] = static_cast<std::int32_t>(i); s.first_frame[t] = frame[i]; s.sense[t] = sense[i]; }
    s.last_core[t] = static_cast<std::int32_t>(i); s.last_frame[t] = frame[i]; ++s.n_frames[t];
    if (previous[i] >= 0) s.path[t] += (centre.row(static_cast<Eigen::Index>(i)) - centre.row(previous[i])).norm();
  }
  for (std::size_t t = 0; t < T; ++t) s.displacement[t] = (centre.row(s.last_core[t]) - centre.row(s.first_core[t])).norm();
  return s;
}

}  // namespace nxr::select
