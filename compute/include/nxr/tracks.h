#pragma once
// nxr::tracks — the TRACKS of vortex cores (spec docs/specs/2026-11-14-vortex-cores-port.md, piece 6; family C, a feature followed in time). PURE MATHS on arrays: cores in (their frame, sense and centre), track labels out.
// The function never knows where the cores came from (a recording, a stored Selection, a synthetic list) or what unit the centres are in (the link distance is in that unit). A port of nxr's `_track_stats`
// (nxr/detect/vortex.py), the reference, on the linking only:
//
//   link      each core of a frame is matched, in the order it is given (by sense, +1 first, then its place in the frame), to the nearest ACTIVE track of the PREVIOUS frame of the SAME sense, within `link`
//             of the track's last centre, a track taking at most one core a frame (greedy: the first core in the order gets the nearest). A core that matches none starts a track.
//   frames    CONSECUTIVE frames only: a core's frame must be the previous cores' frame + 1; any gap (a frame without a core, a bad segment) closes every track.
//   order     tracks are numbered by the order they start (track 0 first): the frame of their first core, then the core's place in the frame. Every core belongs to one, a lone core to a track of one frame.
//   streaming Linker::feed takes the cores in blocks of any size (a frame may be split between two blocks): the open tracks are carried, and the result does not depend on the cut (tested).
// These are DESCRIPTIONS of where a core went, never evidence that anything travels: the speed and lifetime of the tracks equal the lag-free null's (the population work, 2026-10-30).
#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace nxr::tracks {

class Linker {
 public:
  /** `link` > 0, in the unit of the centres. std::invalid_argument otherwise. */
  explicit Linker(double link);
  /** the next cores: `frame` non-decreasing across the whole stream, `sense` ±1, `centre` [n x 3]. std::invalid_argument for sizes that disagree, a frame that goes back, a sense that is not ±1, or a non-finite centre. */
  void feed(const std::vector<std::int64_t>& frame, const std::vector<std::int8_t>& sense, const Eigen::Ref<const Eigen::MatrixXd>& centre);
  /** ends the stream: the frame still waiting for a next block is linked. Feeding after it is refused. */
  void finish();
  /** per core, in the order fed (after finish): its track's number, and the core before it in its track (-1: it starts the track) */
  const std::vector<std::int32_t>& track() const { return track_; }
  const std::vector<std::int32_t>& previous() const { return prev_; }
  std::int64_t tracks() const { return n_tracks_; }

 private:
  struct Active { std::int32_t id, core; std::int8_t sense; Eigen::Vector3d at; };
  void link_frame(std::size_t from, std::size_t to);
  double link_;
  bool finished_ = false;
  std::vector<std::int64_t> frame_;
  std::vector<std::int8_t> sense_;
  std::vector<Eigen::Vector3d> centre_;
  std::vector<std::int32_t> track_, prev_;
  std::size_t linked_ = 0;                 // the cores [0, linked_) have their track
  bool have_prev_ = false;
  std::int64_t prev_frame_ = 0;
  std::vector<Active> active_;
  std::int64_t n_tracks_ = 0;
};

/** a whole list at once (the Linker fed one block) */
struct Linked { std::vector<std::int32_t> track, previous; std::int64_t tracks = 0; };
Linked link(const std::vector<std::int64_t>& frame, const std::vector<std::int8_t>& sense, const Eigen::Ref<const Eigen::MatrixXd>& centre, double linkDistance);

/** what each track is, from the labels: first and last core (indices), first and last frame, frames held (cores), path length (Σ the steps' lengths) and displacement (|last centre − first centre|), in the centres' unit.
 *  A track of one frame has path 0 and displacement 0. */
struct TrackStats {
  std::vector<std::int32_t> first_core, last_core;
  std::vector<std::int64_t> first_frame, last_frame, n_frames;
  std::vector<std::int8_t> sense;
  std::vector<double> path, displacement;
  std::int64_t size() const { return static_cast<std::int64_t>(first_core.size()); }
};
TrackStats trackStats(const std::vector<std::int64_t>& frame, const std::vector<std::int8_t>& sense, const Eigen::Ref<const Eigen::MatrixXd>& centre, const std::vector<std::int32_t>& track, const std::vector<std::int32_t>& previous);

}  // namespace nxr::tracks
