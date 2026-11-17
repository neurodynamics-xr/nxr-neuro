// nxr::selection (compute) — LABELLED SPANS: intervals [start, stop) on a line, each with a code read through a lookup. Unlike a partition they may leave gaps
// and overlap: events on the time line (bursts, spindles, slow waves, bad segments), bands on the frequency line (named bands), ranges of eigenvalues. Made
// in memory here; io stores them as a spans selection with its dictionary.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "nxr/selection/lookup.h"

namespace nxr::selection {

/** one interval [start, stop) in the line's unit, and its code; start == stop is an INSTANT (an event with no duration: Brainstorm's simple events) */
struct Span {
  double start = 0, stop = 0;
  std::int64_t code = 0;
};

struct Spans {
  std::string on;                  // the line's quantity: time · frequency · eigenvalue · …
  std::string unit;                // the unit of start and stop: s · Hz · 1/m^2 · …
  std::vector<Span> spans;         // sorted by start, then stop
  Lookup lookup;                   // the codes' names (every code a span carries has an entry, unless the lookup is empty)
};

/** labelled spans on the line `on` in `unit`: sorted; std::invalid_argument for a span with stop < start or not finite, or a code the lookup lacks */
Spans spans(std::string on, std::string unit, std::vector<Span> spans, Lookup lookup);

/** the indices (into `s.spans`) of the spans holding `value`: none, one, or several where they overlap */
std::vector<std::size_t> spans_at(const Spans& s, double value);

}  // namespace nxr::selection
