// nxr-graph — the SIGNATURES of the registered nodes the shipped definitions use (signature.h; spec docs/specs/2026-11-17-type-system.md, sections 4, 5 and 11).
// Each node is one move and its ports' patterns; `refine` states what the patterns cannot (a parameter that picks a band, a level, a form; what a quantity fits).
// The type checks that were in a node's `plan` (which stays for now) are here once; parameter and array-shape checks stay in the nodes.
// A port's name and order are the node's own (pipelines/src/*.cpp); a test (test_definitions_typed) compares them with the registered nodes.
#include <algorithm>
#include <array>
#include <cstdio>
#include <set>

#include "nxr/graph/signature.h"

namespace nxr::graph {

namespace {

using VT = ValueType;

const ManifoldRef M = var("M"), A = var("A"), B = var("B"), R = var_rest("R"), T = time_var("T"), TF = frames_var("T");

std::optional<std::array<double, 2>> band_of(const Json& j) {
  if (j.is_array() && j.size() == 2 && j[0].is_number() && j[1].is_number()) return std::array<double, 2>{j[0].get<double>(), j[1].get<double>()};
  return std::nullopt;
}

std::string id_of(const Unified& u, const std::string& port) {
  const PortType& t = u.in.at(port);
  if (t.sel && !t.sel->identity.empty()) return t.sel->identity;
  const auto nm = u.in_names.find(port);
  return nm == u.in_names.end() ? port : nm->second;
}

ManifoldRef with_of(const ManifoldRef& m, const ManifoldRef& of) { Manifold x = *m; x.of = of; return std::make_shared<const Manifold>(std::move(x)); }
ManifoldRef strip_glued(const ManifoldRef& m) { return m && m->form == Manifold::Form::derived && m->key == "glued" ? m->of : m; }
bool is_members(const ManifoldRef& m) { return m && m->form == Manifold::Form::members; }
// a member set of a partition (a measurement's factor), or the time tiles a node made of one: what a time event is found on
bool member_like(const ManifoldRef& m) { return m && (m->form == Manifold::Form::members || m->role == Role::time_tiles || m->role == Role::frames); }
std::string partition_id(const ManifoldRef& m) { return m->form == Manifold::Form::members ? m->by : m->key; }
bool is_joint_members(const ManifoldRef& m) { return is_members(m) && m->label == "joint"; }
bool is_surface(const ManifoldRef& m) { return m && m->kind == ManifoldKind::surface; }
bool over_members(const PortType& t, const std::string& id) { return t.domain.factors.size() == 1 && is_members(t.domain.factors[0]) && t.domain.factors[0]->by == id; }
bool is_complex(const PortType& t) { return t.value == VT::complex || t.complex_parts; }
Domain dom_of(std::vector<ManifoldRef> f) { Domain d; d.factors = std::move(f); return d; }
Domain without_last(const Domain& d) { Domain x = d; if (!x.factors.empty()) x.factors.pop_back(); return x; }
std::string level_text(const Resolution& r) {
  std::string s = "level ";
  for (std::size_t i = 0; i < r.levels.size(); ++i) s += (i ? "," : "") + std::to_string(r.levels[i]);
  return s + " of `" + r.ladder + "`";
}

// ── events: a selection and the measurements over its members (an event table)
void events_out(Unified& u, const PortType& x, const std::string& sel_port, const std::vector<std::string>& measures, const ManifoldRef& line) {
  PortType& e = u.out.at(sel_port);
  e.domain = x.domain;
  e.quantity = "spans";
  e.sel = SelectionShape{"spans", true, u.out_names.at(sel_port), ""};
  for (const auto& m : measures) {
    PortType& o = u.out.at(m);
    o.domain = dom_of({members(e.sel->identity, line)});
    o.quantity = m;
  }
}
void need_over(Unified& u, const std::string& port, const std::string& id, const std::string& what) {
  if (!u.has(port)) return;
  const PortType& t = u.in.at(port);
  if (t.cls != PortClass::measurement || !over_members(t, id))
    u.fail(port, "must be " + what + " of the events `" + id + "` (a measurement over their members)");
}

Signature s_analytic() {
  return sig(Move::pointwise)
      .in("x", field({M, T}).streamed().value(VT::real).because("The analytic signal is taken of a real field."))
      .out("y", like("x").with_value(VT::complex))
      .refine([](Unified& u, const Json& p) { if (auto b = band_of(p.value("band", Json()))) u.out.at("y").res.band = b; });
}

Signature s_envelope() {
  return sig(Move::pointwise)
      .in("x", field({M, T}).streamed().value(VT::real).because("The envelope is taken of the real field."))
      .out("y", like("x").with_quantity("envelope").non_linear())
      .refine([](Unified& u, const Json& p) { if (auto b = band_of(p.value("band", Json()))) u.out.at("y").res.band = b; });
}

Signature s_frames() {
  return sig(Move::apply)
      .in("x", field({M, T}).streamed().linear().because("The frames interpolate the band-limited signal; after a non-linear node its band limit is gone."))
      .in("tiling", selection({T}).because("The frame tiling is made for x's time line."))
      .in("within", selection({T}).optional())
      .out("y", like("x").on({M, members_of_port("tiling")}))
      .refine([](Unified& u, const Json&) {
        const PortType& t = u.type("tiling");
        if (!t.sel || t.sel->type != "tiling") u.fail("tiling", "is a " + (t.sel ? t.sel->type : std::string("?")) + " selection, not a frame tiling (a closed-form tiling of the time line)");
        PortType& y = u.out.at("y");
        y.domain.factors.back() = members(id_of(u, "tiling"), u.type("x").domain.factors.back(), Role::frames);
        y.axis = Axis::frames;
      });
}

Signature s_apply_kernel() {
  return sig(Move::apply)
      .in("x", field({A, R, TF}).streamed().linear())
      .in("kernel", operator_(from_or_restriction_of(A), B).because("The kernel reads the channels of x; make the cross-spectrum or the recording with `--channels-of <kernel>`."))
      .in("tiles", selection({B}).optional().because("The tile ladder must be the ladder of the kernel's target surface."))
      .out("y", like("x").on({B, R, TF}))
      .refine([](Unified& u, const Json& p) {
        const PortType& x = u.type("x");
        const PortType& k = u.type("kernel");
        PortType& y = u.out.at("y");
        const std::string form = p.value("form", std::string("none"));
        if (p.value("glued", false)) y.domain.factors[0] = derived("glued", y.domain.factors[0]);
        y.res.ladder.clear(); y.res.levels.clear(); y.res.spatial_shape.clear();
        if (u.has("tiles")) {
          const PortType& t = u.in.at("tiles");
          if (!t.sel || t.sel->type != "ladder") u.fail("tiles", "is a " + (t.sel ? t.sel->type : std::string("?")) + " selection, not a tile ladder");
          std::vector<int> lv;
          if (p.contains("levels") && p["levels"].is_array()) for (const auto& l : p["levels"]) lv.push_back(l.get<int>());
          else if (p.contains("level") && p["level"].is_number()) lv.push_back(p["level"].get<int>());
          if (lv.empty()) u.fail("tiles", "`tiles` comes with `level` or `levels`");
          y.res.ladder = id_of(u, "tiles");
          y.res.levels = lv;
          y.res.spatial_shape = p.value("spatial_shape", std::string("mexhat"));
        }
        VT vt = k.value;
        if (form != "none") vt = VT::real;
        if (p.value("envelope", std::string("temporal")) == "spatiotemporal") vt = VT::monogenic;
        if (is_complex(x) && vt == VT::vector3) { y.value = VT::vector3; y.complex_parts = true; }
        else if (is_complex(x) && vt == VT::real) { y.value = VT::complex; y.complex_parts = false; }
        else { y.value = vt; y.complex_parts = false; }
        y.quantity = form == "none" ? "current" : form;
      });
}

Signature s_magnitude() {
  return sig(Move::pointwise).in("x", field({M, TF}).streamed()).out("y", like("x").with_value(VT::real).with_quantity("amplitude").non_linear());
}

Signature s_power() {
  return sig(Move::pointwise)
      .in("x", field({M, TF}).streamed())
      .out("y", like("x").with_value(VT::real).with_quantity("power").non_linear())
      .refine([](Unified& u, const Json&) {
        if (!is_complex(u.type("x"))) u.fail("x", "is real; the band power |z|^2 is taken of the analytic signal's map: band-limit it with `analytic` first");
      });
}

Signature s_phase_current() {
  return sig(Move::pointwise)
      .in("x", field({M, TF}).streamed().linear())
      .out("y", like("x").with_value(VT::vector3).with_quantity("phase_current").non_linear())
      .refine([](Unified& u, const Json&) {
        const PortType& x = u.type("x");
        if (!is_complex(x)) u.fail("x", "must be complex (a phase is kept)");
        if (!is_surface(x.domain.factors.front())) u.fail("x", "its first factor " + print(x.domain.factors.front()) + " is not a surface; the phase current is a tangent vector of a surface");
      });
}

Signature s_measure() {
  return sig(Move::measure).accumulating()
      .in("x", field({R, TF}).streamed().value_in({VT::real, VT::vector3}).because("Measure a real quantity of it (power, a form, the phase current)."))
      .in("exclude", selection({TF}).optional())
      .in("over", selection_any().optional().over())
      .out("y", make_of(PortClass::measurement))
      .refine([](Unified& u, const Json& p) {
        const PortType& x = u.type("x");
        const std::string m = p.value("measure", std::string("mean"));
        static const std::map<std::string, std::vector<std::string>> fits = {{"amplitude", {"amplitude", "envelope"}}, {"envelope_max", {"amplitude", "envelope"}},
                                                                             {"median", {"envelope"}}, {"power", {"power"}}, {"phase_current", {"phase_current"}}};
        const auto f = fits.find(m);
        if (f != fits.end() && std::find(f->second.begin(), f->second.end(), x.quantity) == f->second.end()) {
          std::string want;
          for (const auto& q : f->second) want += (want.empty() ? "" : " or ") + q;
          u.fail("x", "is " + x.quantity + "; the measure " + m + " is a measure of " + want);
        }
        PortType& y = u.out.at("y");
        y.quantity = m;
        y.value = m == "phase_current" && u.has("over") && u.in.at("over").sel && u.in.at("over").sel->type != "ladder" ? VT::vector3 : VT::real;
      });
}

Signature s_tile_max() {
  return sig(Move::measure)
      .in("x", field({A, TF}).streamed().quantity({"amplitude"}).because("|z| per vertex (`magnitude`) is the trace's input."))
      .in("tiles", selection({A}).because("The tiles are the tile ladder of x's surface."))
      .out("y", make_of(PortClass::measurement).with_quantity("trace"))
      .refine([](Unified& u, const Json& p) {
        const PortType& x = u.type("x");
        const PortType& t = u.type("tiles");
        if (!t.sel || t.sel->type != "ladder") u.fail("tiles", "is a " + (t.sel ? t.sel->type : std::string("?")) + " selection, not a tile ladder");
        if (!p.contains("level") || !p["level"].is_number()) u.fail("tiles", "`level` names the tile level");
        const int level = p["level"].get<int>();
        if (x.res.ladder != id_of(u, "tiles") || x.res.levels != std::vector<int>{level})
          u.fail("tiles", "the tiles' level " + std::to_string(level) + " of `" + id_of(u, "tiles") + "` is not x's spatial band (" + level_text(x.res) + "); the level IS the band");
        PortType& y = u.out.at("y");
        const std::string key = id_of(u, "tiles") + "@" + std::to_string(level) + "," + p.value("groups", std::string("whole_brain"));
        y.domain = dom_of({made("groups", key, ManifoldKind::members), x.domain.factors.back()});
        y.res = x.res;
        y.axis = x.axis;
        y.sfreq = x.sfreq;
      });
}

Signature s_collect() {
  // the samples of the run kept in order: a streamed field, or a streamed measurement (`tile_max`'s trace), becomes an array: a field on the same domain
  return sig(Move::pointwise).accumulating()
      .in("x", any_port())
      .out("y", like("x").as(PortClass::field).non_linear())
      .refine([](Unified& u, const Json&) {
        const PortType& x = u.type("x");
        if (!(x.cls == PortClass::field || x.cls == PortClass::measurement) || !x.domain.streamed()) u.fail("x", "must be a series on a time line (a field or a measurement of the samples)");
        if (x.value != VT::real || x.complex_parts) u.fail("x", "is not real; collect a real series");
        PortType& y = u.out.at("y");
        y.axis = Axis::none;
        y.sfreq.reset();
      });
}

Signature s_reduce_cells() {
  return sig(Move::summarize)
      .in("x", measurement_any().value(VT::real).because("The maximum over the spatial tiles of each time tile is of a scalar."))
      .out("y", make_of(PortClass::measurement))
      .refine([](Unified& u, const Json& p) {
        const PortType& x = u.type("x");
        if (x.domain.factors.size() != 1 || !is_joint_members(x.domain.factors[0]))
          u.fail("x", "must be a measurement over a joint partition's cells (one member set of space x time)");
        const ManifoldRef& c = x.domain.factors[0];
        PortType& y = u.out.at("y");
        y.domain = dom_of({made("groups", c->by + "," + p.value("groups", std::string("whole_brain")), ManifoldKind::members),
                           with_of(made("time_tiles", c->by, ManifoldKind::line, Role::time_tiles), c->of)});
        y.quantity = x.quantity;
        y.res = x.res;
      });
}

Signature s_summarize() {
  return sig(Move::summarize)
      .in("x", measurement_any().value(VT::real).because("A reduction is a measurement over a partition: measure the field over a partition first, then summarize its members."))
      .out("y", make_of(PortClass::statistic).with_quantity("summaries"))
      .refine([](Unified& u, const Json& p) {
        const PortType& x = u.type("x");
        PortType& y = u.out.at("y");
        y.domain = dom_of({});
        for (std::size_t i = 0; i < x.domain.factors.size(); ++i) {
          const ManifoldRef& f = x.domain.factors[i];
          const bool timelike = f->role == Role::time_tiles || f->role == Role::frames;
          const bool last_members = is_members(f) && i + 1 == x.domain.factors.size();
          if (!timelike && !last_members) y.domain.factors.push_back(f);
        }
        if (p.contains("summaries")) y.facts["summaries"] = p["summaries"];
      });
}

Signature s_detect_spans() {
  return sig(Move::detect)
      .in("x", measurement_any().value(VT::real).because("Events are detected on a measurement over a time partition's members: measure it over the constant-Q level first (measure with `over`)."))
      .in("baseline", statistic_any().optional())
      .out("spans", make_of(PortClass::selection))
      .refine([](Unified& u, const Json&) {
        const PortType& x = u.type("x");
        if (x.domain.factors.empty()) u.fail("x", "has no factor");
        const ManifoldRef& last = x.domain.factors.back();
        if (!(last->role == Role::time_tiles || last->role == Role::frames) || !last->of)
          u.fail("x", "its last factor " + print(last) + " is not the member set of a time partition; events are found on a time line. Measure x over a time partition first.");
        if (u.has("baseline")) {
          const PortType& b = u.in.at("baseline");
          if (!b.of || b.of->quantity != x.quantity || !same(b.of->on, x.domain)) u.fail("baseline", "is not a statistic of x (a baseline is summarized from the same measurement)");
        }
        PortType& s = u.out.at("spans");
        s.domain = dom_of({last->of});
        s.quantity = "spans";
        s.sel = SelectionShape{"spans", true, u.out_names.at("spans"), partition_id(last)};
        s.res.band = x.res.band;
      });
}

Signature s_rollup_events() {
  return sig(Move::measure)
      .in("x", measurement_any().because("The roll-up re-measures a measurement over a tile level."))
      .in("events", selection_any())
      .out("y", make_of(PortClass::measurement))
      .refine([](Unified& u, const Json& p) {
        const PortType& x = u.type("x");
        const PortType& ev = u.type("events");
        if (x.domain.factors.empty() || !member_like(x.domain.factors.back())) u.fail("x", "must be a measurement over a partition's members (a tile level)");
        if (!ev.sel || ev.sel->type != "spans") u.fail("events", "must be a spans selection");
        const std::string over = partition_id(x.domain.factors.back());
        if (!ev.sel->detected_over.empty() && ev.sel->detected_over != over)
          u.fail("events", "were detected on the members of `" + ev.sel->detected_over + "`; x is measured over `" + over + "`. Roll up a measurement over the partition the events were found on (or detect them on x's).");
        const std::string m = p.value("measure", std::string());
        if (m.empty()) u.fail("x", "the roll-up needs a `measure`");
        if (m != "duration" && m != "interval_next" && m != x.quantity) u.fail("x", "the measure " + m + " is not x's quantity (" + x.quantity + ")");
        PortType& y = u.out.at("y");
        y.domain = x.domain;
        y.domain.factors.back() = members(id_of(u, "events"), ev.domain.factors.empty() ? ManifoldRef() : ev.domain.factors.back());
        y.quantity = m;
        y.value = x.value;
        y.complex_parts = x.complex_parts;
        y.res = x.res;
      });
}

Signature s_risefall() {
  Signature s = sig(Move::measure).in("x", any_port()).in("events", selection_any());
  const std::vector<std::string> outs = {"t_on", "t_off", "tau_rise", "tau_fall", "fit_baseline", "fit_height", "fit_r2", "plateau_duration"};
  for (const auto& o : outs) s.out(o, make_of(PortClass::measurement).with_quantity(o));
  return s.refine([outs](Unified& u, const Json&) {
    const PortType& x = u.type("x");
    const PortType& ev = u.type("events");
    if (!(x.cls == PortClass::measurement || (x.cls == PortClass::field && !x.streamed()))) u.fail("x", "must be the sample series of the run (`collect`)");
    if (!ev.sel || ev.sel->type != "spans" || ev.sel->detected_over.empty()) u.fail("events", "must be the spans `detect_spans` made");
    for (const auto& o : outs) u.out.at(o).domain = dom_of({members(id_of(u, "events"), ev.domain.factors.empty() ? ManifoldRef() : ev.domain.factors.back())});
  });
}

Signature s_fit_gate() {
  return sig(Move::detect)
      .in("events", selection_any())
      .in("fit_r2", measurement_any())
      .out("spans", like("events"))
      .refine([](Unified& u, const Json&) {
        const PortType& ev = u.type("events");
        const PortType& r2 = u.type("fit_r2");
        if (r2.quantity != "fit_r2" || !over_members(r2, id_of(u, "events"))) u.fail("fit_r2", "must be `risefall`'s fit_r2 of these events");
        u.out.at("spans").sel->identity = u.out_names.at("spans");
        (void)ev;
      });
}

Signature s_burst_stages() {
  return sig(Move::detect)
      .in("events", selection_any())
      .in("t_on", measurement_any()).in("t_off", measurement_any()).in("tau_rise", measurement_any()).in("tau_fall", measurement_any())
      .in("kept", selection_any().optional())
      .out("onset", make_of(PortClass::selection)).out("within", make_of(PortClass::selection)).out("offset", make_of(PortClass::selection))
      .refine([](Unified& u, const Json&) {
        const PortType& ev = u.type("events");
        for (const char* m : {"t_on", "t_off", "tau_rise", "tau_fall"}) need_over(u, m, id_of(u, "events"), std::string("`risefall`'s ") + m);
        for (const char* o : {"onset", "within", "offset"}) {
          PortType& s = u.out.at(o);
          s.domain = ev.domain;
          s.quantity = "spans";
          s.sel = SelectionShape{"spans", false, u.out_names.at(o), ""};
        }
      });
}

Signature s_cell_extent() {
  return sig(Move::detect)
      .in("x", measurement_any().because("The extent is found on a measurement over a joint partition."))
      .in("baseline", statistic_any())
      .in("events", selection_any())
      .out("labels", make_of(PortClass::selection))
      .refine([](Unified& u, const Json&) {
        const PortType& x = u.type("x");
        const PortType& b = u.type("baseline");
        const PortType& ev = u.type("events");
        if (x.domain.factors.empty() || !is_joint_members(x.domain.factors[0])) u.fail("x", "must be a measurement over a joint partition");
        if (!b.of || !same(b.of->on, x.domain)) u.fail("baseline", "is not a statistic of x");
        if (!ev.sel || ev.sel->type != "spans") u.fail("events", "must be spans");
        PortType& l = u.out.at("labels");
        l.domain = x.domain;
        l.quantity = "set";
        l.sel = SelectionShape{"set", true, u.out_names.at("labels"), x.domain.factors[0]->by};
      });
}

Signature s_extent_measure() {
  return sig(Move::measure)
      .in("labels", selection_any())
      .in("x", measurement_any().optional())
      .out("y", make_of(PortClass::measurement))
      .refine([](Unified& u, const Json& p) {
        const PortType& l = u.type("labels");
        if (!l.sel || l.sel->type != "set" || l.domain.factors.empty() || !is_joint_members(l.domain.factors[0])) u.fail("labels", "must be the labelling `cell_extent` made");
        if (u.has("x") && !same(u.in.at("x").domain, l.domain)) u.fail("x", "its cells are not the labelling's");
        PortType& y = u.out.at("y");
        y.domain = dom_of({members(id_of(u, "labels"), l.domain.factors[0]->of)});
        y.quantity = p.value("measure", std::string());
      });
}

Signature s_relative() {
  return sig(Move::pointwise)
      .in("x", field({M, TF}).streamed().quantity({"power"}).because("A band's power is divided by its baseline."))
      .in("baseline", statistic_any())
      .out("y", like("x").with_quantity("relative power"))
      .refine([](Unified& u, const Json&) {
        if (u.type("baseline").quantity != "mean") u.fail("baseline", "must be the spatial baseline (a statistic: the mean)");
      });
}

Signature s_at_level() {
  return sig(Move::restrict)
      .in("x", field({M, TF}).streamed())
      .out("y", like("x"))
      .refine([](Unified& u, const Json& p) {
        const PortType& x = u.type("x");
        if (x.res.levels.size() < 2) u.fail("x", "is not a multi-level field (`apply_kernel` with `levels`); its spatial band is " + level_text(x.res));
        u.out.at("y").res.levels = {p.value("level", 0)};
      });
}

Signature s_scale() {
  return sig(Move::pointwise)
      .in("x", field({M, TF}).streamed())
      .out("y", like("x").with_quantity("characteristic level"))
      .refine([](Unified& u, const Json&) {
        const PortType& x = u.type("x");
        if (x.res.levels.size() < 2) u.fail("x", "carries no stack of spatial levels");
        const ManifoldRef& f = x.domain.factors.front();
        if (!(f->form == Manifold::Form::derived && f->key == "glued")) u.fail("x", "must be on the glued surface (`apply_kernel` with `glued`)");
        PortType& y = u.out.at("y");
        y.res.ladder.clear();
        y.res.levels.clear();
      });
}

// a fused statistic estimation: a measure over a declared partition of the stream, then summarize (spec section 5)
void statistic_of_stream(Unified& u, const std::string& port, const std::string& statistic_quantity, const ManifoldRef& axis_override = nullptr) {
  const PortType& x = u.type("x");
  PortType& y = u.out.at(port);
  y.quantity = statistic_quantity;
  y.domain = axis_override ? dom_of({axis_override}) : without_last(x.domain);
  y.of = OfType{x.quantity, x.res, without_last(x.domain)};
}

Signature s_baseline() {
  return sig(Move::summarize).accumulating().reduces_over("the good frames")
      .in("x", field({R, TF}).streamed().quantity({"power"}).because("The baseline is of a band's power."))
      .in("exclude", selection_any().optional()).in("outside", selection_any().optional())
      .out("y", make_of(PortClass::statistic))
      .refine([](Unified& u, const Json&) { statistic_of_stream(u, "y", "mean"); });
}

Signature s_relative_histogram() {
  return sig(Move::summarize).reduces_over("the good frames outside the bursts")
      .in("x", field({R, TF}).streamed().quantity({"relative power"}))
      .in("exclude", selection_any().optional()).in("outside", selection_any())
      .out("y", make_of(PortClass::statistic))
      .refine([](Unified& u, const Json&) { statistic_of_stream(u, "y", "histogram", made("bins", u.out_names.at("y") + ".bins", ManifoldKind::points)); });
}

Signature s_core_histogram() {
  return sig(Move::summarize).accumulating().reduces_over("the good frames")
      .in("x", field({R, TF}).streamed())
      .in("exclude", selection_any().optional())
      .out("y", make_of(PortClass::statistic))
      .refine([](Unified& u, const Json&) { statistic_of_stream(u, "y", "histogram", made("bins", u.out_names.at("y") + ".bins", ManifoldKind::points)); });
}

Signature s_cores() {
  const std::vector<std::string> ms = {"frame", "sense", "centre_vertex", "centre", "area", "peak"};
  Signature s = sig(Move::detect).accumulating()
      .in("x", field({R, TF}).streamed().quantity({"curl_tan"}).because("The cores are found on the tangential curl (`apply_kernel` with form curl_tan)."))
      .in("threshold", statistic_any()).in("exclude", selection_any().optional())
      .out("spans", make_of(PortClass::selection));
  for (const auto& m : ms) s.out(m, make_of(PortClass::measurement));
  return s.refine([ms](Unified& u, const Json&) { const PortType& x = u.type("x"); events_out(u, x, "spans", ms, x.domain.factors.back()); });
}

Signature s_patches() {
  const std::vector<std::string> ms = {"frame", "burst", "centre_vertex", "centre", "area", "diameter_cm", "characteristic_level", "size_cm", "relative_peak"};
  Signature s = sig(Move::detect).accumulating()
      .in("x", field({R, TF}).streamed().quantity({"relative power"}).because("Patches are found on a relative power map."))
      .in("scale", field({var_rest("R2"), TF}).because("The scale is on x's frames."))
      .in("bursts", selection_any().optional()).in("outside", selection_any().optional()).in("exclude", selection_any().optional()).in("threshold", statistic_any().optional())
      .out("spans", make_of(PortClass::selection));
  for (const auto& m : ms) s.out(m, make_of(PortClass::measurement));
  s.out("home", make_of(PortClass::statistic)).out("summary", make_of(PortClass::statistic));
  return s.refine([ms](Unified& u, const Json&) {
    const PortType& x = u.type("x");
    events_out(u, x, "spans", ms, x.domain.factors.back());
    PortType& h = u.out.at("home");
    h.quantity = "home";
    h.domain = without_last(x.domain);
    h.of = OfType{x.quantity, x.res, without_last(x.domain)};
    PortType& sm = u.out.at("summary");
    sm.quantity = "summary";
    sm.domain = dom_of({});
  });
}

Signature s_tracks() {
  const std::vector<std::string> ms = {"first_frame", "last_frame", "n_frames", "path_length", "displacement", "sense", "track"};
  Signature s = sig(Move::detect)
      .in("events", selection_any())
      .in("frame", measurement_any()).in("sense", measurement_any()).in("centre", measurement_any())
      .out("spans", make_of(PortClass::selection));
  for (const auto& m : ms) s.out(m, make_of(PortClass::measurement).with_quantity(m));
  return s.refine([ms](Unified& u, const Json&) {
    const PortType& ev = u.type("events");
    const std::string eid = id_of(u, "events");
    for (const char* m : {"frame", "sense", "centre"}) need_over(u, m, eid, std::string("a measurement"));
    const ManifoldRef line = ev.domain.factors.empty() ? ManifoldRef() : ev.domain.factors.back();
    PortType& tr = u.out.at("spans");
    tr.domain = dom_of({members(eid, line)});
    tr.quantity = "set";
    tr.sel = SelectionShape{"set", true, u.out_names.at("spans"), ""};
    for (const auto& m : ms) u.out.at(m).domain = dom_of({m == "track" ? members(eid, line) : members(tr.sel->identity, line)});
  });
}

Signature s_tile_crossings() {
  return sig(Move::measure)
      .in("events", selection_any())
      .in("frame", measurement_any()).in("sense", measurement_any()).in("centre", measurement_any()).in("track", measurement_any())
      .in("tiles", selection_any())
      .in("onset", selection_any().optional()).in("within", selection_any().optional()).in("offset", selection_any().optional())
      .out("crossings", make_of(PortClass::measurement).with_quantity("crossings"))
      .refine([](Unified& u, const Json& p) {
        const PortType& ev = u.type("events");
        const PortType& t = u.type("tiles");
        if (!t.sel || t.sel->type != "ladder") u.fail("tiles", "must be a tile ladder");
        if (ev.domain.factors.empty() || t.domain.factors.empty() || !same(strip_glued(ev.domain.factors[0]), t.domain.factors[0]))
          u.fail("tiles", "the ladder is on " + print(t.domain) + "; the events lie on " + print(ev.domain) + ": the tile ladder of the events' surface is wanted");
        if (!p.contains("level") || !p["level"].is_number()) u.fail("tiles", "`level` names the tile level");
        const std::string lvl = id_of(u, "tiles") + "@" + std::to_string(p["level"].get<int>());
        PortType& y = u.out.at("crossings");
        y.domain = dom_of({members(lvl, t.domain.factors[0], Role::none)});
        if (p.value("as", std::string("measurement")) == "statistic") y.cls = PortClass::statistic;   // a null draw's crossings: a statistic of its own, never the data's rows
      });
}

Signature s_power_series() {
  return sig(Move::summarize).accumulating().reduces_over("the dyadic time tiles")
      .in("x", field({R, T}).streamed().value(VT::complex).because("The band power is taken of the analytic signal."))
      .in("cross_spectrum", statistic_any()).in("exclude", selection_any().optional())
      .out("series", make_of(PortClass::statistic))
      .refine([](Unified& u, const Json&) {
        const PortType& x = u.type("x");
        PortType& y = u.out.at("series");
        y.quantity = "power_tiles";
        y.domain = dom_of({made("tiles", u.out_names.at("series") + ".tiles", ManifoldKind::members)});
        y.of = OfType{"power", Resolution{x.res.band, x.res.band_shape, "", {}, ""}, without_last(x.domain)};
      });
}

Signature s_lagfree_surrogate() {
  // a surrogate recording drawn on the cross-spectrum's channels: the first factor is replaced by theirs (one factor changes, the others are carried)
  return sig(Move::apply)
      .in("x", field({A, R, T}).streamed())
      .in("psd", statistic_any()).in("cross_spectrum", statistic_any()).in("power_series", statistic_any())
      .out("y", like("x").with_quantity("surrogate"))
      .refine([](Unified& u, const Json&) {
        const PortType& x = u.type("x");
        const PortType& cs = u.type("cross_spectrum");
        if (cs.domain.factors.empty()) u.fail("cross_spectrum", "has no channels axis");
        const ManifoldRef& ch = cs.domain.factors[0];
        const ManifoldRef base = ch->form == Manifold::Form::restricted ? ch->of : ch;
        if (!same(base, x.domain.factors[0]))
          u.fail("cross_spectrum", "its channels " + print(ch) + " are not channels of " + print(x.domain.factors[0]) + ": the surrogate is drawn on the cross-spectrum's channels, of x's sensors");
        PortType& y = u.out.at("y");
        y.domain.factors[0] = ch;
      });
}

const std::map<std::string, Signature>& table() {
  static const std::map<std::string, Signature> t = [] {
    std::map<std::string, Signature> m;
    m["analytic"] = s_analytic();
    m["envelope"] = s_envelope();
    m["frames"] = s_frames();
    m["apply_kernel"] = s_apply_kernel();
    m["magnitude"] = s_magnitude();
    m["power"] = s_power();
    m["phase_current"] = s_phase_current();
    m["measure"] = s_measure();
    m["tile_max"] = s_tile_max();
    m["collect"] = s_collect();
    m["reduce_cells"] = s_reduce_cells();
    m["summarize"] = s_summarize();
    m["detect_spans"] = s_detect_spans();
    m["rollup_events"] = s_rollup_events();
    m["risefall"] = s_risefall();
    m["fit_gate"] = s_fit_gate();
    m["burst_stages"] = s_burst_stages();
    m["cell_extent"] = s_cell_extent();
    m["extent_measure"] = s_extent_measure();
    m["relative"] = s_relative();
    m["at_level"] = s_at_level();
    m["scale"] = s_scale();
    m["baseline"] = s_baseline();
    m["relative_histogram"] = s_relative_histogram();
    m["core_histogram"] = s_core_histogram();
    m["cores"] = s_cores();
    m["patches"] = s_patches();
    m["tracks"] = s_tracks();
    m["tile_crossings"] = s_tile_crossings();
    m["power_series"] = s_power_series();
    m["lagfree_surrogate"] = s_lagfree_surrogate();
    for (const auto& [name, s] : m) s.validate(name);
    return m;
  }();
  return t;
}

}  // namespace

const Signature* builtin_signature(const std::string& node) {
  const auto& t = table();
  const auto it = t.find(node);
  return it == t.end() ? nullptr : &it->second;
}
std::vector<std::string> builtin_signature_names() {
  std::vector<std::string> v;
  for (const auto& [n, _] : table()) v.push_back(n);
  return v;
}

}  // namespace nxr::graph
