// nxr_graph — the type system's terms (spec docs/specs/2026-11-17-type-system.md, T1): manifolds, domains, the printed form and its parser, unification. Synthetic only.
//   1. the printed form round-trips for every class, and the spec's own examples print as written;
//   2. unification: variables (and their roles), ids, anonymous manifolds, made ones, member sets, restrictions, derived manifolds, the one `apply` subsumption;
//      each refusal says which two terms did not meet;
//   3. the closed vocabulary of quantities; G5's spin and gauge are carried and printed.
#include <cstdio>
#include <functional>
#include <string>

#include "nxr/graph/types.h"

namespace g = nxr::graph;

namespace {
int failures = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("  CHECK failed: %s @ %s:%d\n", #cond, __FILE__, __LINE__); ++failures; } } while (0)
#define CHECK_EQ(a, b) do { const std::string x_ = (a), y_ = (b); if (x_ != y_) { std::printf("  CHECK_EQ failed @ %s:%d\n    got:    %s\n    wanted: %s\n", __FILE__, __LINE__, x_.c_str(), y_.c_str()); ++failures; } } while (0)

bool refused(const std::function<void()>& f, const std::string& part) {
  try { f(); } catch (const std::invalid_argument& e) {
    if (std::string(e.what()).find(part) != std::string::npos) return true;
    std::printf("  refused, but with: %s (wanted '%s')\n", e.what(), part.c_str());
    return false;
  }
  std::printf("  not refused (wanted '%s')\n", part.c_str());
  return false;
}

g::Domain dom(std::vector<g::ManifoldRef> f) { g::Domain d; d.factors = std::move(f); return d; }

void test_print_parse() {
  // the spec's examples (section 3.3), printed as written, and parsed back to the same text
  const char* examples[] = {
      "field<cortex(kernel) × frames(tiling); complex vector3; current; band 8-12 Hz morlet; level tiles@5>",
      "measurement<members(cells_l5); amplitude; band 8-16 Hz; level tiles@5>",
      "selection<spans on time(rest); partition; detected over ladder@8-16>",
      "statistic<summaries; axes groups(cells_l5); of amplitude>",
      "operator<sensors(rest)|channels(kernel) → cortex(kernel); real → vector3; inverse kernel>",
      "operator<transition head: device:run1 → head>",
      "field<sensors(rest) × time(rest); real; recording>",
      "field<glued(cortex(kernel)) × time(rest); complex; power; band 8-16 Hz; level tiles@[4,5] mexhat; non-linear; spin 1; gauge gc@8579865/vertex:cortex; unit T>",
      "statistic<summaries; axes sensors(rest); of envelope band 8-16 Hz on sensors(rest)>",
      "statistic<histogram; axes bins(~hist.out); of relative power level tiles@5 on glued(cortex(kernel)) × frames(tiling)>",
      "selection<set on cortex(kernel) × time(rest); partition; as cells; band 8-16 Hz; level tiles@5>",
      "manifold<frequency(~psd.f)>",
      "constant<real; histogram>",
      "field<#1 × #2 {cells 1}; real>",
      "field<M × T; real>",
  };
  for (const char* e : examples) {
    const g::PortType t = g::parse_type(e);
    CHECK(t.typed);
    CHECK_EQ(g::print(t), e);
  }
  const g::PortType f = g::parse_type("field<cortex(kernel) × frames(tiling); complex vector3; current; band 8-12 Hz morlet; level tiles@5>");
  CHECK(f.cls == g::PortClass::field && f.value == g::ValueType::vector3 && f.complex_parts);
  CHECK(f.domain.factors.size() == 2 && f.domain.streamed() != nullptr && g::is_tiled(*f.domain.streamed()));
  CHECK(f.res.band && (*f.res.band)[0] == 8 && (*f.res.band)[1] == 12 && f.res.band_shape == "morlet");
  CHECK(f.res.ladder == "tiles" && f.res.levels == std::vector<int>{5});
  const g::PortType o = g::parse_type("operator<transition head: device:run1 → head>");
  CHECK(o.transition && o.transition->first.name == "device:run1" && o.transition->second.name == "head" && o.transition->first.space == "head");
  const g::PortType m = g::parse_type("measurement<members(cells_l5); amplitude; band 8-16 Hz>");
  CHECK(m.domain.factors[0]->form == g::Manifold::Form::members && m.domain.factors[0]->by == "cells_l5");
  const g::PortType s = g::parse_type("selection<spans on time(rest); partition; detected over ladder@8-16>");
  CHECK(s.sel && s.sel->partition && s.sel->detected_over == "ladder@8-16" && s.sel->type == "spans");
  const g::PortType st = g::parse_type("statistic<summaries; axes sensors(rest); of envelope band 8-16 Hz on sensors(rest)>");
  CHECK(st.of && st.of->quantity == "envelope" && st.of->res.band && st.of->on.factors.size() == 1);

  // a built type prints and parses to the same text
  g::PortType t;
  t.typed = true;
  t.cls = g::PortClass::field;
  t.domain = dom({g::manifold_id("sensors", "rest"), g::time_line("rest", 100.0)});
  t.quantity = "envelope";
  t.nonlinear = true;
  t.res.band = {{8.0, 16.0}};
  CHECK_EQ(g::print(t), "field<sensors(rest) × time(rest); real; envelope; band 8-16 Hz; non-linear>");
  CHECK_EQ(g::print(g::parse_type(g::print(t))), g::print(t));

  CHECK(refused([] { g::parse_type("cortex(kernel)"); }, "not a printed type"));
  CHECK(refused([] { g::parse_type("blob<real>"); }, "not a class"));
  CHECK(refused([] { g::parse_type("field<cortex(kernel); quaternionic>"); }, "not a value type"));
  CHECK(refused([] { g::parse_type("operator<a(b) to c(d); real → real>"); }, "FROM → TO"));
}

void test_manifolds() {
  const auto cortex = g::manifold_id("cortex", "kernel", g::ManifoldKind::surface, g::CoordinateSystem{"head", "scs", "m"}, 15000);
  const auto cortex2 = g::manifold_id("cortex", "other", g::ManifoldKind::surface);
  CHECK(g::same(cortex, g::manifold_id("cortex", "kernel")));
  CHECK(!g::same(cortex, cortex2));
  CHECK(g::same(g::anonymous(1), g::anonymous(1)) && !g::same(g::anonymous(1), g::anonymous(2)) && !g::same(g::anonymous(1), g::manifold_id("", "1")));
  CHECK(g::same(g::members("cells"), g::members("cells")) && !g::same(g::members("cells"), g::members("cells2")));
  CHECK(g::same(g::restricted(cortex, "channels(k)"), g::restricted(cortex, "channels(k)")) && !g::same(g::restricted(cortex, "channels(k)"), g::restricted(cortex, "channels(j)")));
  CHECK(g::same(g::derived("glued", cortex), g::derived("glued", cortex)) && !g::same(g::derived("glued", cortex), g::derived("glued", cortex2)) && !g::same(g::derived("glued", cortex), cortex));
  CHECK(g::same(g::made("frequency", "psd.f"), g::made("frequency", "psd.f")) && !g::same(g::made("frequency", "psd.f"), g::made("frequency", "psd2.f")));
  CHECK_EQ(g::print(g::derived("glued", cortex)), "glued(cortex(kernel))");
  CHECK_EQ(g::print(g::restricted(g::manifold_id("sensors", "rest"), "channels(kernel)")), "sensors(rest)|channels(kernel)");
  CHECK_EQ(g::print(g::parse_manifold("glued(cortex(kernel))")), "glued(cortex(kernel))");
  CHECK(g::in_seconds(*g::parse_manifold("time(rest)")));
  CHECK(g::parse_manifold("members(cells_l5)")->form == g::Manifold::Form::members);
  CHECK(g::parse_manifold("#7")->form == g::Manifold::Form::anonymous);
  CHECK(g::parse_manifold("M")->form == g::Manifold::Form::variable);
  const g::Domain d = dom({cortex, g::frame_line("tiling")});
  CHECK(d.streamed() && g::is_tiled(*d.streamed()));
  CHECK(dom({cortex}).streamed() == nullptr);
  CHECK(g::same(dom({cortex}), dom({cortex})) && !g::same(dom({cortex}), dom({cortex2})));
}

void test_unify() {
  const auto sensors = g::manifold_id("sensors", "rest", g::ManifoldKind::points);
  const auto time = g::time_line("rest", 100.0), frames = g::frame_line("tiling"), cortex = g::manifold_id("cortex", "kernel", g::ManifoldKind::surface);
  g::Bindings b;
  std::string why;

  // variables, and a time variable's constraints (a line in seconds, raw samples, a path)
  CHECK(g::unify(dom({g::var("M"), g::time_var("T")}), dom({sensors, time}), b, why));
  CHECK(g::same(b.vars["M"], sensors) && g::same(b.vars["T"], time));
  g::Bindings b2;
  CHECK(!g::unify(dom({g::var("M"), g::time_var("T")}), dom({sensors, cortex}), b2, why));
  CHECK(why.find("a line in seconds was expected where cortex(kernel)") != std::string::npos);
  g::Bindings b3;
  CHECK(!g::unify(dom({g::var("M"), g::time_var("T")}), dom({sensors, frames}), b3, why));   // a time var does not take a frame line: a tiled line is not raw samples
  CHECK(why.find("raw samples") != std::string::npos && why.find("frames(tiling)") != std::string::npos);
  g::Bindings b4;
  CHECK(g::unify(dom({g::var("M"), g::frames_var("T")}), dom({sensors, frames}), b4, why));  // a frames var takes both
  g::Bindings b5;
  CHECK(g::unify(dom({g::var("M"), g::frames_var("T")}), dom({sensors, time}), b5, why));
  // a frame line is a line in seconds, where a line in seconds is asked; the frequency line (Hz) and the tile members of another unit are not
  g::Bindings b6;
  CHECK(g::unify(dom({g::line_var("T", "s")}), dom({frames}), b6, why) && g::in_seconds(*frames) && g::is_tiled(*frames) && !g::is_tiled(*time));
  g::Bindings b7;
  CHECK(!g::unify(dom({g::line_var("T", "s")}), dom({g::made_line("frequency", "u.spectrum", "Hz", 0.0, 100.0, time)}), b7, why));
  CHECK(why.find("a line in seconds was expected") != std::string::npos);
  // a TOPOLOGY constraint: a cycle is refused by a node that reads a path (a zero-phase filter), and a path by one that reads a cycle; no constraint takes both; the topology is part of equality
  const auto cycle = g::time_line("rest", 100.0, g::Topology::cycle);
  g::Bindings b8;
  CHECK(!g::unify(dom({g::var("M"), g::time_var("T")}), dom({sensors, cycle}), b8, why));
  CHECK(why.find("a path was expected where time(rest)") != std::string::npos && why.find("a cycle") != std::string::npos);
  g::Bindings b9;
  CHECK(g::unify(dom({g::var("M"), g::frames_var("T")}), dom({sensors, cycle}), b9, why));
  g::Bindings b10;
  CHECK(g::unify(dom({g::line_var("T", "s", false, g::Topology::cycle)}), dom({cycle}), b10, why));
  g::Bindings b11;
  CHECK(!g::unify(dom({g::line_var("T", "s", false, g::Topology::cycle)}), dom({time}), b11, why));
  CHECK(!g::same(time, cycle) && g::same(time, g::time_line("rest", 100.0)));

  // a variable met twice must be the same term; ids by equality
  g::Bindings c;
  CHECK(g::unify(dom({g::var("M")}), dom({cortex}), c, why));
  CHECK(!g::unify(dom({g::var("M")}), dom({g::derived("glued", cortex)}), c, why));
  CHECK(why.find("glued(cortex(kernel)) is not cortex(kernel)") != std::string::npos && why.find("variable M") != std::string::npos);
  CHECK(g::unify(g::manifold_id("cortex", "kernel"), cortex, c, why));
  CHECK(!g::unify(g::manifold_id("cortex", "x"), cortex, c, why));

  // one manifold, one size: the same id with two known vertex counts is refused (a variable bound to it, or the id itself); an unknown count (0) still unifies
  const auto m10 = g::manifold_id("cortex", "m", g::ManifoldKind::surface, std::nullopt, 10), m12 = g::manifold_id("cortex", "m", g::ManifoldKind::surface, std::nullopt, 12);
  g::Bindings sz;
  CHECK(g::unify(dom({g::var("M")}), dom({m10}), sz, why));
  CHECK(!g::unify(dom({g::var("M")}), dom({m12}), sz, why));
  CHECK(why.find("12 vertices where cortex(m) has 10") != std::string::npos);
  g::Bindings sz2;
  CHECK(!g::unify(m10, m12, sz2, why));
  CHECK(g::unify(g::manifold_id("cortex", "m"), m12, sz2, why));

  // rest variables carry the other factors through unchanged
  g::Bindings r;
  CHECK(g::unify(dom({g::var("A"), g::var_rest("R")}), dom({sensors, time}), r, why));
  CHECK(r.rests["R"].size() == 1 && g::same(r.rests["R"][0], time));
  const g::Domain out = g::instantiate(dom({g::var("B"), g::var_rest("R")}), [&] { g::Bindings x = r; x.vars["B"] = cortex; return x; }());
  CHECK_EQ(g::print(out), "cortex(kernel) × time(rest)");
  g::Bindings r2;
  CHECK(g::unify(dom({g::var_rest("R"), g::time_var("T")}), dom({sensors, cortex, time}), r2, why));
  CHECK(r2.rests["R"].size() == 2);
  g::Bindings r3;
  CHECK(!g::unify(dom({g::var("A"), g::var("B")}), dom({sensors}), r3, why) && why.find("1 factor(s)") != std::string::npos);

  // anonymous: equal only to itself
  g::Bindings a;
  CHECK(g::unify(g::anonymous(1), g::anonymous(1), a, why));
  CHECK(!g::unify(g::anonymous(1), g::anonymous(2), a, why));

  // members(P) with members(Q) when P = Q
  g::Bindings m;
  CHECK(g::unify(g::members("cells"), g::members("cells"), m, why));
  CHECK(!g::unify(g::members("cells"), g::members("other"), m, why));

  // derived(rel, M) with derived(rel, N) when M = N: a pattern may hold a variable
  g::Bindings d;
  CHECK(g::unify(g::derived("glued", g::var("M")), g::derived("glued", cortex), d, why) && g::same(d.vars["M"], cortex));
  CHECK(!g::unify(g::derived("glued", g::var("M")), cortex, d, why));

  // restricted(M, S) with restricted(N, U) when M = N and S = U
  const auto ks = g::restricted(sensors, "channels(kernel)"), cs = g::restricted(sensors, "channels(cross_spectrum)");
  g::Bindings rs;
  CHECK(g::unify(g::restricted(g::var("M"), ""), ks, rs, why) && g::same(rs.vars["M"], sensors));
  CHECK(!g::unify(g::restricted(g::var("M"), "channels(other)"), ks, rs, why));

  // the one subsumption: an operator's source restricted to S accepts the whole manifold, and the same restriction; another selection of it is refused
  const auto src = g::from_or_restriction_of(g::var("A"));
  g::Bindings s1;                                           // x on the whole sensors; the kernel from sensors|channels(kernel)
  CHECK(g::unify(dom({g::var("A"), g::var_rest("R")}), dom({sensors, time}), s1, why));
  CHECK(g::unify(src, ks, s1, why));
  g::Bindings s2;                                           // x on the kernel's own channels: the same restriction
  CHECK(g::unify(dom({g::var("A")}), dom({ks}), s2, why) && g::unify(src, ks, s2, why));
  g::Bindings s3;                                           // x on the cross-spectrum's channels: refused, naming both
  CHECK(g::unify(dom({g::var("A")}), dom({cs}), s3, why));
  CHECK(!g::unify(src, ks, s3, why));
  CHECK(why.find("sensors(rest)|channels(kernel)") != std::string::npos && why.find("sensors(rest)|channels(cross_spectrum)") != std::string::npos);
  g::Bindings s4;                                           // x on the cortex, a kernel from sensors: not subsumed
  CHECK(g::unify(dom({g::var("A")}), dom({cortex}), s4, why) && !g::unify(src, ks, s4, why));

  // cells
  g::Domain cells1 = dom({cortex}); cells1.cells = "1";
  g::Bindings cb;
  CHECK(!g::unify(dom({g::var("A")}), cells1, cb, why) && why.find("cells") != std::string::npos);

  // instantiate: members_of_port and an unbound variable
  g::Bindings ib;
  ib.vars["M"] = cortex;
  CHECK_EQ(g::print(g::instantiate(g::members_of_port("tiling"), ib, {{"tiling", "tiling"}})), "members(tiling)");
  CHECK(refused([&] { g::instantiate(g::var("Z"), ib); }, "variable Z is not bound"));
  CHECK(refused([&] { g::instantiate(g::members_of_port("p"), ib); }, "port 'p'"));
}

void test_vocabulary_and_gauge() {
  CHECK(g::is_quantity("amplitude") && g::is_quantity("relative power") && g::is_quantity("phase_current") && g::is_quantity("curl_tan") && g::is_quantity(""));
  CHECK(!g::is_quantity("amplitudee"));
  const g::PortType t = g::parse_type("field<cortex(kernel) × time(rest); complex; current; spin 1; gauge gc@1/vertex:cortex>");
  CHECK(t.spin == 1 && t.gauge == "gc@1/vertex:cortex");
  const g::PortType v = g::parse_type("field<cortex(kernel) × time(rest); vector3; phase_current>");
  CHECK(v.spin == 0 && v.gauge.empty());   // an embedded value is gauge-free
  g::PortType old;                          // C1's type is still a type: nothing typed about it
  CHECK(!old.typed && old.domain.empty());
}
}  // namespace

int main() {
  test_print_parse();
  test_manifolds();
  test_unify();
  test_vocabulary_and_gauge();
  if (failures) { std::printf("test_graph_types: %d failure(s)\n", failures); return 1; }
  std::printf("test_graph_types: ok\n");
  return 0;
}
