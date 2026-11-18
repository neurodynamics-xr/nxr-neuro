// nxr::forms against the Studio's own functions (golden: test/golden/gen_forms_golden.mts, from fieldForms.ts `formValues` on faceGradient.ts
// `assembleFaceGradient`): on a bumpy icosphere with random ℝ, ℂ and ℍ fields (two frames each), the vertex normals, the mean curvature, the signed volume
// and EVERY form of every type equal the Studio's, value by value (float32, to the bit). Then the complex extension: a linear form of re + i·im is
// form(re) + i·form(im); abs / absq give the envelope over every real and imaginary component; a type without the form, or a length that is not whole
// frames, is refused.
#include "nxr/signal/forms.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static const std::string kDir = NXR_TEST_GOLDEN_DIR;
static int fails = 0;
#define EXPECT(c, m) do { if (!(c)) { ++fails; std::cout << "  [FAIL] " << m << "\n"; } } while (0)

using namespace nxr::signal;

// the golden: one record a line, "<tag words…> <count> <numbers…>"; a float32 written as its shortest double repr reads back to the same bits
static std::map<std::string, std::vector<double>> load(const std::string& path, int& nV, int& nF, double& volume) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("no golden at " + path);
  std::map<std::string, std::vector<double>> r;
  std::string ln;
  while (std::getline(in, ln)) {
    std::istringstream s(ln);
    std::string w, tag;
    s >> w;
    if (w == "mesh") { s >> nV >> nF; continue; }
    if (w == "signedVolume") { s >> volume; continue; }
    tag = w;
    if (w == "field") { std::string t; s >> t; tag += " " + t; }
    if (w == "form") { std::string t, f; s >> t >> f; tag += " " + t + " " + f; }
    std::size_t n = 0;
    s >> n;
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) { std::string x; s >> x; v[i] = std::strtod(x.c_str(), nullptr); }
    r[tag] = std::move(v);
  }
  return r;
}

template <class T> static std::vector<float> asFloat(const std::vector<T>& v) { std::vector<float> o(v.size()); for (std::size_t i = 0; i < v.size(); ++i) o[i] = static_cast<float>(v[i]); return o; }

// the count of values that differ from the golden, and the largest difference
static std::pair<std::size_t, double> diff(const std::vector<float>& got, const std::vector<double>& want) {
  if (got.size() != want.size()) return {std::max(got.size(), want.size()), INFINITY};
  std::size_t n = 0;
  double worst = 0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const float w = static_cast<float>(want[i]);
    if (!(got[i] == w || (std::isnan(got[i]) && std::isnan(w)))) { ++n; worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - w)); }
  }
  return {n, worst};
}

int main() {
  int nV = 0, nF = 0;
  double volume = 0;
  const auto g = load(kDir + "/forms_golden.txt", nV, nF, volume);
  const std::vector<float> P = asFloat(g.at("positions"));
  std::vector<int32_t> F(g.at("faces").size());
  for (std::size_t i = 0; i < F.size(); ++i) F[i] = static_cast<int32_t>(g.at("faces")[i]);
  const FaceGradientOperator op = assembleFaceGradient(P.data(), F.data(), nV, nF);

  // THE OPERATOR the forms stand on: vertex normals, mean curvature, signed volume
  const auto dn = diff(op.vertexNormals, g.at("vertexNormals"));
  const auto dh = diff(meanCurvature(op), g.at("meanCurvature"));
  std::printf("  operator: vertex normals %zu values differ (max %.2e); mean curvature %zu (max %.2e); signed volume %.17g vs %.17g\n", dn.first, dn.second, dh.first, dh.second,
              op.signedVolume, volume);
  EXPECT(dn.first == 0, "vertex normals differ from the Studio's");
  EXPECT(dh.first == 0, "mean curvature differs from the Studio's");
  EXPECT(op.signedVolume == volume && volume > 0, "signed volume differs, or the winding is not outward");

  // EVERY FORM of every type, to the bit
  const std::pair<const char*, FieldType> types[] = {{"real", FieldType::Real}, {"complex", FieldType::Complex}, {"quaternion", FieldType::Quaternion}};
  int forms = 0;
  for (const auto& [name, t] : types) {
    const std::vector<float> values = asFloat(g.at(std::string("field ") + name));
    for (const FormInfo& info : formsOf(t)) {
      const auto it = g.find(std::string("form ") + name + " " + info.key);
      EXPECT(it != g.end(), "the golden has no " << name << " " << info.key);
      if (it == g.end()) continue;
      const auto d = diff(formValues(t, info.id, values.data(), values.size(), op), it->second);
      if (d.first) std::printf("  %s %s: %zu values differ (max %.2e)\n", name, info.key, d.first, d.second);
      EXPECT(d.first == 0, name << " " << info.key << " differs from the Studio's");
      ++forms;
    }
  }
  std::printf("  %d forms equal the Studio's, value by value\n", forms);
  EXPECT(forms == 27, "the Studio has 27 forms (1 + 13 + 13), checked " << forms);

  // THE COMPLEX EXTENSION
  const std::vector<float> re = asFloat(g.at("field quaternion"));
  std::vector<float> im(re.rbegin(), re.rend());                 // another field of the same shape
  for (Form id : {Form::nx, Form::xt, Form::x, Form::curl, Form::div_tan}) {
    const ComplexForm c = formComplex(FieldType::Quaternion, id, re.data(), im.data(), re.size(), op);
    EXPECT(c.re == formValues(FieldType::Quaternion, id, re.data(), re.size(), op) && c.im == formValues(FieldType::Quaternion, id, im.data(), im.size(), op),
           "a linear form of a complex field is not form(re) + i form(im)");
  }
  const ComplexForm env = formComplex(FieldType::Quaternion, Form::absq, re.data(), im.data(), re.size(), op);
  double worst = 0;
  for (std::size_t k = 0; k < static_cast<std::size_t>(nV); ++k) {
    double s = 0;
    for (int d = 0; d < 4; ++d) s += static_cast<double>(re[4 * k + d]) * re[4 * k + d] + static_cast<double>(im[4 * k + d]) * im[4 * k + d];
    worst = std::max(worst, std::fabs(env.re[k] - std::sqrt(s)) / std::sqrt(s));
  }
  std::printf("  the envelope |q| over C^4 against sqrt(sum of squares): %.2e (relative); im empty: %d\n", worst, env.im.empty());
  EXPECT(worst < 1e-6 && env.im.empty() && env.re.size() == 2 * static_cast<std::size_t>(nV), "abs / absq of a complex field is not its envelope");
  EXPECT(isLinear(Form::nx) && isLinear(Form::curl) && !isLinear(Form::abs) && !isLinear(Form::absq), "isLinear");

  // REFUSALS
  auto refused = [](auto&& f) { try { f(); } catch (const std::invalid_argument&) { return true; } return false; };
  EXPECT(refused([&] { formValues(FieldType::Real, Form::nx, re.data(), static_cast<std::size_t>(nV), op); }), "ℝ has no form nx: not refused");
  EXPECT(refused([&] { formValues(FieldType::Quaternion, Form::nx, re.data(), 4 * static_cast<std::size_t>(nV) + 1, op); }), "a partial frame: not refused");
  EXPECT(formOf("div_tan") == Form::div_tan && !formOf("nope"), "formOf");

  std::cout << (fails ? "FAILED " : "ok ") << fails << "\n";
  return fails ? 1 : 0;
}
