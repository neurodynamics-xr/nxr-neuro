// nxr::forms — a field's forms (forms.h): a line port of the Studio's fieldForms.ts and the faceGradient.ts functions they read. JavaScript's arithmetic:
// every value a double, stored as float where the Studio writes a Float32Array; Math.hypot as V8 computes it; compiled without fused multiply-add.
#include "nxr/forms.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>

namespace nxr::forms {

namespace {

constexpr double kAreaEps = 1e-12;   // faceGradient.ts AREA_EPS

// Math.hypot as V8 computes it (scaled by the largest magnitude, a compensated sum), for any number of arguments: the same as ladder::jsHypot for three.
double jsHypot(std::initializer_list<double> args) {
    double mx = 0;
    for (double v : args) {
        const double a = std::fabs(v);
        if (std::isnan(a)) return std::numeric_limits<double>::quiet_NaN();
        mx = std::max(mx, a);
    }
    if (mx == 0) return 0.0;
    if (std::isinf(mx)) return std::numeric_limits<double>::infinity();
    double sum = 0, comp = 0;
    for (double v : args) {
        const double n = std::fabs(v) / mx, summand = n * n - comp, pre = sum + summand;
        comp = (pre - sum) - summand;
        sum = pre;
    }
    return std::sqrt(sum) * mx;
}

const std::vector<FormInfo> kReal = {{Form::f, "f", "f", "value", FieldType::Real}};

std::vector<FormInfo> divCurl(const char* v) {
    // the Studio builds the symbols from the vector's letter (v or x); the ids and names are fixed
    const std::string s(v);
    static std::deque<std::string> keep;    // the symbols' storage, alive for the program (a deque never moves its elements)
    auto sym = [&](const std::string& x) { keep.push_back(x); return keep.back().c_str(); };
    return {{Form::div, "div", sym("∇·" + s), "divergence", FieldType::Real},
            {Form::curl, "curl", sym("∇×" + s), "rotation", FieldType::Real},
            {Form::div_tan, "div_tan", sym("∇·" + s + "ₜ"), "divergence, tangent flow", FieldType::Real},
            {Form::div_3d, "div_3d", sym("∇·" + s + " (3D)"), "divergence, full vector", FieldType::Real},
            {Form::div_curv, "div_curv", sym("∇·(" + s + "ₙn)"), "divergence, bending of the surface", FieldType::Real},
            {Form::div_2H, "div_2H", sym("2H " + s + "ₙ"), "divergence, bending (curvature estimate)", FieldType::Real},
            {Form::curl_tan, "curl_tan", sym("∇×" + s + "ₜ"), "rotation, tangent flow", FieldType::Real},
            {Form::curl_3d, "curl_3d", sym("∇×" + s + " (3D)"), "rotation, full vector", FieldType::Real},
            {Form::curl_curv, "curl_curv", sym("∇×(" + s + "ₙn)"), "rotation, normal part (mesh error)", FieldType::Real}};
}

const std::vector<FormInfo>& complexForms() {
    static const std::vector<FormInfo> f = [] {
        std::vector<FormInfo> r = {{Form::abs, "abs", "|v|", "strength", FieldType::Real}};
        for (const auto& d : divCurl("v")) r.push_back(d);
        r.push_back({Form::v, "v", "v", "the vector", FieldType::Complex});
        r.push_back({Form::Jv, "Jv", "Jv", "turned a quarter", FieldType::Complex});
        return r;
    }();
    return f;
}

const std::vector<FormInfo>& quaternionForms() {
    static const std::vector<FormInfo> f = [] {
        std::vector<FormInfo> r = {{Form::absq, "absq", "|q|", "strength", FieldType::Real},
                                   {Form::nx, "nx", "n·x", "out of the surface", FieldType::Real},
                                   {Form::w, "w", "w", "the number part", FieldType::Real}};
        for (const auto& d : divCurl("x")) r.push_back(d);
        r.push_back({Form::xt, "xt", "xₜ", "along the surface", FieldType::Complex});
        r.push_back({Form::x, "x", "x", "the vector in space", FieldType::Quaternion});
        return r;
    }();
    return f;
}

// flowFaceScalarsCpu: per face Σ ∇φ_i · J_i and (Σ ∇φ_i × J_i) · N (J [3 nV]; `tangential` removes each vertex's normal component first)
void flowFaceScalars(const FaceGradientOperator& op, const float* J, bool tangential, std::vector<float>& div, std::vector<float>& curl) {
    div.assign(static_cast<std::size_t>(op.nF), 0.0f);
    curl.assign(static_cast<std::size_t>(op.nF), 0.0f);
    for (int f = 0; f < op.nF; ++f) {
        if (op.areas[static_cast<std::size_t>(f)] == 0.0f) continue;
        double d = 0, wx = 0, wy = 0, wz = 0;
        for (int s = 0; s < 3; ++s) {
            const int v = op.faces[static_cast<std::size_t>(3 * f + s)];
            const double gx = op.grads[static_cast<std::size_t>(9 * f + 3 * s)], gy = op.grads[static_cast<std::size_t>(9 * f + 3 * s + 1)],
                         gz = op.grads[static_cast<std::size_t>(9 * f + 3 * s + 2)];
            double jx = J[3 * v], jy = J[3 * v + 1], jz = J[3 * v + 2];
            if (tangential) {
                const double nx = op.vertexNormals[static_cast<std::size_t>(3 * v)], ny = op.vertexNormals[static_cast<std::size_t>(3 * v + 1)],
                             nz = op.vertexNormals[static_cast<std::size_t>(3 * v + 2)];
                const double dot = jx * nx + jy * ny + jz * nz;
                jx -= dot * nx;
                jy -= dot * ny;
                jz -= dot * nz;
            }
            d += gx * jx + gy * jy + gz * jz;
            wx += gy * jz - gz * jy;
            wy += gz * jx - gx * jz;
            wz += gx * jy - gy * jx;
        }
        div[static_cast<std::size_t>(f)] = static_cast<float>(d);
        curl[static_cast<std::size_t>(f)] = static_cast<float>(wx * op.normals[static_cast<std::size_t>(3 * f)] + wy * op.normals[static_cast<std::size_t>(3 * f + 1)] +
                                                               wz * op.normals[static_cast<std::size_t>(3 * f + 2)]);
    }
}

// faceToVertexCpu: the area-weighted face -> vertex mean (a vertex with no live face stays 0)
std::vector<float> faceToVertex(const FaceGradientOperator& op, const std::vector<float>& fv) {
    std::vector<float> out(static_cast<std::size_t>(op.nV), 0.0f);
    for (int v = 0; v < op.nV; ++v) {
        const double wsum = op.vertexAreaSum[static_cast<std::size_t>(v)];
        if (!(wsum > 0)) continue;
        double acc = 0;
        for (int p = op.vertexFaceOffsets[static_cast<std::size_t>(v)]; p < op.vertexFaceOffsets[static_cast<std::size_t>(v + 1)]; ++p) {
            const int f = op.vertexFaceIndices[static_cast<std::size_t>(p)];
            acc += static_cast<double>(op.areas[static_cast<std::size_t>(f)]) * fv[static_cast<std::size_t>(f)];
        }
        out[static_cast<std::size_t>(v)] = static_cast<float>(acc / wsum);
    }
    return out;
}

// flowFieldCpu
void flowField(const FaceGradientOperator& op, const float* J, bool tangential, std::vector<float>& div, std::vector<float>& curl) {
    std::vector<float> fd, fc;
    flowFaceScalars(op, J, tangential, fd, fc);
    div = faceToVertex(op, fd);
    curl = faceToVertex(op, fc);
}

// normalPartCpu: J_n n along the vertex normals
std::vector<float> normalPart(const FaceGradientOperator& op, const float* J) {
    std::vector<float> out(static_cast<std::size_t>(3 * op.nV));
    for (int v = 0; v < op.nV; ++v) {
        const double nx = op.vertexNormals[static_cast<std::size_t>(3 * v)], ny = op.vertexNormals[static_cast<std::size_t>(3 * v + 1)],
                     nz = op.vertexNormals[static_cast<std::size_t>(3 * v + 2)];
        const double d = J[3 * v] * nx + J[3 * v + 1] * ny + J[3 * v + 2] * nz;
        out[static_cast<std::size_t>(3 * v)] = static_cast<float>(d * nx);
        out[static_cast<std::size_t>(3 * v + 1)] = static_cast<float>(d * ny);
        out[static_cast<std::size_t>(3 * v + 2)] = static_cast<float>(d * nz);
    }
    return out;
}

// divCurv2HCpu: 2 H (J · n)
std::vector<float> divCurv2H(const FaceGradientOperator& op, const float* J) {
    const std::vector<float> H = meanCurvature(op);
    std::vector<float> out(static_cast<std::size_t>(op.nV));
    for (int v = 0; v < op.nV; ++v) {
        const double h = H[static_cast<std::size_t>(v)];
        out[static_cast<std::size_t>(v)] = static_cast<float>(2 * h * (J[3 * v] * static_cast<double>(op.vertexNormals[static_cast<std::size_t>(3 * v)]) +
                                                                       J[3 * v + 1] * static_cast<double>(op.vertexNormals[static_cast<std::size_t>(3 * v + 1)]) +
                                                                       J[3 * v + 2] * static_cast<double>(op.vertexNormals[static_cast<std::size_t>(3 * v + 2)])));
    }
    return out;
}

// divCurlOf (fieldForms.ts): a divergence / rotation form of an embedded vector field J [3 nV]
std::vector<float> divCurlOf(Form id, const float* J, const FaceGradientOperator& op) {
    std::vector<float> d, c;
    if (id == Form::div || id == Form::curl) { flowField(op, J, false, d, c); return id == Form::div ? d : c; }
    if (id == Form::div_2H) return divCurv2H(op, J);
    const bool isDiv = id == Form::div_tan || id == Form::div_3d || id == Form::div_curv;
    const Form k = id;
    if (k == Form::div_tan || k == Form::curl_tan) flowField(op, J, true, d, c);
    else if (k == Form::div_3d || k == Form::curl_3d) flowField(op, J, false, d, c);
    else { const std::vector<float> np = normalPart(op, J); flowField(op, np.data(), false, d, c); }
    return isDiv ? d : c;
}

}  // namespace

const std::vector<FormInfo>& formsOf(FieldType t) {
    return t == FieldType::Real ? kReal : t == FieldType::Complex ? complexForms() : quaternionForms();
}

std::optional<FormInfo> formInfo(FieldType t, Form id) {
    for (const auto& f : formsOf(t)) if (f.id == id) return f;
    return std::nullopt;
}

std::optional<Form> formOf(const std::string& key) {
    for (FieldType t : {FieldType::Real, FieldType::Complex, FieldType::Quaternion})
        for (const auto& f : formsOf(t)) if (key == f.key) return f.id;
    return std::nullopt;
}

std::size_t widthOf(FieldType t, std::size_t n) { return t == FieldType::Real ? n : t == FieldType::Complex ? 3 * n : 4 * n; }

bool isDivCurl(Form id) {
    switch (id) {
        case Form::div: case Form::curl: case Form::div_tan: case Form::div_3d: case Form::div_curv: case Form::div_2H:
        case Form::curl_tan: case Form::curl_3d: case Form::curl_curv: return true;
        default: return false;
    }
}

bool isLinear(Form id) { return id != Form::abs && id != Form::absq; }

FaceGradientOperator assembleFaceGradient(const float* position, const int32_t* index, int nV, int nF) {
    FaceGradientOperator op;
    op.nF = nF;
    op.nV = nV;
    op.grads.assign(static_cast<std::size_t>(9 * nF), 0.0f);
    op.normals.assign(static_cast<std::size_t>(3 * nF), 0.0f);
    op.areas.assign(static_cast<std::size_t>(nF), 0.0f);
    op.faces.assign(index, index + 3 * nF);
    op.position.assign(position, position + 3 * nV);
    double cx = 0, cy = 0, cz = 0;
    for (int f = 0; f < nF; ++f) {
        const int i0 = index[3 * f], i1 = index[3 * f + 1], i2 = index[3 * f + 2];
        const double ax = position[3 * i0], ay = position[3 * i0 + 1], az = position[3 * i0 + 2];
        const double bx = position[3 * i1], by = position[3 * i1 + 1], bz = position[3 * i1 + 2];
        const double px = position[3 * i2], py = position[3 * i2 + 1], pz = position[3 * i2 + 2];
        const double e1x = bx - ax, e1y = by - ay, e1z = bz - az;
        const double e2x = px - ax, e2y = py - ay, e2z = pz - az;
        const double nx = e1y * e2z - e1z * e2y, ny = e1z * e2x - e1x * e2z, nz = e1x * e2y - e1y * e2x;
        const double twoA = jsHypot({nx, ny, nz});
        if (!(twoA > kAreaEps)) { ++op.degenerateFaces; continue; }
        op.areas[static_cast<std::size_t>(f)] = static_cast<float>(twoA / 2);
        const double ux = nx / twoA, uy = ny / twoA, uz = nz / twoA;
        op.normals[static_cast<std::size_t>(3 * f)] = static_cast<float>(ux);
        op.normals[static_cast<std::size_t>(3 * f + 1)] = static_cast<float>(uy);
        op.normals[static_cast<std::size_t>(3 * f + 2)] = static_cast<float>(uz);
        const std::size_t g = static_cast<std::size_t>(9 * f);
        auto put = [&](int slot, double ex, double ey, double ez) {
            op.grads[g + static_cast<std::size_t>(slot * 3)] = static_cast<float>((uy * ez - uz * ey) / twoA);
            op.grads[g + static_cast<std::size_t>(slot * 3 + 1)] = static_cast<float>((uz * ex - ux * ez) / twoA);
            op.grads[g + static_cast<std::size_t>(slot * 3 + 2)] = static_cast<float>((ux * ey - uy * ex) / twoA);
        };
        put(0, px - bx, py - by, pz - bz);
        put(1, ax - px, ay - py, az - pz);
        put(2, bx - ax, by - ay, bz - az);
        cx += (ax + bx + px) / 3;
        cy += (ay + by + py) / 3;
        cz += (az + bz + pz) / 3;
    }
    const int live = nF - op.degenerateFaces;
    if (live > 0) { cx /= live; cy /= live; cz /= live; }
    double vol = 0;
    for (int f = 0; f < nF; ++f) {
        if (op.areas[static_cast<std::size_t>(f)] == 0.0f) continue;
        const int i0 = op.faces[static_cast<std::size_t>(3 * f)], i1 = op.faces[static_cast<std::size_t>(3 * f + 1)], i2 = op.faces[static_cast<std::size_t>(3 * f + 2)];
        const double mx = (static_cast<double>(position[3 * i0]) + position[3 * i1] + position[3 * i2]) / 3 - cx;
        const double my = (static_cast<double>(position[3 * i0 + 1]) + position[3 * i1 + 1] + position[3 * i2 + 1]) / 3 - cy;
        const double mz = (static_cast<double>(position[3 * i0 + 2]) + position[3 * i1 + 2] + position[3 * i2 + 2]) / 3 - cz;
        const double d = op.normals[static_cast<std::size_t>(3 * f)] * mx + op.normals[static_cast<std::size_t>(3 * f + 1)] * my + op.normals[static_cast<std::size_t>(3 * f + 2)] * mz;
        vol += op.areas[static_cast<std::size_t>(f)] * d;
    }
    op.signedVolume = vol / 3;
    std::vector<int32_t> counts(static_cast<std::size_t>(nV), 0);
    for (int f = 0; f < nF; ++f) {
        if (op.areas[static_cast<std::size_t>(f)] == 0.0f) continue;
        for (int s = 0; s < 3; ++s) { const int v = op.faces[static_cast<std::size_t>(3 * f + s)]; if (v >= 0 && v < nV) ++counts[static_cast<std::size_t>(v)]; }
    }
    op.vertexFaceOffsets.assign(static_cast<std::size_t>(nV + 1), 0);
    for (int v = 0; v < nV; ++v) op.vertexFaceOffsets[static_cast<std::size_t>(v + 1)] = op.vertexFaceOffsets[static_cast<std::size_t>(v)] + counts[static_cast<std::size_t>(v)];
    op.vertexFaceIndices.assign(static_cast<std::size_t>(op.vertexFaceOffsets[static_cast<std::size_t>(nV)]), 0);
    std::vector<int32_t> cursor(op.vertexFaceOffsets.begin(), op.vertexFaceOffsets.begin() + nV);
    op.vertexAreaSum.assign(static_cast<std::size_t>(nV), 0.0f);
    for (int f = 0; f < nF; ++f) {
        if (op.areas[static_cast<std::size_t>(f)] == 0.0f) continue;
        for (int s = 0; s < 3; ++s) {
            const int v = op.faces[static_cast<std::size_t>(3 * f + s)];
            if (v < 0 || v >= nV) continue;
            op.vertexFaceIndices[static_cast<std::size_t>(cursor[static_cast<std::size_t>(v)]++)] = f;
            // vertexAreaSum[v] += areas[f]: a Float32Array element, so the sum is rounded to float at every step
            op.vertexAreaSum[static_cast<std::size_t>(v)] = static_cast<float>(static_cast<double>(op.vertexAreaSum[static_cast<std::size_t>(v)]) + op.areas[static_cast<std::size_t>(f)]);
        }
    }
    op.vertexNormals.assign(static_cast<std::size_t>(3 * nV), 0.0f);
    for (int v = 0; v < nV; ++v) {
        double nx2 = 0, ny2 = 0, nz2 = 0;
        for (int p = op.vertexFaceOffsets[static_cast<std::size_t>(v)]; p < op.vertexFaceOffsets[static_cast<std::size_t>(v + 1)]; ++p) {
            const int f = op.vertexFaceIndices[static_cast<std::size_t>(p)];
            const double a = op.areas[static_cast<std::size_t>(f)];
            nx2 += a * op.normals[static_cast<std::size_t>(3 * f)];
            ny2 += a * op.normals[static_cast<std::size_t>(3 * f + 1)];
            nz2 += a * op.normals[static_cast<std::size_t>(3 * f + 2)];
        }
        const double len = jsHypot({nx2, ny2, nz2});
        if (len > 0) {
            op.vertexNormals[static_cast<std::size_t>(3 * v)] = static_cast<float>(nx2 / len);
            op.vertexNormals[static_cast<std::size_t>(3 * v + 1)] = static_cast<float>(ny2 / len);
            op.vertexNormals[static_cast<std::size_t>(3 * v + 2)] = static_cast<float>(nz2 / len);
        }
    }
    return op;
}

std::vector<float> meanCurvature(const FaceGradientOperator& op) {
    const std::vector<float>& P = op.position;
    std::vector<double> DX(static_cast<std::size_t>(3 * op.nV), 0.0), A(static_cast<std::size_t>(op.nV), 0.0);
    for (int f = 0; f < op.nF; ++f) {
        for (int k = 0; k < 3; ++k) {
            const int i = op.faces[static_cast<std::size_t>(3 * f + k)], j = op.faces[static_cast<std::size_t>(3 * f + (k + 1) % 3)], l = op.faces[static_cast<std::size_t>(3 * f + (k + 2) % 3)];
            auto p = [&](int a, int d) { return static_cast<double>(P[static_cast<std::size_t>(3 * a + d)]); };
            const double ux = p(i, 0) - p(l, 0), uy = p(i, 1) - p(l, 1), uz = p(i, 2) - p(l, 2);
            const double wx = p(j, 0) - p(l, 0), wy = p(j, 1) - p(l, 1), wz = p(j, 2) - p(l, 2);
            const double cx = uy * wz - uz * wy, cy = uz * wx - ux * wz, cz = ux * wy - uy * wx;
            const double cr = std::sqrt(cx * cx + cy * cy + cz * cz);
            const double c = cr > kAreaEps ? 0.5 * (ux * wx + uy * wy + uz * wz) / cr : 0;
            for (int d = 0; d < 3; ++d) {
                const double diff = c * (p(j, d) - p(i, d));
                DX[static_cast<std::size_t>(3 * i + d)] += diff;
                DX[static_cast<std::size_t>(3 * j + d)] -= diff;
            }
            A[static_cast<std::size_t>(i)] += 0.5 * cr / 3;
        }
    }
    std::vector<float> H(static_cast<std::size_t>(op.nV));
    for (int v = 0; v < op.nV; ++v) {
        const double a = A[static_cast<std::size_t>(v)] > 0 ? A[static_cast<std::size_t>(v)] : 1;
        const auto& N = op.vertexNormals;
        H[static_cast<std::size_t>(v)] = static_cast<float>(-0.5 * (DX[static_cast<std::size_t>(3 * v)] * N[static_cast<std::size_t>(3 * v)] +
                                                                   DX[static_cast<std::size_t>(3 * v + 1)] * N[static_cast<std::size_t>(3 * v + 1)] +
                                                                   DX[static_cast<std::size_t>(3 * v + 2)] * N[static_cast<std::size_t>(3 * v + 2)]) / a);
    }
    return H;
}

std::vector<float> formValues(FieldType t, Form id, const float* values, std::size_t length, const FaceGradientOperator& op) {
    const std::size_t n = static_cast<std::size_t>(op.nV), W = widthOf(t, n);
    const auto info = formInfo(t, id);
    if (!info) throw std::invalid_argument("formValues: this field type has no such form");
    if (W == 0 || length % W != 0) throw std::invalid_argument("formValues: the values are not whole frames of the field");
    const std::size_t frames = std::max<std::size_t>(1, length / W);
    // a scalar form is n a frame, every vector form 3n (ℍ's `x` too: its vector part, as the Studio stacks it)
    const std::size_t outPer = info->out == FieldType::Real ? n : 3 * n;
    std::vector<float> out(frames * outPer, 0.0f);
    const auto& N = op.vertexNormals;
    for (std::size_t fr = 0; fr < frames; ++fr) {
        const float* a = values + fr * W;
        if (t == FieldType::Real) { std::copy(a, a + n, out.begin() + static_cast<std::ptrdiff_t>(fr * n)); continue; }
        if (t == FieldType::Complex) {
            if (id == Form::abs) {
                for (std::size_t k = 0; k < n; ++k) out[fr * n + k] = static_cast<float>(jsHypot({a[3 * k], a[3 * k + 1], a[3 * k + 2]}));
            } else if (isDivCurl(id)) {
                const std::vector<float> r = divCurlOf(id, a, op);
                std::copy(r.begin(), r.end(), out.begin() + static_cast<std::ptrdiff_t>(fr * n));
            } else if (id == Form::v) {
                std::copy(a, a + 3 * n, out.begin() + static_cast<std::ptrdiff_t>(fr * 3 * n));
            } else {   // Jv = n × v
                for (std::size_t k = 0; k < n; ++k) {
                    const double nx = N[3 * k], ny = N[3 * k + 1], nz = N[3 * k + 2], x = a[3 * k], y = a[3 * k + 1], z = a[3 * k + 2];
                    const std::size_t b = fr * 3 * n + 3 * k;
                    out[b] = static_cast<float>(ny * z - nz * y);
                    out[b + 1] = static_cast<float>(nz * x - nx * z);
                    out[b + 2] = static_cast<float>(nx * y - ny * x);
                }
            }
            continue;
        }
        if (isDivCurl(id)) {   // ℍ: the operator on the vector part x
            std::vector<float> x(3 * n);
            for (std::size_t k = 0; k < n; ++k) { x[3 * k] = a[4 * k + 1]; x[3 * k + 1] = a[4 * k + 2]; x[3 * k + 2] = a[4 * k + 3]; }
            const std::vector<float> r = divCurlOf(id, x.data(), op);
            std::copy(r.begin(), r.end(), out.begin() + static_cast<std::ptrdiff_t>(fr * n));
            continue;
        }
        for (std::size_t k = 0; k < n; ++k) {
            const double w = a[4 * k], x = a[4 * k + 1], y = a[4 * k + 2], z = a[4 * k + 3];
            const double nx = N[3 * k], ny = N[3 * k + 1], nz = N[3 * k + 2], dn = x * nx + y * ny + z * nz;
            if (id == Form::absq) out[fr * n + k] = static_cast<float>(jsHypot({w, x, y, z}));
            else if (id == Form::nx) out[fr * n + k] = static_cast<float>(dn);
            else if (id == Form::w) out[fr * n + k] = static_cast<float>(w);
            else {
                const std::size_t b = fr * 3 * n + 3 * k;
                const double t2 = id == Form::xt ? dn : 0;
                out[b] = static_cast<float>(x - t2 * nx);
                out[b + 1] = static_cast<float>(y - t2 * ny);
                out[b + 2] = static_cast<float>(z - t2 * nz);
            }
        }
    }
    return out;
}

ComplexForm formComplex(FieldType t, Form id, const float* re, const float* im, std::size_t length, const FaceGradientOperator& op) {
    ComplexForm c;
    if (isLinear(id)) {
        c.re = formValues(t, id, re, length, op);
        c.im = formValues(t, id, im, length, op);
        return c;
    }
    // the ENVELOPE: |v| over C^3, |q| over C^4 (every real and imaginary component), real with no phase
    if (!formInfo(t, id)) throw std::invalid_argument("formComplex: this field type has no such form");
    const std::size_t n = static_cast<std::size_t>(op.nV), W = widthOf(t, n), per = W / n;
    if (W == 0 || length % W != 0) throw std::invalid_argument("formComplex: the values are not whole frames of the field");
    const std::size_t frames = length / W;
    c.re.assign(frames * n, 0.0f);
    for (std::size_t fr = 0; fr < frames; ++fr)
        for (std::size_t k = 0; k < n; ++k) {
            double mx = 0, sum = 0;
            const std::size_t o = fr * W + per * k;
            for (std::size_t d = 0; d < per; ++d) mx = std::max({mx, std::fabs(static_cast<double>(re[o + d])), std::fabs(static_cast<double>(im[o + d]))});
            if (mx > 0)
                for (std::size_t d = 0; d < per; ++d) { const double a = re[o + d] / mx, b = im[o + d] / mx; sum += a * a + b * b; }
            c.re[fr * n + k] = static_cast<float>(std::sqrt(sum) * mx);
        }
    return c;
}

Eigen::MatrixXd formOfFreeColumns(Form id, const Eigen::MatrixXd& K, const FaceGradientOperator& op) {
    const auto info = formInfo(FieldType::Quaternion, id);
    if (!info || info->out != FieldType::Real || !isLinear(id)) throw std::invalid_argument("formOfFreeColumns: not a linear scalar form of a free current");
    if (K.rows() != 3 * static_cast<Eigen::Index>(op.nV)) throw std::invalid_argument("formOfFreeColumns: K has " + std::to_string(K.rows()) + " rows, the surface needs 3 x " + std::to_string(op.nV));
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(op.nV, K.cols());
    std::vector<float> q(widthOf(FieldType::Quaternion, static_cast<std::size_t>(op.nV)), 0.0f);
    for (Eigen::Index c = 0; c < K.cols(); ++c) {
        if (K.col(c).isZero(0)) continue;                       // a channel the kernel does not use: its form is 0
        for (int v = 0; v < op.nV; ++v)
            for (int a = 0; a < 3; ++a) q[static_cast<std::size_t>(4 * v + 1 + a)] = static_cast<float>(K(3 * v + a, c));
        const std::vector<float> f = formValues(FieldType::Quaternion, id, q.data(), q.size(), op);
        for (int v = 0; v < op.nV; ++v) out(v, c) = f[static_cast<std::size_t>(v)];
    }
    return out;
}

}  // namespace nxr::forms
