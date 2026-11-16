// nxr::frames: the canonical frames (north / west) and the anatomical frames (anterior / superior-or-medial).
// (1) PARITY with Rheome's own code: golden/gen_frames_golden.m runs rheome.geom.sphereframe (axis +z and +x) on a synthetic registered-sphere "brain"; e1, e2, colatitude, longitude, condition, singular, poles agree;
// (2) the analytic north of an ellipsoid whose registered sphere is its pre-image; e2 = n x e1;
// (3) the anatomical frame on a two-hemisphere ellipsoid "brain" in SCS (x anterior, y left, z up): anterior = +x on the lateral surface, the second axis SUPERIOR on lateral / medial cortex and MEDIAL on
//     dorsal / ventral cortex with the right sign on BOTH hemispheres (the mirror), the mask covers the frontal and occipital poles;
// (4) the template route: axes defined once on a template in RAS (a different mesh), carried through the registered sphere, equal the SCS-native ones;
// (5) the components are invariant under a rotation of the gauge, and equal their gauge-angle form (ChartKit::gaugeHeading).
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

#include "golden_io.h"
#include "mesh_helpers.h"
#include "brain_helpers.h"
#include "nxr/frames.h"

using namespace nxr::frames;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

static double wrapd(double a) { return std::remainder(a, 2 * M_PI); }
static double maxDiffNaN(const MatrixXd& a, const MatrixXd& b) {
    double d = 0;
    for (Eigen::Index j = 0; j < a.cols(); ++j) for (Eigen::Index i = 0; i < a.rows(); ++i) {
        const double x = a(i, j), y = b(i, j);
        if (std::isnan(x) || std::isnan(y)) { if (std::isnan(x) != std::isnan(y)) d = 1e9; continue; }
        d = std::max(d, std::abs(x - y));
    }
    return d;
}

static void parity() {
    Golden g(std::string(NXR_TEST_GOLDEN_DIR) + "/frames_golden.txt");
    if (!g.good()) { std::cout << "  [FAIL] frames_golden.txt not found\n"; ++g_failures; return; }
    const MatrixXd P = g.mat("positions"), Fm = g.mat("faces"), N = g.mat("normals"), S = g.mat("sphere");
    const int n = static_cast<int>(P.rows()), nF = static_cast<int>(Fm.rows());
    std::vector<double> p(3 * n), nn(3 * n), s(3 * n);
    std::vector<int32_t> f(3 * nF);
    for (int i = 0; i < n; ++i) for (int k = 0; k < 3; ++k) { p[3 * i + k] = P(i, k); nn[3 * i + k] = N(i, k); s[3 * i + k] = S(i, k); }
    for (int i = 0; i < nF; ++i) for (int k = 0; k < 3; ++k) f[3 * i + k] = static_cast<int32_t>(Fm(i, k));
    for (const char* name : {"+z", "+x"}) {
        const MatrixXd e1 = g.mat("e1"), e2 = g.mat("e2"), colat = g.mat("colat"), lon = g.mat("lon"), cond = g.mat("condition"), sing = g.mat("singular"), poles = g.mat("poles");
        const Canonical c = canonicalFrames(p.data(), n, f.data(), nF, nn.data(), s.data(), std::string(name) == "+z" ? Vector3d(0, 0, 1) : Vector3d(1, 0, 0));
        int nSing = 0, sd = 0;
        for (int i = 0; i < n; ++i) { nSing += c.singular[i]; sd += (c.singular[i] != 0) != (sing(i, 0) != 0); }
        const double de = std::max(maxDiffNaN(c.e1, e1), maxDiffNaN(c.e2, e2));
        std::cout << "  axis " << name << ": " << nSing << " singular; max |diff| e1,e2 " << sci(de) << ", colat " << sci(maxDiffNaN(c.colat, colat)) << ", lon " << sci(maxDiffNaN(c.lon, lon))
                  << ", cond " << sci(relErr(c.condition, cond)) << "\n";
        EXPECT(de < 1e-9 && maxDiffNaN(c.colat, colat) < 1e-12 && maxDiffNaN(c.lon, lon) < 1e-12 && relErr(c.condition, cond) < 1e-9 && sd == 0 && c.north == static_cast<int>(poles(0, 0)) && c.south == static_cast<int>(poles(0, 1)),
               std::string("axis ") + name + ": equals Rheome's sphereframe (frames, colat, lon, condition, singular set, poles)");
    }
}

static Vector3d rowv(const Mat3& A, int i) { return A.row(i).transpose(); }

static void north() {
    const Brain b = brain(3, false, 0.0);
    std::vector<double> nn(3 * b.n);
    for (int i = 0; i < b.n; ++i) for (int k = 0; k < 3; ++k) nn[3 * i + k] = b.N(i, k);
    const Canonical c = canonicalFrames(b.P.data(), b.n, b.F.data(), b.nF, nn.data(), b.Sph.data());
    // the push-forward of the meridian of the pre-image sphere by x = D u (hemisphere shifts do not change a tangent): north = normalize(D m), m = z - (u.z) u
    double worst = 0, ortho = 0; int used = 0;
    const Vector3d D(2.0, 1.0, 1.5);
    for (int i = 0; i < b.n; ++i) {
        if (c.singular[i] || c.colat[i] < 0.35 || c.colat[i] > M_PI - 0.35) continue;      // the meridians converge at the poles: a coarse mesh cannot push them forward
        const Vector3d u(b.Sph[3 * i], b.Sph[3 * i + 1], b.Sph[3 * i + 2]);
        Vector3d m = Vector3d(0, 0, 1) - u.z() * u;
        m = (D.cwiseProduct(m)).normalized();
        worst = std::max(worst, std::acos(std::min(1.0, rowv(c.e1, i).dot(m))));
        ortho = std::max({ortho, std::abs(rowv(c.e1, i).dot(rowv(b.N, i))), std::abs(rowv(c.e2, i).dot(rowv(c.e1, i))), std::abs(rowv(c.e2, i).dot(rowv(b.N, i)))});
        ++used;
    }
    std::cout << "  ellipsoid north: " << used << " vertices, worst angle to the analytic push-forward of the meridian " << worst * 180 / M_PI << " deg; orthogonality " << sci(ortho) << "\n";
    EXPECT(used > b.n / 2 && worst < 3 * M_PI / 180 && ortho < 1e-12, "north is the push-forward of the registered sphere's meridian (<3 deg at level 3), e1 e2 n orthonormal");
}

int main() {
    std::cout << "frames: canonical (parity with Rheome), anatomical (axes, mirror, mask, template route, gauge)\n";
    parity();
    north();

    const Brain b = brain(3, false);
    std::vector<double> nn(3 * b.n);
    for (int i = 0; i < b.n; ++i) for (int k = 0; k < 3; ++k) nn[3 * i + k] = b.N(i, k);
    const Canonical cf = canonicalFrames(b.P.data(), b.n, b.F.data(), b.nF, nn.data(), b.Sph.data());
    AnatomicalOptions ao;                                       // SCS axes, hemispheres by piece
    const Anatomical A = anatomicalFrame(b.P.data(), b.n, b.F.data(), b.nF, b.N, cf.e1, cf.e2, ao);
    EXPECT(A.definition == "scs" && A.reference == "canonical frames", "the SCS-native definition is recorded");
    auto anterior3 = [&](const Anatomical& a, const Canonical& c, int i) { return Vector3d(std::cos(a.angle[i]) * rowv(c.e1, i) + std::sin(a.angle[i]) * rowv(c.e2, i)); };
    auto second3 = [&](const Anatomical& a, const Canonical& c, int i) { return Vector3d(a.handed[i] * rowv(b.N, i).cross(anterior3(a, c, i))); };
    int nLat = 0, nDor = 0, badLat = 0, badDor = 0, labelBad = 0, handedL = 0, handedR = 0, nLatL = 0, nLatR = 0;
    int poleCount = 0, poleBad = 0, relBad = 0;
    for (int i = 0; i < b.n; ++i) {
        const Vector3d n = rowv(b.N, i);
        const bool left = i < b.nHalf;
        if (std::abs(n.x()) > 0.95) { ++poleCount; poleBad += A.reliable[i]; }
        if (A.reliable[i] && std::abs(n.x()) >= 0.945) ++relBad;
        if (!A.reliable[i]) continue;
        const Vector3d a = anterior3(A, cf, i), s = second3(A, cf, i);
        if (std::abs(n.y()) > 0.9) {                                // lateral (facing away from the midline) or medial (facing it): the second axis is SUPERIOR
            ++nLat;
            if (A.second[i] != static_cast<int8_t>(Second::Superior)) ++labelBad;
            if (a.dot(Vector3d(1, 0, 0)) < 0.9 || s.dot(Vector3d(0, 0, 1)) < 0.7) ++badLat;
            if (n.y() * (left ? 1 : -1) > 0) { (left ? nLatL : nLatR)++; (left ? handedL : handedR) += A.handed[i]; }
        } else if (std::abs(n.z()) > 0.9) {                         // dorsal / ventral: the second axis is MEDIAL
            ++nDor;
            if (A.second[i] != static_cast<int8_t>(Second::Medial)) ++labelBad;
            if (a.dot(Vector3d(1, 0, 0)) < 0.9 || s.dot(Vector3d(0, left ? -1 : 1, 0)) < 0.5) ++badDor;
        }
    }
    std::cout << "  SCS-native: " << nLat << " lateral/medial and " << nDor << " dorsal/ventral reliable vertices; poles " << poleCount << "\n";
    EXPECT(nLat > 50 && nDor > 50 && badLat == 0 && badDor == 0 && labelBad == 0, "anterior = +x, second axis superior on lateral / medial and medial (towards the midline) on dorsal / ventral, labels right, on BOTH hemispheres");
    EXPECT(nLatL > 10 && nLatR > 10 && handedL == -nLatL && handedR == nLatR, "mirror: on lateral cortex the second axis is -(n x anterior) on the left hemisphere and +(n x anterior) on the right");
    EXPECT(poleCount > 10 && poleBad == 0 && relBad == 0, "the reliability mask covers the frontal and occipital poles (and only where the anterior projection is short)");

    // the template route: a finer template (level 4) in RAS (x right, y anterior), its axes defined once, carried through the registered spheres
    const Brain t = brain(4, true);
    std::vector<double> tn(3 * t.n);
    for (int i = 0; i < t.n; ++i) for (int k = 0; k < 3; ++k) tn[3 * i + k] = t.N(i, k);
    const Canonical tc = canonicalFrames(t.P.data(), t.n, t.F.data(), t.nF, tn.data(), t.Sph.data());
    AnatomicalOptions to;
    to.axes.anterior = Vector3d(0, 1, 0); to.axes.superior = Vector3d(0, 0, 1); to.axes.left = Vector3d(-1, 0, 0);
    const Anatomical TA = anatomicalFrame(t.P.data(), t.n, t.F.data(), t.nF, t.N, tc.e1, tc.e2, to);
    const Anatomical CA = carryAnatomical(TA, t.Sph.data(), t.n, pieces(t.F.data(), t.nF, t.n), b.Sph.data(), b.n, pieces(b.F.data(), b.nF, b.n), cf.e1);
    std::vector<double> dif;
    int agree = 0, both = 0;
    for (int i = 0; i < b.n; ++i) {
        if (!A.reliable[i] || !CA.reliable[i] || A.projection[i] < 0.6) continue;     // away from the poles, where the direction turns fast
        ++both;
        dif.push_back(std::abs(wrapd(CA.angle[i] - A.angle[i])));
        agree += CA.second[i] == A.second[i] && CA.handed[i] == A.handed[i];
    }
    std::sort(dif.begin(), dif.end());
    const double med = dif[dif.size() / 2] * 180 / M_PI, p95 = dif[dif.size() * 95 / 100] * 180 / M_PI;
    std::cout << "  template route: " << both << " vertices reliable in both; angle difference median " << med << " deg, p95 " << p95 << " deg; label+handedness agree " << agree << "\n";
    EXPECT(CA.definition == "template" && both > b.n / 2 && med < 3 && p95 < 10 && agree >= 0.97 * both, "the template's axes (RAS, defined once) carried through the registered sphere equal the subject's SCS-native axes");

    // the components under a gauge rotation, and in gauge angles
    std::mt19937 rng(3);
    std::uniform_real_distribution<double> ud(-M_PI, M_PI);
    Mat3 e1r = cf.e1, e2r = cf.e2, vec = Mat3::Zero(b.n, 3);
    Anatomical Ar = A;
    for (int i = 0; i < b.n; ++i) {
        if (cf.singular[i]) continue;
        const double rho = ud(rng);
        e1r.row(i) = std::cos(rho) * cf.e1.row(i) + std::sin(rho) * cf.e2.row(i);
        e2r.row(i) = rowv(b.N, i).cross(rowv(e1r, i)).transpose();
        Ar.angle[i] = A.angle[i] - rho;
        vec.row(i) = (ud(rng) * cf.e1.row(i) + ud(rng) * cf.e2.row(i));
    }
    const auto c0 = anatomicalComponents(A, cf.e1, cf.e2, vec), c1 = anatomicalComponents(Ar, e1r, e2r, vec);
    EXPECT(maxDiffNaN(c0, c1) < 1e-12, "the anatomical components do not change when the reference frame is rotated (the angle absorbs it)");
    nxr::chart::ChartKit kit(b.P.data(), b.n, b.F.data(), b.nF);
    const VectorXd head = angleFromNorth(kit, vec, cf.e1);
    const VectorXd viaGauge = angleFromAnterior(A, head), via3d = angleFromAnterior(c0);
    double dg = 0; int cnt = 0;
    for (int i = 0; i < b.n; ++i) if (std::isfinite(viaGauge[i]) && std::isfinite(via3d[i]) && vec.row(i).norm() > 1e-3) { dg = std::max(dg, std::abs(wrapd(viaGauge[i] - via3d[i]))); ++cnt; }
    std::cout << "  gauge form: " << cnt << " vertices, max |angle difference| " << sci(dg) << " rad\n";
    EXPECT(cnt > b.n / 2 && dg < 2e-3, "the angle from anterior via the signpost gauge (gaugeHeading) equals the one from the 3D components");
    if (g_failures) { std::cout << g_failures << " failure(s)\n"; return 1; }
    std::cout << "all frames tests passed\n";
    return 0;
}
