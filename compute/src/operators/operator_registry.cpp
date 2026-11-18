#include "nxr/operators/registry/operator_registry.h"
#include "nxr/core/errors.h"

namespace nxr::operators::registry {

using nxr::core::Error;
using nxr::core::ErrorCode;

static CrossLink squaresTo(std::string t, Relation r = Relation::exact)  { return {true,  true,  std::move(t), r}; }
static CrossLink squareOf (std::string t, Relation r = Relation::exact)  { return {true,  false, std::move(t), r}; }

const std::vector<OperatorVariant>& operatorRegistry() {
    static const std::vector<OperatorVariant> table = [] {
        std::vector<OperatorVariant> t = {
            // ── scalar ──
            { "laplaceBeltrami", "Laplace-Beltrami (cotan)", Bundle::scalar, Holonomy::intrinsic_curved,
              Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
              squareOf("intrinsicDirac", Relation::principal_part), "massMixedVoronoi", false, "", Status::built, geometry::OperatorId::LaplacianCotan, "" },
            { "graphLaplacian", "Graph Laplacian (d0Td0)", Bundle::scalar, Holonomy::combinatorial,
              Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
              squareOf("d0", Relation::exact), "identity", false, "", Status::built, geometry::OperatorId::LaplacianGraph, "" },
            { "faceLaplacianGreenGauss", "Face Laplacian (Green-Gauss dual)", Bundle::scalar, Holonomy::intrinsic_curved,
              Order::second, Role::laplacian, FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na,
              squareOf("faceGradient", Relation::exact), "", false, "", Status::built, geometry::OperatorId::LapFace, "" },
            { "faceLaplacian2Form", "Face Laplacian (DEC 2-form, d1*1inv*d1T)", Bundle::scalar, Holonomy::intrinsic_curved,
              Order::second, Role::laplacian, FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na,
              {}, "", false, "", Status::built, geometry::OperatorId::DiracFace, "relativeFaceDirac tau=0 anchor; internal cacheTwoFormLaplacian_" },

            // ── tangent (complex, nSym) ──
            { "leviCivitaConnectionLaplacian", "Levi-Civita connection (Bochner) Laplacian", Bundle::tangent, Holonomy::intrinsic_curved,
              Order::second, Role::connection_laplacian, FieldType::complex, Domain::vertex, Singular::none, Gauge::levi_civita, Coupling::na,
              {}, "massMixedVoronoi*I2", false, "", Status::built, geometry::OperatorId::LaplacianConnection, "domain in {vertex,face,edge}; real2N BLOCK [re; im]; mass MIXED VORONOI — galerkin*I2 couples frames without transport" },
            { "trivialConnectionLaplacian", "Trivial connection Laplacian", Bundle::tangent, Holonomy::flat,
              Order::second, Role::connection_laplacian, FieldType::complex, Domain::vertex, Singular::chi_defects, Gauge::trivial, Coupling::na,
              {}, "massMixedVoronoi*I2", false, "", Status::built, geometry::OperatorId::LaplacianConnection, "Sum singularity index == chi (Gauss-Bonnet); mass MIXED VORONOI as leviCivitaConnectionLaplacian" },

            // ── tangent gradient (complex, nSym) ──
            { "leviCivitaConnectionGradient", "Levi-Civita connection gradient (d^nabla)", Bundle::tangent, Holonomy::intrinsic_curved,
              Order::first, Role::gradient, FieldType::complex, Domain::edge, Singular::none, Gauge::levi_civita, Coupling::na,
              squaresTo("leviCivitaConnectionLaplacian", Relation::exact), "", false, "", Status::built, geometry::OperatorId::ConnectionGradient,
              "edge<-vertex; nSym. Levi-Civita gauge only (v1) — assembleConnectionGradient reads the LC transport; the trivial-gauge gradient (phi-corrected root of trivialConnectionLaplacian) is a deferred follow-up." },

            // ── ambient (real, 3-comp) ──
            { "flatCovariantLaplacian", "Flat covariant Laplacian (ambient)", Bundle::ambient, Holonomy::flat,
              Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::ambient,
              squareOf("covariantGradient", Relation::exact), "", false, "", Status::built, geometry::OperatorId::LaplacianCovariant, "" },
            { "productCovariantLaplacian", "Product covariant Laplacian (tan+nor)", Bundle::ambient, Holonomy::intrinsic_curved,
              Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::product,
              {}, "", false, "", Status::built, geometry::OperatorId::LaplacianCovariant, "" },
            { "covariantGradient", "Covariant gradient (flat transport)", Bundle::ambient, Holonomy::flat,
              Order::first, Role::gradient, FieldType::real, Domain::edge, Singular::none, Gauge::na, Coupling::na,
              squaresTo("flatCovariantLaplacian", Relation::exact), "", false, "", Status::built, geometry::OperatorId::Gradient3D, "edge<-vertex" },
            { "faceGradient", "Face gradient (Green-Gauss)", Bundle::ambient, Holonomy::intrinsic_curved,
              Order::first, Role::gradient, FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na,
              squaresTo("faceLaplacianGreenGauss", Relation::exact), "", false, "", Status::built, geometry::OperatorId::GradFace, "" },
            { "extrinsicWeitzenbockLaplacian", "Extrinsic Weitzenbock Laplacian (D3+D_N)", Bundle::ambient, Holonomy::extrinsic_curved,
              Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
              {}, "", false, "", Status::built, geometry::OperatorId::ExtrinsicWeitzenbock,
              "imag (vector) block of the immersion squared Dirac 2*dirac(0.5)=D4+D_N; world-frame component-major. Flat part == kron(I3,cotanL)." },

            // ── immersion (quaternion, 4-comp) ──
            { "intrinsicDirac", "Intrinsic Dirac (1st-order)", Bundle::immersion, Holonomy::intrinsic_curved,
              Order::first, Role::dirac, FieldType::quaternion, Domain::face, Singular::none, Gauge::na, Coupling::na,
              squaresTo("laplaceBeltrami", Relation::principal_part), "", false, "", Status::built, geometry::OperatorId::DiracIntrinsicD, "face<-vertex; scalar part of square == cotanL" },
            { "extrinsicDirac", "Extrinsic Dirac (1st-order, Gauss map)", Bundle::immersion, Holonomy::extrinsic_curved,
              Order::first, Role::dirac, FieldType::quaternion, Domain::face, Singular::none, Gauge::na, Coupling::na,
              squaresTo("relativeDirac", Relation::exact), "", false, "", Status::built, geometry::OperatorId::DiracD, "face<-vertex; squares to relativeDirac@tau=1 (E)" },
            { "relativeDirac", "Relative Dirac (vertex tau-family)", Bundle::immersion, Holonomy::graded,
              Order::second, Role::dirac, FieldType::quaternion, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
              squareOf("extrinsicDirac", Relation::exact), "massMixedVoronoi*I4", true, "intrinsic=0;extrinsic=1;squared=0.5x2", Status::built, geometry::OperatorId::Dirac, "" },
            { "intrinsicFaceDirac", "Intrinsic face Dirac (1st-order)", Bundle::immersion, Holonomy::intrinsic_curved,
              Order::first, Role::dirac, FieldType::quaternion, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
              squaresTo("faceLaplacian2Form", Relation::principal_part), "", false, "", Status::built, geometry::OperatorId::DiracFaceIntrinsicD, "vertex<-face; closed-mesh v1" },
            { "extrinsicFaceDirac", "Extrinsic face Dirac (1st-order)", Bundle::immersion, Holonomy::extrinsic_curved,
              Order::first, Role::dirac, FieldType::quaternion, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
              squaresTo("relativeFaceDirac", Relation::exact), "", false, "", Status::built, geometry::OperatorId::DiracFaceD, "vertex<-face; closed-mesh v1" },
            { "relativeFaceDirac", "Relative face Dirac (tau-family)", Bundle::immersion, Holonomy::graded,
              Order::second, Role::dirac, FieldType::quaternion, Domain::face, Singular::none, Gauge::na, Coupling::na,
              squareOf("extrinsicFaceDirac", Relation::exact), "diag(faceArea)*I4", true, "intrinsic=0;extrinsic=1;squared=0.5x2", Status::built, geometry::OperatorId::DiracFace, "closed-mesh v1" },

            // ── eigen-solved Diracs on forms / face spinors, and the degree-mass graph Laplacian ──
            { "graphLaplacianDegreeMass", "Graph Laplacian, degree mass (L = D - A, L phi = lambda D phi)", Bundle::scalar, Holonomy::combinatorial,
              Order::second, Role::laplacian, FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na,
              {}, "degree", false, "", Status::built, geometry::OperatorId::LaplacianGraph, "unweighted topological Laplacian; eigs graphDegreeMass=true; same K as graphLaplacian, mass diag(degree)" },
            { "hodgeDirac", "Hodge-Dirac d+delta (0/1/2-forms, intrinsic Delaunay)", Bundle::scalar, Holonomy::intrinsic_curved,
              Order::first, Role::dirac, FieldType::real, Domain::mixed, Singular::none, Gauge::na, Coupling::na,
              {}, "diag(hodge0,hodge1,hodge2)", false, "", Status::built, geometry::OperatorId::HodgeDirac,
              "real-valued forms (Bundle scalar, no tangent bundle), graded 0/1/2 (Domain mixed = V+E+F); input/output field = the 0-form (vertex) block, the eigenvector spans V+E+F: pencil (K,M) on the intrinsic Delaunay mesh (rows: vertices, edges, faces; edges/faces are the intrinsic mesh's); "
              "star1 raw cotan weight (optional floor, default off), mollified (default 1e-6). Spectrum +-mu; eigs default = k lowest +mu by exact/coexact split with kind per mode; "
              "explicit sigma = indefinite shift-invert; port of cortical-flow hodgeDirac.ts" },
            { "spinExtrinsicDirac", "Extrinsic Dirac of spin transformations D_f (Hoffmann-Ye), face spinors", Bundle::immersion, Holonomy::extrinsic_curved,
              Order::first, Role::dirac, FieldType::quaternion, Domain::face, Singular::none, Gauge::na, Coupling::na,
              {}, "diag(faceArea)*I4", false, "", Status::built, geometry::OperatorId::SpinDirac,
              "4F x 4F, 1/2 sum E_ij - H_i; closed mesh. eigs: Ye et al. quotient on VERTEX spinors (A^T D M^-1 D A psi = mu A^T D A psi), plain vertexToFace; port of cortical-flow spinDirac.ts" },
            { "spinIntrinsicHYDirac", "Intrinsic Dirac, Hoffmann-Ye (D_f + H), face spinors", Bundle::immersion, Holonomy::intrinsic_curved,
              Order::first, Role::dirac, FieldType::quaternion, Domain::face, Singular::none, Gauge::na, Coupling::na,
              {}, "diag(faceArea)*I4", false, "", Status::built, geometry::OperatorId::SpinDirac, "4F x 4F, 1/2 sum E_ij; quotient solve with the covariant vertexToFace" },
            { "spinIntrinsicYeDirac", "Intrinsic Dirac, Ye et al. (cos(theta/2) E_ij), face spinors", Bundle::immersion, Holonomy::intrinsic_curved,
              Order::first, Role::dirac, FieldType::quaternion, Domain::face, Singular::none, Gauge::na, Coupling::na,
              {}, "diag(faceArea)*I4", false, "", Status::built, geometry::OperatorId::SpinDirac, "4F x 4F, 1/2 sum cos(theta/2) E_ij (edge lengths only); quotient solve with the covariant vertexToFace" },

            // ── metrics & exterior calculus ──
            { "massMixedVoronoi", "Mixed Voronoi mass (Meyer et al. 2003) of the mollified input, recorded before the flips (the default)", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::mass,
              FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, geometry::OperatorId::MassMixedVoronoi, "holonomy n/a (metric)" },
            { "massBarycentric", "Barycentric mass (A/3 per vertex)", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::mass,
              FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, geometry::OperatorId::MassBarycentric, "holonomy n/a (metric)" },
            { "massGalerkin", "Galerkin (FEM) mass", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::mass,
              FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, geometry::OperatorId::MassGalerkin, "holonomy n/a (metric)" },
            { "d0", "Exterior derivative d0 (grad)", Bundle::scalar, Holonomy::combinatorial, Order::first, Role::exterior_derivative,
              FieldType::real, Domain::edge, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, geometry::OperatorId::Dec, "edge<-vertex" },
            { "d1", "Exterior derivative d1 (curl)", Bundle::scalar, Holonomy::combinatorial, Order::first, Role::exterior_derivative,
              FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, geometry::OperatorId::Dec, "face<-edge" },
            { "hodge0", "Hodge star star0", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::hodge_star,
              FieldType::real, Domain::vertex, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, geometry::OperatorId::Dec, "holonomy n/a (metric)" },
            { "hodge1", "Hodge star star1", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::hodge_star,
              FieldType::real, Domain::edge, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, geometry::OperatorId::Dec, "holonomy n/a (metric)" },
            { "hodge2", "Hodge star star2", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::hodge_star,
              FieldType::real, Domain::face, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, geometry::OperatorId::Dec, "holonomy n/a (metric)" },
            { "hodge1inv", "Hodge star star1 inverse", Bundle::scalar, Holonomy::flat, Order::zeroth, Role::hodge_star,
              FieldType::real, Domain::edge, Singular::none, Gauge::na, Coupling::na, {}, "", false, "", Status::built, geometry::OperatorId::Dec, "holonomy n/a (metric)" },
        };
        // Field I/O declared adjacent, keyed by curated id (NOT geometry::OperatorId: DiracFace
        // backs both faceLaplacian2Form and relativeFaceDirac with different I/O).
        struct IO { const char* id; const char* in; const char* out; };
        static const IO io[] = {
            { "laplaceBeltrami",                "scalarVertex",      "scalarVertex" },
            { "graphLaplacian",                 "scalarVertex",      "scalarVertex" },
            { "faceLaplacianGreenGauss",        "twoFormFace",       "twoFormFace" },
            { "faceLaplacian2Form",             "twoFormFace",       "twoFormFace" },
            { "leviCivitaConnectionLaplacian",  "tangentVertex",     "tangentVertex" },
            { "trivialConnectionLaplacian",     "tangentVertex",     "tangentVertex" },
            { "leviCivitaConnectionGradient",   "tangentVertex",     "tangentEdge" },
            { "flatCovariantLaplacian",         "ambientVertexLocal","ambientVertexLocal" },
            { "productCovariantLaplacian",      "ambientVertexLocal","ambientVertexLocal" },
            { "covariantGradient",              "ambientVertexLocal","ambientEdge" },
            { "faceGradient",                   "twoFormFace",       "ambientFaceWorld" },
            { "extrinsicWeitzenbockLaplacian",  "ambientVertexWorld","ambientVertexWorld" },
            { "intrinsicDirac",                 "immersionVertex",   "immersionFace" },
            { "extrinsicDirac",                 "immersionVertex",   "immersionFace" },
            { "relativeDirac",                  "immersionVertex",   "immersionVertex" },
            { "intrinsicFaceDirac",             "immersionFace",     "immersionVertex" },
            { "extrinsicFaceDirac",             "immersionFace",     "immersionVertex" },
            { "relativeFaceDirac",              "immersionFace",     "immersionFace" },
            { "massMixedVoronoi",               "scalarVertex",      "scalarVertex" },
            { "massBarycentric",                "scalarVertex",      "scalarVertex" },
            { "massGalerkin",                   "scalarVertex",      "scalarVertex" },
            { "graphLaplacianDegreeMass",       "scalarVertex",      "scalarVertex" },
            { "hodgeDirac",                     "scalarVertex",       "scalarVertex" },
            { "spinExtrinsicDirac",             "immersionFace",     "immersionFace" },
            { "spinIntrinsicHYDirac",           "immersionFace",     "immersionFace" },
            { "spinIntrinsicYeDirac",           "immersionFace",     "immersionFace" },
            { "d0",                             "scalarVertex",      "oneFormEdge" },
            { "d1",                             "oneFormEdge",       "twoFormFace" },
            { "hodge0",                         "scalarVertex",      "scalarVertex" },
            { "hodge1",                         "oneFormEdge",       "oneFormEdge" },
            { "hodge2",                         "twoFormFace",       "twoFormFace" },
            { "hodge1inv",                      "oneFormEdge",       "oneFormEdge" },
        };
        for (auto& e : t)
            for (const auto& x : io)
                if (e.id == x.id) { e.input_field = x.in; e.output_field = x.out; break; }
        return t;
    }();
    return table;
}

const OperatorVariant* operatorById(std::string_view id) {
    for (const auto& v : operatorRegistry()) if (v.id == id) return &v;
    return nullptr;
}

std::vector<const OperatorVariant*>
operatorsWhere(const std::function<bool(const OperatorVariant&)>& pred) {
    std::vector<const OperatorVariant*> out;
    for (const auto& v : operatorRegistry()) if (pred(v)) out.push_back(&v);
    return out;
}

void requireBundle(std::string_view id, Bundle required) {
    const OperatorVariant* v = operatorById(id);
    if (!v)
        throw Error(ErrorCode::InvalidInput,
                    "unknown operator id: " + std::string(id),
                    "check nxr::manifold::registry::operatorRegistry()");
    if (v->bundle != required)
        throw Error(ErrorCode::InvalidInput,
                    std::string("operator '") + std::string(id) + "' has bundle " +
                        toString(v->bundle) + ", required " + toString(required),
                    "immersion operators are not valid ambient differentiators (LJC trap)");
}

// No `default:` — adding an geometry::OperatorId enumerator without a branch triggers
// -Wswitch (promoted to error for this TU) so metadata can't be forgotten.
std::vector<std::string> variantIdsFor(geometry::OperatorId op) {
    switch (op) {
        case geometry::OperatorId::LaplacianCotan:      return {"laplaceBeltrami"};
        case geometry::OperatorId::LaplacianGraph:      return {"graphLaplacian", "graphLaplacianDegreeMass"};
        case geometry::OperatorId::HodgeDirac:          return {"hodgeDirac"};
        case geometry::OperatorId::SpinDirac:           return {"spinExtrinsicDirac", "spinIntrinsicHYDirac", "spinIntrinsicYeDirac"};
        case geometry::OperatorId::LaplacianConnection: return {"leviCivitaConnectionLaplacian", "trivialConnectionLaplacian"};
        case geometry::OperatorId::LaplacianCovariant:  return {"flatCovariantLaplacian", "productCovariantLaplacian"};
        case geometry::OperatorId::Dec:                 return {"d0", "d1", "hodge0", "hodge1", "hodge2", "hodge1inv"};
        case geometry::OperatorId::MassMixedVoronoi:    return {"massMixedVoronoi"};
        case geometry::OperatorId::MassBarycentric:     return {"massBarycentric"};
        case geometry::OperatorId::MassGalerkin:        return {"massGalerkin"};
        case geometry::OperatorId::Gradient3D:          return {"covariantGradient"};
        case geometry::OperatorId::ExtrinsicWeitzenbock: return {"extrinsicWeitzenbockLaplacian"};
        case geometry::OperatorId::Dirac:               return {"relativeDirac"};
        case geometry::OperatorId::DiracFace:           return {"relativeFaceDirac", "faceLaplacian2Form"};
        case geometry::OperatorId::DiracD:              return {"extrinsicDirac"};
        case geometry::OperatorId::DiracFaceD:          return {"extrinsicFaceDirac"};
        case geometry::OperatorId::DiracIntrinsicD:     return {"intrinsicDirac"};
        case geometry::OperatorId::DiracFaceIntrinsicD: return {"intrinsicFaceDirac"};
        case geometry::OperatorId::GradFace:            return {"faceGradient"};
        case geometry::OperatorId::LapFace:             return {"faceLaplacianGreenGauss"};
        case geometry::OperatorId::ConnectionGradient:  return {"leviCivitaConnectionGradient"};
    }
    return {};   // unreachable; silences control-reaches-end warning
}

} // namespace nxr::manifold::registry
