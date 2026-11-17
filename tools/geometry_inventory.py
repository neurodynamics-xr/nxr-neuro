#!/usr/bin/env python3
"""Inventory of the discrete geometry that the geometry-central switch replaces (spec docs/specs/2026-11-17-geometry-central-switch.md).

Scans the tracked sources (git ls-files; the vendored compute/deps tree is skipped) for two families of symbols:
  PORT      everything built on the intrinsic Delaunay port (compute/include/nxr/delaunay.h) and on its gauge: the port's own API, the charts
            (nxr/chart.h), the frames that read the signpost gauge (nxr/frames.h, ladder, crossings), the optical-flow connection gauge, the
            Manifold's operator mesh and the port-gauge frames (vertexFrames, vertexGrid);
  GC-FRAME  the places that already use geometry-central's own gauge (vertexFramesEmbedded, vertexGridEmbedded, vertexTangentBasis,
            halfedgeVectorsInVertex, transportVectorsAlongHalfedge, vertexConnectionLaplacian).
Each symbol is mapped to the geometry-central API that replaces it, or to "rebuilt on geometry-central's triangulation" where there is none.
It also lists the goldens that the affected tests read (and their generators), and the stored tangent quantities named in pipelines/ and io/.

Comments are blanked before matching, so a symbol named only in a comment is not counted (except in the stored-names scan, which reads
string literals).

Usage:  tools/geometry_inventory.py [--repo .] [--section summary|symbols|files|goldens|stored|all]
Rerun after each PR of the switch: the PORT counts are zero since G7 (the port is deleted); GC marks what runs on geometry-central (G3, G4, G7).
"""
import argparse, collections, os, re, subprocess, sys

# (symbol id, family, regex, replacement in geometry-central)
SYMBOLS = [
    # ── the port itself (nxr/delaunay.h) ──
    ("delaunay.h include",        "PORT", r'#\s*include\s*"nxr/delaunay\.h"',        "geometrycentral/surface/signpost_intrinsic_triangulation.h (+ intrinsic_mollification.h)"),
    ("delaunay::Mesh",            "PORT", r'\bdelaunay::Mesh\b',                       "SignpostIntrinsicTriangulation (intrinsicMesh, edgeLengths, signpostAngle) over the input ManifoldSurfaceMesh"),
    ("delaunay::DelaunayOptions", "GC", r'\bDelaunayOptions\b',                      "nxr option struct kept (mollify -> mollifyIntrinsic relativeFactor; flipEps -> IntrinsicTriangulation::triangleTestEPS)"),
    ("delaunay::makeMesh",        "PORT", r'\bmakeMesh\s*\(',                          "makeManifoldSurfaceMeshAndGeometry / EdgeLengthGeometry"),
    ("delaunay::mollify",         "PORT", r'\bdelaunay::mollify\s*\(|\bmollify\s*\(\s*m\b', "mollifyIntrinsic(mesh, edgeLengths, relativeFactor) on the INPUT lengths, before the triangulation is built"),
    ("delaunay::addSignposts",    "PORT", r'\baddSignposts\s*\(',                      "SignpostIntrinsicTriangulation constructor (signpostAngle)"),
    ("delaunay::flipToDelaunay",  "PORT", r'\bdelaunay::flipToDelaunay\s*\(',                    "IntrinsicTriangulation::flipToDelaunay() (flips counted through edgeFlipCallbackList)"),
    ("delaunay::flip",            "PORT", r'\bdelaunay::flip\s*\(',                    "SignpostIntrinsicTriangulation::flipEdgeIfNotDelaunay / flipEdgeIfPossible"),
    ("delaunay::prepare",         "PORT", r'\bdelaunay::prepare\s*\(|\bprepare\s*\(\s*(positions|vertices|P\.data|V\.data)', "nxr IntrinsicSurface (new, thin): mollifyIntrinsic -> SignpostIntrinsicTriangulation -> mass recorded -> flipToDelaunay"),
    ("delaunay::cotanLaplacian",  "PORT", r'\bdelaunay::cotanLaplacian\s*\(',          "IntrinsicGeometryInterface::cotanLaplacian (requireCotanLaplacian)"),
    ("delaunay::edgeCotanWeights","PORT", r'\bdelaunay::edgeCotanWeights\s*\(',        "edgeCotanWeights (requireEdgeCotanWeights)"),
    ("delaunay::faceAreas",       "PORT", r'\bdelaunay::faceAreas\s*\(',               "faceAreas (requireFaceAreas)"),
    ("delaunay::galerkinMass",    "PORT", r'\bdelaunay::galerkinMass\s*\(',                      "vertexGalerkinMassMatrix (requireVertexGalerkinMassMatrix)"),
    ("delaunay::barycentricMass", "PORT", r'\bdelaunay::barycentricMass\s*\(',                   "vertexLumpedMassMatrix / vertexDualAreas (geometry-central's lumped mass IS barycentric)"),
    ("delaunay::mixedVoronoiMass","PORT", r'\bdelaunay::mixedVoronoiMass\s*\(',                  "NONE: rebuilt on geometry-central (cornerAngles, edgeLengths, faceAreas of the mollified input), served through an overridden computeVertexDualAreas"),
    ("Mesh::mass (recorded)",     "PORT", r'\b(om|mesh|m|opMesh|M)\s*\.\s*mass\b',     "NONE: the recorded input mass, served by the IntrinsicSurface's computeVertexDualAreas override"),
    ("delaunay::rescaledAngle",   "PORT", r'\brescaledAngle\s*\(',                     "halfedgeVectorsInVertex[he].arg() (signpost: signpostAngle / vertexAngleScaling)"),
    ("delaunay::reverseAngle",    "PORT", r'\breverseAngle\s*\(',                      "halfedgeVectorsInVertex[he.twin()].arg()"),
    ("delaunay::connectionLaplacian","PORT", r'\bdelaunay::connectionLaplacian\s*\(',  "vertexConnectionLaplacian (nSym = 1); nSym > 1 rebuilt from transportVectorsAlongHalfedge^n as direction_fields.cpp does"),
    ("delaunay::dec / Dec",       "PORT", r'\bdelaunay::dec\s*\(|\bdelaunay::Dec\b',   "requireDECOperators: d0, d1, hodge1, hodge2 (hodge0 from the chosen mass)"),
    ("delaunay::nonDelaunayEdges","PORT", r'\bnonDelaunayEdges\s*\(',                  "IntrinsicTriangulation::isDelaunay(e) counted"),
    ("delaunay immediates",       "PORT", r'\bdelaunay::(triangleArea|cornerAngle|halfCot|edgeWeight)\s*\(', "EdgeLengthGeometry::faceArea / cornerAngle / halfedgeCotanWeight / edgeCotanWeight"),
    ("Manifold::operatorMesh",    "PORT", r'\boperatorMesh\s*\(\s*\)',                 "Manifold::intrinsic() -> the IntrinsicSurface (its IntrinsicGeometryInterface)"),
    ("Manifold::operator* caches","GC", r'\boperator(Cotan|DecMatrices|DualAreas)\s*\(', "the IntrinsicSurface's require* quantities"),
    # ── charts on the port (nxr/chart.h) ──
    ("chart::HeatMethod",         "GC", r'\b(chart::)?HeatMethod\b(?!Distance)',     "HeatMethodDistanceSolver on the IntrinsicSurface (mass through computeVertexDualAreas)"),
    ("chart::ChartKit",           "GC", r'\bChartKit\b',                             "VectorHeatMethodSolver + HeatMethodDistanceSolver on the IntrinsicSurface, per connected piece"),
    ("ChartKit::transported",     "GC", r'\btransported\s*\(',                       "VectorHeatMethodSolver::transportTangentVector"),
    ("ChartKit::logMap",          "GC", r'\.\s*logMap\s*\(|\bLogMap\b',              "VectorHeatMethodSolver::computeLogMap (LogMapStrategy::VectorHeat)"),
    ("chart::tangentBasis / Basis","GC", r'\btangentBasis\s*\(|\bchart::Basis\b|\.\s*basis\s*\(\s*\)', "vertexTangentBasis (+ vertexNormals) of the input VertexPositionGeometry"),
    ("gaugeHeading / headingVector","GC", r'\bgaugeHeading\s*\(|\bheadingVector\s*\(', "atan2 on vertexTangentBasis (nxr, one line each)"),
    # ── frames that read the port's gauge ──
    ("frames::gaugeAngle",        "GC", r'\bgaugeAngle\s*\(',                        "canonical frame as angle array in geometry-central's gauge (vertexTangentBasis)"),
    ("frames::angleFromNorth",    "GC", r'\bangleFromNorth\s*\(',                    "frame.rotate(field, to=canonical, spin=1) node"),
    ("frames::vertexNormals",     "GC", r'\bframes::vertexNormals\s*\(|\bvertexNormals\s*\(\s*(positions|S\.vertices|dom\.)', "vertexNormals (angle-weighted) of the input VertexPositionGeometry"),
    ("ladder::leviCivita",        "GC", r'\bleviCivita\s*\(',                        "VectorHeatMethodSolver::transportTangentVector (unchanged maths, new gauge)"),
    ("geometry::vertexFrames",    "GC", r'\bvertexFrames\s*\(',                      "vertexTangentBasis (one frame; vertexFramesEmbedded becomes vertexFrames)"),
    ("geometry::vertexGrid",      "GC", r'\bvertexGrid\s*\((?!Embedded)',            "vertexTangentBasis packed e1 + i e2 (one grid; vertexGridEmbedded becomes vertexGrid)"),
    ("flow::connection::Corners", "GC", r'\bCorners\b|\bcomponentPencil\b',          "cornerScaledAngles + halfedgeVectorsInVertex of the mollified input geometry (rebuilt; same gauge as the triangulation)"),
    # ── geometry-central's gauge, used today ──
    ("vertexFramesEmbedded",      "GC-FRAME", r'\bvertexFramesEmbedded\s*\(',          "kept, renamed vertexFrames"),
    ("vertexGridEmbedded",        "GC-FRAME", r'\bvertexGridEmbedded\s*\(',            "kept, renamed vertexGrid"),
    ("vertexTangentBasis",        "GC-FRAME", r'\b(require)?VertexTangentBasis\b|\bvertexTangentBasis\b', "kept (the one 3D realisation of the gauge)"),
    ("halfedgeVectorsInVertex",   "GC-FRAME", r'\b(require)?HalfedgeVectorsInVertex\b|\bhalfedgeVectorsInVertex\b', "kept; read from the IntrinsicSurface"),
    ("transportVectorsAlongHalfedge","GC-FRAME", r'\b(require)?TransportVectorsAlongHalfedge\b|\btransportVectorsAlongHalfedge\b', "kept; read from the IntrinsicSurface"),
    ("vertexConnectionLaplacian", "GC-FRAME", r'\b(require)?VertexConnectionLaplacian\b|\bvertexConnectionLaplacian\b', "kept; read from the IntrinsicSurface"),
]

# the definitions of the port's own symbols, counted separately (they disappear with the files)
PORT_FILES = {"compute/include/nxr/delaunay.h", "compute/src/delaunay.cpp", "compute/include/nxr/chart.h", "compute/src/chart.cpp"}

STORED = [  # stored tangent quantities and gauge-bearing rows (string literals in pipelines/, io/, cli/, bindings/)
    (r'_canonical_frames\b',       "canonical frames Field [V,3,3] (e1 north, e2 west, n): 3D vectors, gauge-free; kept"),
    (r'_canonical_gauge_angle\b',  "north's angle in the PORT's signpost gauge: recomputed in geometry-central's gauge (gauge recorded)"),
    (r'_frame_singularities\b',    "Selection of singular vertices: gauge-free; kept"),
    (r'_anatomical_(angle|second|handed|reliable)\b|_anatomical_', "anatomical frame: angle FROM NORTH (gauge-free); becomes a reporting projection"),
    (r'flow_velocity(_anterior|_second)?\b|velocity_(anterior|second)\b', "optical-flow velocity: embedded 3D (gauge-free) + anatomical components"),
    (r'levi-civita',               "tiles' levi-civita form (ChartKit::transported in the port's gauge): recomputed, gauge recorded"),
    (r'"gauge"',                   "a row/provenance that already names a gauge"),
    (r'eigenmodes\b|_eigenvectors\b', "eigenmodes (scalar: gauge-free; complex connection modes: gauge recorded)"),
]

EXT = (".h", ".hpp", ".cpp", ".cc", ".c", ".mjs", ".mts", ".ts", ".js", ".py", ".m", ".yaml", ".yml", ".json")


def blank_comments(src, ext):
    if ext in (".py", ".yaml", ".yml"):
        return re.sub(r"(?m)#(?!\s*include).*$", "", src)
    if ext == ".m":
        return re.sub(r"(?m)%.*$", "", src)
    if ext == ".json":
        return src
    src = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), src, flags=re.S)
    return re.sub(r"//.*$", "", src, flags=re.M)


def kind(path):
    if path in PORT_FILES: return "port itself"
    if path.startswith("compute/test/golden/"): return "golden generator"
    if path.startswith("compute/test/parity/"): return "parity reference"
    if path.startswith("compute/test/"): return "compute test"
    if path.startswith("compute/include/"): return "compute header"
    if path.startswith("compute/src/"): return "compute source"
    if path.startswith("pipelines/test/"): return "pipeline test"
    if path.startswith("pipelines/definitions/"): return "pipeline definition"
    if path.startswith("pipelines/"): return "pipeline (nodes, steps)"
    if path.startswith("bindings/") and "/test" in path or path.startswith("bindings/scripts/"): return "binding test"
    if path.startswith("bindings/"): return "binding (" + path.split("/")[1] + ")"
    if path.startswith("cli/"): return "command line"
    if path.startswith("io/"): return "io"
    if path.startswith("meeg/"): return "meeg"
    return "other"


def tracked(repo):
    out = subprocess.run(["git", "-C", repo, "ls-files"], capture_output=True, text=True, check=True).stdout.split("\n")
    return [p for p in out if p and p.endswith(EXT) and not p.startswith("compute/deps/") and "/extern/" not in p and not p.startswith("docs/") and p != "tools/geometry_inventory.py"]


def scan(repo):
    hits = collections.defaultdict(lambda: collections.Counter())   # path -> Counter(symbol)
    for p in tracked(repo):
        try:
            src = open(os.path.join(repo, p), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        code = blank_comments(src, os.path.splitext(p)[1])
        for sid, fam, rx, _ in SYMBOLS:
            n = len(re.findall(rx, code))
            if n: hits[p][sid] += n
    return hits


def goldens(repo, hits):
    """The golden files (compute/test/golden, pipelines/test, bindings tests) read by the affected tests, and their generators."""
    gdir = os.path.join(repo, "compute/test/golden")
    names = sorted(os.listdir(gdir)) if os.path.isdir(gdir) else []
    data = [n for n in names if not n.startswith("gen_") and not n.startswith("check_")]
    out = []
    for p in sorted(hits):
        if kind(p) not in ("compute test", "pipeline test", "binding test"): continue
        src = open(os.path.join(repo, p), encoding="utf-8", errors="replace").read()
        for d in data:
            if d in src:
                stem = d.rsplit("_golden", 1)[0]
                gens = [n for n in names if n.startswith("gen_" + stem)]
                out.append((d, p, ", ".join(gens) or "-"))
    return out


def stored(repo):
    rows = collections.defaultdict(set)
    for p in tracked(repo):
        if not p.startswith(("pipelines/src", "pipelines/include", "io/src", "io/schema", "cli/", "bindings/cli")): continue
        src = open(os.path.join(repo, p), encoding="utf-8", errors="replace").read()
        lits = " ".join(re.findall(r'"(?:[^"\\\n]|\\.)*"', src))
        for rx, what in STORED:
            if re.search(rx, lits): rows[(rx, what)].add(p)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--section", default="all", choices=["summary", "symbols", "files", "goldens", "stored", "all"])
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    hits = scan(repo)
    fam = {sid: f for sid, f, _, _ in SYMBOLS}
    repl = {sid: r for sid, _, _, r in SYMBOLS}
    sec = a.section

    port_files = sorted(p for p, c in hits.items() if any(fam[s] == "PORT" for s in c))
    gc_files = sorted(p for p, c in hits.items() if any(fam[s] == "GC-FRAME" for s in c))
    if sec in ("summary", "all"):
        print("### Summary\n")
        by_kind = collections.Counter(kind(p) for p in port_files)
        uses = sum(n for p in port_files for s, n in hits[p].items() if fam[s] == "PORT")
        print(f"- files using the port or its gauge (PORT): **{len(port_files)}**, uses: **{uses}**")
        for k, n in sorted(by_kind.items(), key=lambda x: (-x[1], x[0])): print(f"  - {k}: {n}")
        print(f"- files using geometry-central's gauge today (GC-FRAME): **{len(gc_files)}**")
        both = sorted(set(port_files) & set(gc_files))
        print(f"- files using BOTH gauges: **{len(both)}**" + (": " + ", ".join(f"`{p}`" for p in both) if both else ""))
        print()
    if sec in ("symbols", "all"):
        print("### Symbols and their replacement\n")
        print("| symbol | family | uses | files | replaced by |\n|---|---|---|---|---|")
        for sid, f, _, r in SYMBOLS:
            files = [p for p in hits if hits[p][sid]]
            n = sum(hits[p][sid] for p in files)
            if n: print(f"| `{sid}` | {f} | {n} | {len(files)} | {r} |")
        print()
    if sec in ("files", "all"):
        print("### Files\n")
        print("| file | kind | family | symbols (uses) |\n|---|---|---|---|")
        for p in sorted(hits, key=lambda q: (kind(q), q)):
            fams = "+".join(sorted({fam[s] for s in hits[p]}))
            syms = ", ".join(f"{s} ({n})" for s, n in sorted(hits[p].items()))
            print(f"| `{p}` | {kind(p)} | {fams} | {syms} |")
        print()
    if sec in ("goldens", "all"):
        print("### Goldens read by the affected tests\n")
        print("| golden | read by | generator |\n|---|---|---|")
        for d, p, g in goldens(repo, hits): print(f"| `{d}` | `{p}` | {g} |")
        print()
    if sec in ("stored", "all"):
        print("### Stored names (string literals in pipelines/, io/, cli/)\n")
        print("| pattern | what it is, and what the switch does | files |\n|---|---|---|")
        for (rx, what), ps in sorted(stored(repo).items(), key=lambda x: x[0][1]):
            print(f"| `{rx}` | {what} | {', '.join(f'`{p}`' for p in sorted(ps))} |")
        print()


if __name__ == "__main__":
    sys.exit(main())
