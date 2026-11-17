#!/bin/sh
# Compare the optical-flow methods on one recording window: runs `nxr-neuro opticalflow compute` once per method, each into its own named fields
# (<name>_<method>), and prints the seconds per frame, iterations and the exact fraction of each. Every method runs with the defaults (regulariser connection, datafit gauge: the Cortical Flow Studio's functional, alpha 0.01 unless --alpha is given; modal charges the same connection energy, band-limited to --modes).
#
#   cli/scripts/opticalflow-compare.sh NXR_NEURO_BIN DATASET/NAME SUBJECT RECORDING NAME -- <the usual opticalflow options: --family .. --gain .. --freq .. [--kernel K --start S --stop S --alpha A ...]>
#
# e.g.  opticalflow-compare.sh build/cli/nxr-neuro /data/ds/study sub-01 rest cmp -- --family morlet --gain 2 --freq 10 --start 10 --stop 20
# Extra method options by environment: SCALE_FAMILY (default itersine; perscale).
# Then compare the stored fields `<name>_<method>_{flow_modes,divergence,curl,speed}` (perscale: `<name>_perscale_m00_...`) in the viewer or with nxr.
set -e
bin=$1; ds=$2; sub=$3; rec=$4; name=$5; shift 5
[ "$1" = "--" ] && shift
for m in direct pcg perscale; do
  extra=""
  meth=$m
  [ "$m" = perscale ] && extra="--scale-family ${SCALE_FAMILY:-itersine}"
  echo "== $m"
  "$bin" run opticalflow --dataset "$ds" --subject "$sub" --recording "$rec" --name "${name}_$m" --method "$meth" $extra "$@" --replace \
    | python3 -c '
import sys, json
for line in sys.stdin:
    j = json.loads(line)
    if j.get("stage") == "flow": print("   frame %s/%s  %.4f s/frame%s" % (j["frame"], j["frames"], j["seconds_per_frame"], ("  %.1f it" % j["iterations"]) if "iterations" in j else ""))
    if j.get("stage") == "result":
        s = j["summary"]
        series = j.get("series") or s.get("series") or []                                 # perscale: one summary per member, no run-level exact fraction
        ef = s["exact_fraction"] if "exact_fraction" in s else (sum(m.get("exact_fraction", 0) for m in series) / len(series) if series else float("nan"))
        tag = "exact fraction %.3f" % ef if "exact_fraction" in s else "members %d, mean exact fraction %.3f" % (len(series), ef)
        print("   RESULT %s: %.4f s/frame, %s, fields %s" % (j["method"], j["solve_seconds_per_frame"], tag, ",".join(j["fields"][:2]) + ",.."))
' | tail -3
done
