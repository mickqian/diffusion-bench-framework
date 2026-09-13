#!/usr/bin/env bash
# Merge a finished run's per-case JSONs, gate it, and publish.
#
#   scripts/merge_and_publish_run.sh <results-glob> <run-id> <label> <gpu> <reproduce> [--live] [--note K=V ...]
#
#   scripts/merge_and_publish_run.sh \
#       'tmp/pull/run_b200final0913_*.json' b200x4-final-20260913 \
#       '4xB200 cross-framework (latest-vs-latest)' '4x NVIDIA B200 183GB' \
#       scripts/run_b200_final_20260913.sh --live --note harness='...'
#
# Dry run unless --live is passed.
#
# The three gates are the ones that have actually let bad data through before,
# so they are hard failures rather than warnings:
#
#   command drift      a config edited after its cell ran -- the published row
#                      then describes a command nobody executed.
#   partial sources    a mid-run checkpoint looks exactly like a finished run
#                      apart from one flag; merging it silently publishes a
#                      matrix whose missing frameworks look like skips.
#   native fallbacks   sglang loads a component through a Diffusers fallback
#                      when it has no customized implementation, and only
#                      REFUSES to when tp/sp/ulysses/ring/kv_gather > 1. Under
#                      cfg-parallel alone it proceeds behind one log line, and
#                      the row still says "sglang".
set -uo pipefail

GLOB="${1:?usage: merge_and_publish_run.sh <results-glob> <run-id> <label> <gpu> <reproduce> [--live] [--note K=V ...]}"
RUN_ID="${2:?run id}"
LABEL="${3:?label}"
GPU="${4:?gpu description}"
REPRODUCE="${5:?repo-relative path of the script that produced this run}"
shift 5

LIVE=0
# Expanded below as ${NOTES[@]+"${NOTES[@]}"}: under `set -u` an empty array is an
# unbound variable in bash < 4.4, and macOS still ships 3.2.
NOTES=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --live) LIVE=1; shift ;;
    --note) NOTES+=(--note "$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$(git rev-parse --show-toplevel)"
PY_BIN="${DBF_PYTHON:-.venv/bin/python3}"
[ -x "${PY_BIN}" ] || PY_BIN="$(command -v python3)"
OUT="${DBF_REPORT_DIR:-tmp/report}"
mkdir -p "${OUT}"

echo "=== merge (strict command-drift) ==="
# shellcheck disable=SC2086
DIFFUSION_BENCH_STRICT_COMMANDS=1 PYTHONPATH=src "${PY_BIN}" \
  -m diffusion_bench.build_report_artifacts \
  --results ${GLOB} \
  --config configs/comparison_configs.json \
  --output-json "${OUT}/merged.json" \
  --dashboard-md "${OUT}/dashboard.md" \
  --issue-md "${OUT}/issue.md" \
  --run-id "${RUN_ID}" || exit 1

echo "=== gates ==="
"${PY_BIN}" - "${OUT}/merged.json" <<'PY' || { echo "REFUSING to publish"; exit 1; }
import json, sys
m = json.load(open(sys.argv[1]))
bad = 0
if m.get("command_drift_warnings"):
    print("  FAIL command drift:", m["command_drift_warnings"][:3]); bad = 1
else:
    print("  ok   no command drift")
if m.get("partial_sources"):
    print("  FAIL mid-run checkpoints merged:", m["partial_sources"]); bad = 1
else:
    print("  ok   no partial sources")
fb = [(r.get("case_id"), r.get("framework")) for r in m.get("results", [])
      if (r.get("metrics") or {}).get("native_fallback_components")]
if fb:
    print("  FAIL cells that loaded a native fallback:", fb); bad = 1
else:
    print("  ok   no native fallbacks")
errs = [(r.get("case_id"), r.get("framework"), str(r.get("error"))[:60])
        for r in m.get("results", []) if r.get("error")]
print(f"  note {len(errs)} errored cell(s)")
for e in errs:
    print("        ", e)
disp = [(r.get("case_id"), r.get("framework"), (r.get("metrics") or {}).get("latency_spread_pct"))
        for r in m.get("results", []) if (r.get("metrics") or {}).get("latency_dispersed")]
print(f"  note {len(disp)} dispersed cell(s) -- published with their samples listed")
for d in disp:
    print("        ", d)
cases = {r.get("case_id") for r in m.get("results", [])}
cells = sum(1 for r in m.get("results", []) if r.get("latency_s"))
print(f"  ok   {cells} measured cells across {len(cases)} cases")
sys.exit(bad)
PY

echo "=== publish$([ "${LIVE}" = 1 ] || echo ' (dry run)') ==="
PYTHONPATH=src "${PY_BIN}" scripts/publish_bench_run.py \
  --merged "${OUT}/merged.json" \
  --run-id "${RUN_ID}" \
  --label "${LABEL}" \
  --gpu "${GPU}" \
  --reproduce "${REPRODUCE}" \
  ${NOTES[@]+"${NOTES[@]}"} \
  $([ "${LIVE}" = 1 ] || echo --dry-run)
