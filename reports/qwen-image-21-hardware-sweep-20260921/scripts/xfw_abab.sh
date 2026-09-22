#!/usr/bin/env bash
# Cross-framework ABAB on ONE physical node.
#
# The first pass measured sglang at 02:03 and LightX2V at 03:32 -- ninety
# minutes apart. That is A-then-B, and the compute gap it produced (2.6% on
# H200, 1.9% on the 5090) sits under this repo's 5% bar, where A-then-B is not
# evidence. So alternate them, and before each measurement confirm the OTHER
# devbox on the same node is not touching its GPU: the two boxes share a host,
# and a neighbour under load contaminates latency through the host and PCIe.
#
#   xfw_abab.sh <sglang-box> <lightx2v-box> <hw> <rounds> <reps>
set -uo pipefail
REPO=/Users/mick/repos/diffusion-bench-framework
SG_BOX="${1:?sglang box}"; LX_BOX="${2:?lightx2v box}"; HW="${3:?hw}"
ROUNDS="${4:-3}"; REPS="${5:-5}"

idle() {  # $1 = box; true when nothing is resident on its GPU
    local used
    used=$(bash "$REPO/scripts/rxrun.sh" "$1" \
        'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits' 2>/dev/null | tr -dc '0-9')
    [ -z "$used" ] && return 1
    [ "$used" -lt 2000 ]
}

echo "=== XFW_ABAB_START $(date -Is) hw=$HW rounds=$ROUNDS reps=$REPS ==="
for r in $(seq 1 "$ROUNDS"); do
    for arm in sglang lightx2v; do
        if [ "$arm" = sglang ]; then box="$SG_BOX"; other="$LX_BOX"
        else box="$LX_BOX"; other="$SG_BOX"; fi
        if ! idle "$other"; then
            echo "  round=$r arm=$arm SKIPPED: neighbour $other is busy"
            continue
        fi
        if [ "$arm" = sglang ]; then
            out=$(bash "$REPO/scripts/rxrun.sh" "$box" \
                "cd /scratch/state && bash qi21_bench.sh $HW 40 $REPS 2>&1 | grep -aE 'RESULT|REQUEST FAILED|never became'" 2>/dev/null)
        else
            out=$(bash "$REPO/scripts/rxrun.sh" "$box" \
                "cd /scratch/state && bash qi21_lx2v_image.sh $HW $REPS 2>&1 | grep -aE 'RESULT|REQUEST FAILED|never became'" 2>/dev/null)
        fi
        echo "  round=$r arm=$arm :: $(echo "$out" | tr '\n' ' ' | sed 's/  */ /g')"
    done
done
echo "=== XFW_ABAB_DONE $(date -Is) ==="
