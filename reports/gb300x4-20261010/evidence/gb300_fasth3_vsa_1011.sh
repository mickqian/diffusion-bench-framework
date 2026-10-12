#!/usr/bin/env bash
# 4xGB300: FastH3 V2 sglang cell on main 6576e50 with the VSA-H3 profile (dbf bb560dc+), two runner rounds.
set -u
W=/scratch/gb; mkdir -p $W/logs; cd $W
M=6576e50cd5e5817b10e12245104603eb1b0d8a4f
set -a; . ~/.config/radixark/secrets.env; set +a
[ -n "${HF_TOKEN:-}" ] && echo "GB hf token loaded" || echo "GB no hf token"
cd /sgl-workspace/sglang && { git cat-file -t $M >/dev/null 2>&1 || timeout 900 git fetch -q origin $M; } || { echo "GB FATAL fetch"; exit 1; }
[ -d $W/sgl-main ] || git worktree add -q --detach $W/sgl-main $M
cd $W
[ -d dbf ] || git clone -q https://github.com/mickqian/diffusion-bench-framework dbf
git -C dbf fetch -q origin round-20261010 && git -C dbf checkout -q --detach origin/round-20261010
echo "GB dbf $(git -C dbf log --oneline -1 | cut -c1-60)"
PYTHONPATH=$W/sgl-main/python python3 -c "import sglang, sys; print('GB sglang', sglang.__file__); sys.exit(0 if sglang.__file__.startswith('$W/sgl-main/') else 1)" || { echo "GB FATAL sglang tree"; exit 1; }
grep -o "component-attention-backends transformer=video_sparse_attn_h3" dbf/configs/comparison_configs.json | head -1 | sed 's/^/GB profile has: /'
export DBF_STATE_DIR=$W DBF_REPO_DIR=$W/dbf DBF_LOG_DIR=$W/logs DBF_HF_HOME=$W/hf DBF_HARDWARE_PROFILE=gb300 \
       DBF_FRAMEWORKS=sglang DBF_CASE_TIMEOUT=5400 DBF_SGLANG_PYTHONPATH=$W/sgl-main/python
for r in 1 2; do
  bash dbf/scripts/devbox_run_cases.sh gb300fh3vsa1011r$r 0,1,2,3 $((61600 + r * 10)) single_e2e minimax_h3_fasth3_v2_t2va_5s
  f=$W/logs/run_gb300fh3vsa1011r${r}_minimax_h3_fasth3_v2_t2va_5s.json
  python3 - "$f" "$r" <<'PY'
import json, sys
try:
    x = next(r for r in json.load(open(sys.argv[1]))["results"] if r["framework"] == "sglang")
    m = x["metrics"]; st = m.get("diagnostic_stages_s") or {}
    print(f"GB r{sys.argv[2]} sglang median={x['latency_s']} samples={m.get('latency_samples_s')} denoise={st.get('MiniMaxH3DenoisingStage')} decode={st.get('MiniMaxH3DecodingStage')} err={str(x.get('error'))[:160]}")
except Exception as e:
    print(f"GB r{sys.argv[2]} no result ({type(e).__name__})")
PY
  echo "GB r$r backend: $(grep -ahE 'Using [a-z_0-9]+ attention backend' $W/logs/run_gb300fh3vsa1011r${r}_minimax_h3_fasth3_v2_t2va_5s.runlog | sed 's/.*\] //' | sort | uniq -c | tr '\n' ';')"
done
echo "=== GB_DONE $(date -Is)"
