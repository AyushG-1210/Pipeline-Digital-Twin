#!/usr/bin/env bash
# run.sh -- the whole A100 session as one pool. Run from the repo root (the PINN folder):
#     cd PINN && bash run.sh            # 24 at once (default)
#     JOBS=30 bash run.sh                                          # change the pool size
#     NO_BONUS=1 bash run.sh                                       # skip the 2D strong sweep
# Queue, in priority order (finished runs are skipped, so this is safe to restart):
#   2D weak-sweep runs still missing -> 2D annealing (both forms, warm+cold) -> 1D annealing -> 2D strong sweep.
# Runs without a result are retried twice, at half the pool size each time (JOBS/2, then JOBS/4).
set -u
JOBS=${JOBS:-24}
OUT=heat_runs/heat_matched
STARTS=heat_runs/heat_results/ckpt
mkdir -p heat_runs/a100_logs
stamp() { echo "[$(date +%H:%M:%S)] $*"; }

stamp "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
stamp "CPU cores: $(nproc), RAM: $(free -g | awk '/Mem/{print $2}') GB, pool: $JOBS"

stamp "smoke check (30 iterations, about a minute)"
python heat_drift.py --dim 1 --case clash --init warm --form strong --anneal --seed 42 \
    --out heat_runs/a100_smoke --starts $STARTS --smoke > heat_runs/a100_logs/smoke.log 2>&1 \
    && stamp "  passed" || { stamp "  FAILED, see heat_runs/a100_logs/smoke.log"; exit 1; }

BONUS=""; [ -n "${NO_BONUS:-}" ] && BONUS="--no-bonus"
stamp "batch started; progress in heat_runs/a100_logs/batch.log"
python heat_drift.py --batch $BONUS --jobs $JOBS --retries 2 --out $OUT --starts $STARTS 2>&1 | tee heat_runs/a100_logs/batch.log
stamp "all done; the table at the end of heat_runs/a100_logs/batch.log is the summary"