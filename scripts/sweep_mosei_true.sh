#!/bin/bash
# TRUE CMU-MOSEI sweep (rebuttal): re-run Table-1 methods on genuine MOSEI data.
#
# Context: the original "MOSEI" runs (outputs/sweep_mosei/) used CH-SIMS data by
# mistake (see Reviews/rebuttal_plan.md). This sweep uses configs/mosei_true.yaml
# with root data/MOSEI_true/ containing the verified mosei_senti_data.pkl
# (MulT/InfoReg-processed: GloVe 300-d text, COVAREP 74-d audio, FACET 35-d vision,
# 16,326 train / 4,659 test, continuous labels bucketed to 3-class at |0.5|).
#
# Core conditions first (Table-1 PGGB rows), then remaining baselines.
# Output: outputs/sweep_mosei_true/

set -u
cd /home/main/AsynchronousFunction
source /home/main/miniconda3/etc/profile.d/conda.sh
conda activate phd

CONFIG="configs/mosei_true.yaml"
OUTPUT_BASE="outputs/sweep_mosei_true"
mkdir -p "$OUTPUT_BASE"
LOGFILE="$OUTPUT_BASE/sweep.log"
SEEDS=(42 123 456 789 1024)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOGFILE"; }

run_experiment() {
    local run_id="$1"; local mode="$2"; local seed="$3"; shift 3
    local full_id="${run_id}_seed${seed}"
    local exp_dir="$OUTPUT_BASE/$full_id"
    if [ -f "$exp_dir/train.log" ] && grep -q "Training complete" "$exp_dir/train.log" 2>/dev/null; then
        log "skip $full_id (complete)"
        return
    fi
    log "start $full_id"
    python scripts/train.py \
        --config "$CONFIG" \
        --mode "$mode" \
        --seed "$seed" \
        --exp-name "$full_id" \
        --output-dir "$OUTPUT_BASE" \
        "$@" \
        > "$OUTPUT_BASE/${full_id}.stdout" 2>&1
    local acc
    acc=$(grep "Training complete" "$exp_dir/train.log" 2>/dev/null | tail -1 | grep -oP 'Best accuracy: \K[\d.]+')
    log "done $full_id acc=$acc"
}

log "===== TRUE-MOSEI sweep start ====="

# ---- Tier 1: core Table-1 conditions (baseline + PGGB rows + OGM-GE) ----
for seed in "${SEEDS[@]}"; do
    run_experiment "tmosei_baseline"       "baseline" "$seed" --epochs 100
    run_experiment "tmosei_ogm_ge"         "adaptive" "$seed" --ogm-ge --alpha 0.8 --asgml-mode continuous --continuous-alpha 0 --epochs 100
    run_experiment "tmosei_boost_only"     "adaptive" "$seed" --asgml-mode continuous --continuous-alpha 0.5 --epochs 100
    run_experiment "tmosei_boost_ogm_a075" "adaptive" "$seed" --ogm-ge --alpha 0.8 --asgml-mode continuous --continuous-alpha 0.75 --epochs 100
done

# ---- Tier 2: remaining Table-1 baselines ----
for seed in "${SEEDS[@]}"; do
    run_experiment "tmosei_mmpareto" "mmpareto" "$seed" --epochs 100
    run_experiment "tmosei_agm"      "agm"      "$seed" --epochs 100
    run_experiment "tmosei_gblend"   "gblend"   "$seed" --epochs 100
    run_experiment "tmosei_cggm"     "cggm"     "$seed" --epochs 100 --cggm-rou 1.3 --cggm-lamda 0.2
done

log "===== TRUE-MOSEI sweep complete ====="

python - <<'EOF' | tee -a outputs/sweep_mosei_true/sweep.log
import re, glob, statistics
from collections import defaultdict
groups = defaultdict(list)
for f in sorted(glob.glob('outputs/sweep_mosei_true/tmosei_*_seed*/train.log')):
    m = re.search(r'Training complete.*Best accuracy:\s*([\d.]+)', open(f).read())
    if m:
        name = re.sub(r'_seed\d+$', '', f.split('/')[-2])
        groups[name].append(float(m.group(1)) * 100)
print(f"{'method':28s} {'n':>2s} {'mean':>7s} {'std':>6s}")
for name, accs in sorted(groups.items()):
    std = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print(f'{name:28s} {len(accs):2d} {statistics.mean(accs):7.2f} {std:6.2f}')
EOF
