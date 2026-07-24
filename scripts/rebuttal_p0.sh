#!/bin/bash
# P0 rebuttal experiments (NeurIPS 2026 Submission 12365).
#   E2: wall-clock + peak-VRAM timing table  (6 short runs, clean GPU first)
#   E3: per-modality update-norm instrumentation, Adam (MOSEI) vs SGD (CREMA-D)
#   E1: 5 fresh seeds x {alpha=0, alpha=0.75} on CREMA-D 3-frame  (10 full runs)
# Sequential on one GPU. Outputs under outputs/rebuttal_p0/ and outputs/rebuttal_seeds/.

set -u
cd /home/main/AsynchronousFunction
source /home/main/miniconda3/etc/profile.d/conda.sh
conda activate phd

OUT="outputs/rebuttal_p0"
SEEDDIR="outputs/rebuttal_seeds"
mkdir -p "$OUT/timing" "$OUT/norms" "$SEEDDIR"
LOG="$OUT/p0.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

vram_poll() {
  while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$1" 2>/dev/null
    sleep 1
  done
}

run_timed() {
  local name="$1"; shift
  local vfile="$OUT/timing/${name}.vram"
  : > "$vfile"
  vram_poll "$vfile" &
  local VP=$!
  "$@" > "$OUT/timing/${name}.stdout" 2>&1
  local RC=$?
  kill "$VP" 2>/dev/null; wait "$VP" 2>/dev/null
  log "E2 $name done (rc=$RC)"
}

log "===== P0 start ====="

# ---------------- E2: timing (8 epochs each) ----------------
log "E2: timing runs"

run_timed cremad_baseline python scripts/train.py --config configs/cremad.yaml \
  --mode baseline --num-frames 3 --fps 3 --seed 42 --epochs 8 \
  --exp-name timing_cremad_baseline --output-dir "$OUT/timing"

run_timed cremad_boost python scripts/train.py --config configs/cremad.yaml \
  --mode adaptive --asgml-mode continuous --continuous-alpha 0.5 \
  --num-frames 3 --fps 3 --seed 42 --epochs 8 \
  --exp-name timing_cremad_boost --output-dir "$OUT/timing"

run_timed cremad_boost_ogm python scripts/train.py --config configs/cremad.yaml \
  --mode adaptive --asgml-mode continuous --continuous-alpha 0.75 \
  --ogm-ge --alpha 0.8 --num-frames 3 --fps 3 --seed 42 --epochs 8 \
  --exp-name timing_cremad_boost_ogm --output-dir "$OUT/timing"

run_timed mosei_baseline python scripts/train.py --config configs/mosei.yaml \
  --mode baseline --seed 42 --epochs 8 \
  --exp-name timing_mosei_baseline --output-dir "$OUT/timing"

run_timed mosei_boost python scripts/train.py --config configs/mosei.yaml \
  --mode adaptive --asgml-mode continuous --continuous-alpha 0.5 --seed 42 --epochs 8 \
  --exp-name timing_mosei_boost --output-dir "$OUT/timing"

run_timed mosei_boost_ogm python scripts/train.py --config configs/mosei.yaml \
  --mode adaptive --asgml-mode continuous --continuous-alpha 0.75 \
  --ogm-ge --alpha 0.8 --seed 42 --epochs 8 \
  --exp-name timing_mosei_boost_ogm --output-dir "$OUT/timing"

# ---------------- E3: update-norm instrumentation (3 epochs each) ----------------
log "E3: update-norm instrumented runs"

PGGB_UPDATE_NORMS_FILE="$PWD/$OUT/norms/mosei_a075.jsonl" python scripts/train.py \
  --config configs/mosei.yaml --mode adaptive --asgml-mode continuous \
  --continuous-alpha 0.75 --ogm-ge --alpha 0.8 --seed 42 --epochs 3 \
  --exp-name norms_mosei_a075 --output-dir "$OUT/norms" \
  > "$OUT/norms/mosei_a075.stdout" 2>&1
log "E3 mosei_a075 done (rc=$?)"

PGGB_UPDATE_NORMS_FILE="$PWD/$OUT/norms/mosei_a0.jsonl" python scripts/train.py \
  --config configs/mosei.yaml --mode adaptive --asgml-mode continuous \
  --continuous-alpha 0.0 --ogm-ge --alpha 0.8 --seed 42 --epochs 3 \
  --exp-name norms_mosei_a0 --output-dir "$OUT/norms" \
  > "$OUT/norms/mosei_a0.stdout" 2>&1
log "E3 mosei_a0 done (rc=$?)"

PGGB_UPDATE_NORMS_FILE="$PWD/$OUT/norms/cremad_a075.jsonl" python scripts/train.py \
  --config configs/cremad.yaml --mode adaptive --asgml-mode continuous \
  --continuous-alpha 0.75 --ogm-ge --alpha 0.8 --num-frames 3 --fps 3 --seed 42 --epochs 3 \
  --exp-name norms_cremad_a075 --output-dir "$OUT/norms" \
  > "$OUT/norms/cremad_a075.stdout" 2>&1
log "E3 cremad_a075 done (rc=$?)"

PGGB_UPDATE_NORMS_FILE="$PWD/$OUT/norms/cremad_a0.jsonl" python scripts/train.py \
  --config configs/cremad.yaml --mode adaptive --asgml-mode continuous \
  --continuous-alpha 0.0 --ogm-ge --alpha 0.8 --num-frames 3 --fps 3 --seed 42 --epochs 3 \
  --exp-name norms_cremad_a0 --output-dir "$OUT/norms" \
  > "$OUT/norms/cremad_a0.stdout" 2>&1
log "E3 cremad_a0 done (rc=$?)"

# ---------------- E1: 5 fresh seeds x 2 arms, full 100 epochs ----------------
log "E1: new-seed runs (10 x ~31 min)"

SEEDS=(2027 3407 5555 7777 9999)
for SEED in "${SEEDS[@]}"; do
  for ARM in a0 a075; do
    if [ "$ARM" = "a0" ]; then CA=0.0; else CA=0.75; fi
    EXPNAME="r10_${ARM}_seed${SEED}"
    OUTPATH="$SEEDDIR/$EXPNAME"
    if [ -f "$OUTPATH/train.log" ] && grep -q "Training complete" "$OUTPATH/train.log" 2>/dev/null; then
      log "E1 skip $EXPNAME (already complete)"
      continue
    fi
    log "E1 start $EXPNAME"
    python scripts/train.py \
      --config configs/cremad.yaml \
      --mode adaptive --asgml-mode continuous \
      --ogm-ge --alpha 0.8 \
      --continuous-alpha "$CA" \
      --num-frames 3 --fps 3 \
      --seed "$SEED" \
      --exp-name "$EXPNAME" \
      --output-dir "$SEEDDIR" \
      > "$SEEDDIR/${EXPNAME}.stdout" 2>&1
    ACC=$(grep "Training complete" "$OUTPATH/train.log" 2>/dev/null | tail -1 | grep -oP 'Best accuracy: \K[\d.]+')
    log "E1 $EXPNAME done, acc=$ACC"
  done
done

# ---------------- Report ----------------
log "Generating report"
python scripts/rebuttal_p0_report.py | tee -a "$LOG"
log "===== P0 complete ====="
