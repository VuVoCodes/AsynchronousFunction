#!/bin/bash
# n=15 extension for the CREMA-D 3-frame isolation comparison (miLe / AC, discussion window).
# PRE-REGISTRATION: batch-3 seeds fixed in advance as 1111 2222 3333 4444 6666.
# Stopping rule: report all 15 seeds per arm regardless of outcome; no further seeds
# will be added after this batch. Mirrors E1 in scripts/rebuttal_p0.sh exactly.

set -u
cd /home/main/AsynchronousFunction
source /home/main/miniconda3/etc/profile.d/conda.sh
conda activate phd

SEEDDIR="outputs/rebuttal_seeds"
mkdir -p "$SEEDDIR"
LOG="$SEEDDIR/n15.log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

SEEDS=(1111 2222 3333 4444 6666)
for SEED in "${SEEDS[@]}"; do
  for ARM in a0 a075; do
    if [ "$ARM" = "a0" ]; then CA=0.0; else CA=0.75; fi
    EXPNAME="r15_${ARM}_seed${SEED}"
    OUTPATH="$SEEDDIR/$EXPNAME"
    if [ -f "$OUTPATH/train.log" ] && grep -q "Training complete" "$OUTPATH/train.log" 2>/dev/null; then
      log "n15 skip $EXPNAME (already complete)"
      continue
    fi
    log "n15 start $EXPNAME"
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
    log "n15 $EXPNAME done, acc=$ACC"
  done
done
log "n15 batch complete"
