#!/usr/bin/env bash
# run_train_baseline.sh
# Baseline: IDs-only (no side features) + JOINED catalog (no full catalog)

set -euo pipefail
# --------- Edit these paths if needed ----------
DATA_PATH="${DATA_PATH:-data}" # folder with Ratings.csv / Users.csv / Books.csv
SAVE_DIR="${SAVE_DIR:-out}"


mkdir -p "${SAVE_DIR}"
CUDA_VISIBLE_DEVICES="3" 
NUM_GPUS=1

torchrun --standalone --nproc_per_node=1 -m retrieval.two_tower_train \
  --data-path "${DATA_PATH}" --id-maps-json id_maps.json \
  --no-side-features --catalog-mode joined \
  --batch-size 8192 --num-iterations 10000 \
  --num-workers 0 --save-dir "${SAVE_DIR}"

echo "[DONE] Artifacts saved to: ${SAVE_DIR}"

