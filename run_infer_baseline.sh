#!/usr/bin/env bash
# run_infer_baseline.sh

set -euo pipefail
DATA_PATH="${DATA_PATH:-data}" # folder with Ratings.csv / Users.csv / Books.csv
SAVE_DIR="${SAVE_DIR:-out}"


python3 -m retrieval.two_tower_retrieval \
  --save-dir ${SAVE_DIR} \
  --data-path ${DATA_PATH} \
  --id-maps-json id_maps.json \
  --embedding-dim 64 \
  --layer-sizes 128,64 \
  --metric ip \
  --topk 20 \
  --books-csv ${DATA_PATH}/Books.csv \
  --user 276725 --user 277427
