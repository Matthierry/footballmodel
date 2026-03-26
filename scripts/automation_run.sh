#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw data/logs

SOURCE_FILE="data/raw/football_data.csv"
LAST_HASH_FILE="data/logs/last_source_hash.txt"
NEW_HASH_FILE="data/logs/new_source_hash.txt"

echo "Computing source hash"
if [[ -f "$SOURCE_FILE" ]]; then
  sha256sum "$SOURCE_FILE" | awk '{print $1}' > "$NEW_HASH_FILE"
else
  echo "missing" > "$NEW_HASH_FILE"
fi

if [[ -f "$LAST_HASH_FILE" ]] && cmp -s "$LAST_HASH_FILE" "$NEW_HASH_FILE"; then
  echo "No data change, skipping model rerun"
  exit 0
fi

echo "Data changed or first run, executing pipeline"
python scripts/run_pipeline.py
cp "$NEW_HASH_FILE" "$LAST_HASH_FILE"
