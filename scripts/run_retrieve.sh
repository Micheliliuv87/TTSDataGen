#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_retrieve.sh \"your query here\""
  exit 1
fi

QUERY="$*"

mkdir -p logs/rag
mkdir -p outputs/source_packs

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
OUTPUT_PATH="outputs/source_packs/source_pack_${TIMESTAMP}.json"
LATEST_PATH="outputs/source_packs/latest_source_pack.json"
LOG_PATH="logs/rag/retrieve_${TIMESTAMP}.log"

python -m src.retrieve \
  --config configs/rag.yaml \
  --query "$QUERY" \
  --output_path "$OUTPUT_PATH" \
  2>&1 | tee "$LOG_PATH"

cp "$OUTPUT_PATH" "$LATEST_PATH"

echo ""
echo "Wrote source pack:"
echo "  $OUTPUT_PATH"
echo "Latest source pack:"
echo "  $LATEST_PATH"
echo "Retrieve log:"
echo "  $LOG_PATH"