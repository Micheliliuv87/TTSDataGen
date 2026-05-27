#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_pipeline.sh \"your dialogue request\" [extra instructions]"
  exit 1
fi

QUERY="$1"
shift || true

EXTRA_INSTRUCTIONS="$*"

mkdir -p logs/pipeline
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_PATH="logs/pipeline/pipeline_${TIMESTAMP}.log"

python -m src.run_pipeline \
  --query "$QUERY" \
  --extra_instructions "$EXTRA_INSTRUCTIONS" \
  2>&1 | tee "$LOG_PATH"

echo ""
echo "Pipeline log:"
echo "  $LOG_PATH"