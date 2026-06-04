#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_pipeline.sh \"your dialogue request\" [extra instructions]"
  echo ""
  echo "Optional env vars:"
  echo "  PIPELINE_MODE=full|draft"
  echo "  PIPELINE_SKIP_POLISH=1"
  echo "  PIPELINE_SKIP_EXPAND=1"
  exit 1
fi

QUERY="$1"
shift || true

EXTRA_INSTRUCTIONS="$*"

MODE="${PIPELINE_MODE:-full}"

mkdir -p logs/pipeline
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_PATH="logs/pipeline/pipeline_${TIMESTAMP}.log"

CMD=(
  python -m run_pipeline
  --query "$QUERY"
  --extra_instructions "$EXTRA_INSTRUCTIONS"
  --mode "$MODE"
  --save_prompt
)

if [ "${PIPELINE_SKIP_POLISH:-0}" = "1" ]; then
  CMD+=(--skip_polish)
fi

if [ "${PIPELINE_SKIP_EXPAND:-0}" = "1" ]; then
  CMD+=(--skip_expand)
fi

"${CMD[@]}" 2>&1 | tee "$LOG_PATH"

echo ""
echo "Pipeline log:"
echo "  $LOG_PATH"