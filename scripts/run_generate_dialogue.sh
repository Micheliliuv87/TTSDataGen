#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs/generation
mkdir -p outputs/dialogues

TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_PATH="logs/generation/generate_dialogue_${TIMESTAMP}.log"

EXTRA_INSTRUCTIONS="$*"

python -m src.generate_dialogue \
  --config configs/generation.yaml \
  --source_pack outputs/source_packs/latest_source_pack.json \
  --save_prompt \
  --extra_instructions "$EXTRA_INSTRUCTIONS" \
  2>&1 | tee "$LOG_PATH"

echo ""
echo "Generation log:"
echo "  $LOG_PATH"