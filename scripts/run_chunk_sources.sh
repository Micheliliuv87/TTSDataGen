#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p data/processed/rag/podcasts/happyscribe
mkdir -p outputs/evaluations/data_audit/happyscribe
mkdir -p logs/chunk

python -m src.chunk_sources \
  --input_dir data/processed/cleaned/podcasts/happyscribe \
  --output_path data/processed/rag/podcasts/happyscribe/chunks.jsonl \
  --summary_path outputs/evaluations/data_audit/happyscribe/chunk_summary.json \
  --target_chars 2800 \
  --max_chars 3600 \
  --min_chunk_chars 400 \
  --overlap_blocks 1 \
  2>&1 | tee logs/chunk/happyscribe_chunk.log