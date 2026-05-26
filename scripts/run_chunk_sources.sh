#!/usr/bin/env bash
set -euo pipefail

python src/chunk_sources.py \
  --input-dir data/processed/cleaned/podcasts/happyscribe \
  --output data/processed/rag/podcasts/happyscribe/chunks.jsonl
