#!/usr/bin/env bash
set -euo pipefail

python src/build_rag.py \
  --chunks data/processed/rag/podcasts/happyscribe/chunks.jsonl \
  --persist-dir vector_db/chroma_content
