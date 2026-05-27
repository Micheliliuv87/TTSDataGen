#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p logs/rag
mkdir -p vector_db/chroma_content

python -m src.build_rag \
  --config configs/rag.yaml \
  2>&1 | tee logs/rag/build_rag.log