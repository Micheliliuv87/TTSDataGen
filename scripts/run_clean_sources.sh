#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/clean
mkdir -p data/processed/cleaned/podcasts/happyscribe
mkdir -p outputs/evaluations/data_audit/happyscribe/cleaning

PYTHONPATH=. python src/clean_sources.py \
  --input-dir data/interim/podcasts/happyscribe \
  --output-dir data/processed/cleaned/podcasts/happyscribe \
  --report-dir outputs/evaluations/data_audit/happyscribe/cleaning \
  --min-chars 1000 \
  2>&1 | tee logs/clean/happyscribe_clean.log
