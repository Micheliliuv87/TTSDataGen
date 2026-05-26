#!/usr/bin/env bash
set -euo pipefail

python src/clean_sources.py \
  --input-dir data/interim/podcasts/happyscribe \
  --output-dir data/processed/cleaned/podcasts/happyscribe
