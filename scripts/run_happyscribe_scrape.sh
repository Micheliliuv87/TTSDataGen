#!/usr/bin/env bash
set -euo pipefail

python scrapers/happyscribe/scrape.py \
  --all-podcasts \
  --output-dir data/interim/podcasts/happyscribe \
  --raw-html-dir data/raw/podcasts/happyscribe \
  --save-raw-html \
  --sleep 1.0 \
  --validate