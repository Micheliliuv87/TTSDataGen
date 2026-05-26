#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs/audit
mkdir -p outputs/evaluations/data_audit/happyscribe/interim_audit

python src/audit_happyscribe_interim.py \
  --input-dir data/interim/podcasts/happyscribe \
  --report-dir outputs/evaluations/data_audit/happyscribe/interim_audit \
  2>&1 | tee logs/audit/happyscribe_interim_audit.log