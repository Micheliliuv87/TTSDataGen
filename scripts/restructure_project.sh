#!/usr/bin/env bash
set -euo pipefail

# Run from project root:
# bash scripts/restructure_project.sh

PROJECT_ROOT="$(pwd)"

PODCAST_SLUG="stuff-you-missed-in-history-class"
SITE="happyscribe"

echo "Project root: ${PROJECT_ROOT}"
echo "Setting up Content2DialogueV0 project tree..."

# -----------------------------
# 1. Core directories
# -----------------------------

mkdir -p scripts

mkdir -p data/raw/podcasts/${SITE}/${PODCAST_SLUG}/list_pages
mkdir -p data/raw/podcasts/${SITE}/${PODCAST_SLUG}/episode_pages

mkdir -p data/interim/podcasts/${SITE}/${PODCAST_SLUG}
mkdir -p data/interim/podcasts/${SITE}/${PODCAST_SLUG}/logs

mkdir -p data/processed/rag/podcasts/${SITE}/${PODCAST_SLUG}
mkdir -p data/processed/cleaned/podcasts/${SITE}/${PODCAST_SLUG}

mkdir -p vector_db/chroma_content

mkdir -p scrapers/${SITE}
mkdir -p scrapers/legacy
mkdir -p src

mkdir -p outputs/source_packs
mkdir -p outputs/dialogues
mkdir -p outputs/evaluations

mkdir -p knowledge_base

# -----------------------------
# 2. Make Python packages
# -----------------------------

touch scrapers/__init__.py
touch scrapers/${SITE}/__init__.py
touch src/__init__.py

# -----------------------------
# 3. Move current scraper files safely
# -----------------------------

move_if_exists() {
  local src="$1"
  local dst="$2"

  if [ -e "$src" ]; then
    if [ -e "$dst" ]; then
      echo "Skip move: ${dst} already exists. Original kept at ${src}"
    else
      mkdir -p "$(dirname "$dst")"
      mv "$src" "$dst"
      echo "Moved: ${src} -> ${dst}"
    fi
  fi
}

# Prefer lowercase module path
move_if_exists "scrapers/HappyScribe/podcasts.py" "scrapers/${SITE}/scrape.py"

# Keep old generic file as legacy if it exists
move_if_exists "scrapers/podcast_site_a.py" "scrapers/legacy/podcast_site_a.py"

# Remove empty old directory if possible
if [ -d "scrapers/HappyScribe" ]; then
  rmdir "scrapers/HappyScribe" 2>/dev/null || true
fi

# -----------------------------
# 4. Move current JSONL outputs safely
# -----------------------------

# sources.jsonl is not raw; it is parsed source documents
move_if_exists "data/sources.jsonl" \
  "data/interim/podcasts/${SITE}/${PODCAST_SLUG}/source_documents.jsonl"

# test_sources.jsonl is also parsed output, not raw HTML
move_if_exists "data/raw/test_sources.jsonl" \
  "data/interim/podcasts/${SITE}/${PODCAST_SLUG}/test_source_documents.jsonl"

# -----------------------------
# 5. Add placeholder clean.py if missing
# -----------------------------

if [ ! -f "scrapers/${SITE}/clean.py" ]; then
cat > "scrapers/${SITE}/clean.py" <<'PY'
#!/usr/bin/env python3
"""
Cleaning utilities for HappyScribe podcast transcripts.

This file is intentionally minimal for now.
Later it can handle:
- ad removal
- transcript boilerplate removal
- timestamp normalization
- very short / duplicate transcript filtering
"""

from __future__ import annotations


def clean_transcript_text(text: str) -> str:
    """V0 placeholder: return text unchanged."""
    return text
PY
  echo "Created: scrapers/${SITE}/clean.py"
fi

# -----------------------------
# 6. Add run script for HappyScribe scrape
# -----------------------------

if [ ! -f "scripts/run_happyscribe_scrape.sh" ]; then
cat > "scripts/run_happyscribe_scrape.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

PODCAST_URL="https://podcasts.happyscribe.com/stuff-you-missed-in-history-class?page=3"
OUTPUT="data/interim/podcasts/happyscribe/stuff-you-missed-in-history-class/source_documents.jsonl"

python scrapers/happyscribe/scrape.py \
  --podcast-url "$PODCAST_URL" \
  --output "$OUTPUT" \
  --sleep 1.0 \
  --validate
SH
  chmod +x "scripts/run_happyscribe_scrape.sh"
  echo "Created: scripts/run_happyscribe_scrape.sh"
fi

# -----------------------------
# 7. Add placeholder chunk command script
# -----------------------------

if [ ! -f "scripts/run_chunk_sources.sh" ]; then
cat > "scripts/run_chunk_sources.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

INPUT="data/interim/podcasts/happyscribe/stuff-you-missed-in-history-class/source_documents.jsonl"
OUTPUT="data/processed/rag/podcasts/happyscribe/stuff-you-missed-in-history-class/chunks.jsonl"

python src/chunk_sources.py \
  --input "$INPUT" \
  --output "$OUTPUT"
SH
  chmod +x "scripts/run_chunk_sources.sh"
  echo "Created: scripts/run_chunk_sources.sh"
fi

echo ""
echo "Done."
echo ""
echo "Recommended next commands:"
echo "  tree -I '__pycache__'"
echo "  bash scripts/run_happyscribe_scrape.sh"