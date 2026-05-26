#!/usr/bin/env python3
"""
Clean HappyScribe interim source documents.

Input:
  data/interim/podcasts/happyscribe/<podcast_slug>/source_documents.jsonl

Output:
  data/processed/cleaned/podcasts/happyscribe/<podcast_slug>/source_documents_clean.jsonl

Reports:
  outputs/evaluations/data_audit/happyscribe/cleaning/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable

from scrapers.happyscribe.clean_happyscribe import clean_happyscribe_transcript


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> Iterable[tuple[dict, int]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), line_no
            except json.JSONDecodeError as exc:
                print(f"[BAD JSON] {path}:{line_no} {exc}", file=sys.stderr)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_files_from_input(input_dir: Path, max_files: int | None = None) -> list[Path]:
    files = sorted(input_dir.glob("*/source_documents.jsonl"))
    files = [p for p in files if p.parent.name != "catalog"]
    if max_files is not None:
        files = files[:max_files]
    return files


def clean_document(
    obj: dict,
    cleaned_text: str,
    clean_stats: dict,
    cleaned_hash: str,
) -> dict:
    new_obj = deepcopy(obj)
    metadata = dict(new_obj.get("metadata") or {})

    original_hash = new_obj.get("content_hash")
    if original_hash:
        metadata["source_content_hash"] = original_hash

    metadata["cleaning"] = clean_stats

    new_obj["clean_text"] = cleaned_text
    new_obj["content_hash"] = cleaned_hash
    new_obj["cleaning_method"] = clean_stats.get("cleaning_version")
    new_obj["metadata"] = metadata

    return new_obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean HappyScribe source_documents.jsonl files."
    )
    parser.add_argument(
        "--input-dir",
        default="data/interim/podcasts/happyscribe",
        help="Directory containing <podcast_slug>/source_documents.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/cleaned/podcasts/happyscribe",
        help="Directory to write cleaned per-podcast JSONL files.",
    )
    parser.add_argument(
        "--report-dir",
        default="outputs/evaluations/data_audit/happyscribe/cleaning",
        help="Directory to write cleaning summary reports.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=1000,
        help="Skip documents shorter than this after cleaning.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="For testing: only process the first N podcast source files.",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Do not skip exact duplicate cleaned transcripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run cleaning and reports without writing cleaned JSONL files.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)

    source_files = source_files_from_input(input_dir, max_files=args.max_files)
    if not source_files:
        raise SystemExit(f"No source_documents.jsonl files found under {input_dir}")

    seen_clean_hashes: set[str] = set()
    skipped_rows: list[dict] = []
    shortest_rows: list[dict] = []
    largest_reduction_rows: list[dict] = []
    podcast_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    aggregate_remove_reasons: Counter[str] = Counter()

    docs_seen = 0
    docs_written = 0
    total_original_chars = 0
    total_cleaned_chars = 0
    total_removed_blocks = 0
    total_removed_inline_phrases = 0

    for source_file in source_files:
        podcast_slug = source_file.parent.name
        out_file = output_dir / podcast_slug / "source_documents_clean.jsonl"
        cleaned_docs: list[dict] = []

        for obj, line_no in iter_jsonl(source_file):
            docs_seen += 1
            original_text = obj.get("clean_text") or ""
            total_original_chars += len(original_text)

            clean_result = clean_happyscribe_transcript(original_text)
            cleaned_text = clean_result.text
            clean_stats = clean_result.stats
            cleaned_hash = sha256_text(cleaned_text)

            total_cleaned_chars += len(cleaned_text)
            total_removed_blocks += int(clean_stats.get("removed_blocks", 0))
            total_removed_inline_phrases += int(clean_stats.get("removed_inline_phrases", 0))
            aggregate_remove_reasons.update(clean_stats.get("remove_reasons", {}))

            base_report_row = {
                "podcast_slug": podcast_slug,
                "title": obj.get("title"),
                "url": obj.get("url"),
                "source_file": str(source_file),
                "line_no": line_no,
                "original_chars": len(original_text),
                "cleaned_chars": len(cleaned_text),
                "removed_chars": max(0, len(original_text) - len(cleaned_text)),
                "cleaned_hash": cleaned_hash,
            }

            if len(cleaned_text) < args.min_chars:
                reason_counts["too_short_after_cleaning"] += 1
                skipped_rows.append(
                    {**base_report_row, "skip_reason": "too_short_after_cleaning"}
                )
                continue

            if not args.no_dedupe and cleaned_hash in seen_clean_hashes:
                reason_counts["duplicate_clean_content_hash"] += 1
                skipped_rows.append(
                    {**base_report_row, "skip_reason": "duplicate_clean_content_hash"}
                )
                continue

            seen_clean_hashes.add(cleaned_hash)
            new_obj = clean_document(obj, cleaned_text, clean_stats, cleaned_hash)
            cleaned_docs.append(new_obj)
            docs_written += 1
            podcast_counts[podcast_slug] += 1

            shortest_rows.append(base_report_row)
            largest_reduction_rows.append(base_report_row)

        if not args.dry_run:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with out_file.open("w", encoding="utf-8") as f:
                for doc in cleaned_docs:
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    shortest_rows = sorted(shortest_rows, key=lambda x: x["cleaned_chars"])[:200]
    largest_reduction_rows = sorted(
        largest_reduction_rows,
        key=lambda x: x["removed_chars"],
        reverse=True,
    )[:200]

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "report_dir": str(report_dir),
        "source_files": len(source_files),
        "documents_seen": docs_seen,
        "documents_written": docs_written,
        "documents_skipped": len(skipped_rows),
        "skip_reasons": dict(reason_counts),
        "dedupe_enabled": not args.no_dedupe,
        "min_chars": args.min_chars,
        "avg_original_chars": round(total_original_chars / docs_seen, 1) if docs_seen else None,
        "avg_cleaned_chars": round(total_cleaned_chars / docs_seen, 1) if docs_seen else None,
        "total_removed_blocks": total_removed_blocks,
        "total_removed_inline_phrases": total_removed_inline_phrases,
        "aggregate_remove_reasons": dict(aggregate_remove_reasons),
        "dry_run": args.dry_run,
    }

    write_json(report_dir / "summary.json", summary)
    write_jsonl(report_dir / "skipped_documents.jsonl", skipped_rows)
    write_jsonl(report_dir / "shortest_cleaned_documents.jsonl", shortest_rows)
    write_jsonl(report_dir / "largest_reduction_documents.jsonl", largest_reduction_rows)
    write_jsonl(
        report_dir / "podcast_cleaning_counts.jsonl",
        [
            {"podcast_slug": slug, "documents_written": count}
            for slug, count in podcast_counts.most_common()
        ],
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Cleaning report written to: {report_dir}")


if __name__ == "__main__":
    main()
