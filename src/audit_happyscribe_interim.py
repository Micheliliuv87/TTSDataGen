#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line), path, line_no
            except json.JSONDecodeError as exc:
                print(f"[BAD JSON] {path}:{line_no} {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="data/interim/podcasts/happyscribe",
    )
    parser.add_argument(
        "--report-dir",
        default="outputs/evaluations/data_audit/happyscribe/interim_audit",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(input_dir.glob("*/source_documents.jsonl"))

    docs = []
    empty_podcasts = []
    podcast_counts = Counter()
    url_counts = Counter()
    hash_counts = Counter()
    doc_id_counts = Counter()
    length_rows = []

    for source_file in source_files:
        podcast_slug = source_file.parent.name
        count = 0

        for obj, path, line_no in iter_jsonl(source_file):
            count += 1

            url = obj.get("url")
            doc_id = obj.get("doc_id")
            content_hash = obj.get("content_hash")
            clean_text = obj.get("clean_text", "")

            podcast_counts[podcast_slug] += 1

            if url:
                url_counts[url] += 1
            if doc_id:
                doc_id_counts[doc_id] += 1
            if content_hash:
                hash_counts[content_hash] += 1

            length_rows.append(
                {
                    "podcast_slug": podcast_slug,
                    "title": obj.get("title"),
                    "url": url,
                    "chars": len(clean_text),
                    "content_hash": content_hash,
                    "file": str(path),
                    "line_no": line_no,
                }
            )

            docs.append(obj)

        if count == 0:
            empty_podcasts.append(podcast_slug)

    duplicate_urls = [u for u, c in url_counts.items() if c > 1]
    duplicate_hashes = [h for h, c in hash_counts.items() if c > 1]
    duplicate_doc_ids = [d for d, c in doc_id_counts.items() if c > 1]

    length_rows_sorted = sorted(length_rows, key=lambda x: x["chars"])

    summary = {
        "input_dir": str(input_dir),
        "source_files": len(source_files),
        "nonempty_source_files": sum(1 for c in podcast_counts.values() if c > 0),
        "empty_source_files": len(empty_podcasts),
        "documents": len(docs),
        "duplicate_urls": len(duplicate_urls),
        "duplicate_content_hashes": len(duplicate_hashes),
        "duplicate_doc_ids": len(duplicate_doc_ids),
        "min_chars": length_rows_sorted[0]["chars"] if length_rows_sorted else None,
        "max_chars": length_rows_sorted[-1]["chars"] if length_rows_sorted else None,
        "avg_chars": round(sum(x["chars"] for x in length_rows) / len(length_rows), 1)
        if length_rows
        else None,
    }

    def write_json(path: Path, obj):
        with path.open("w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def write_jsonl(path: Path, rows):
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    write_json(report_dir / "summary.json", summary)
    write_json(report_dir / "empty_podcasts.json", empty_podcasts)

    write_jsonl(
        report_dir / "podcast_counts.jsonl",
        [
            {"podcast_slug": slug, "documents": count}
            for slug, count in podcast_counts.most_common()
        ],
    )

    write_jsonl(
        report_dir / "shortest_documents.jsonl",
        length_rows_sorted[:200],
    )

    write_jsonl(
        report_dir / "longest_documents.jsonl",
        list(reversed(length_rows_sorted[-200:])),
    )

    write_json(
        report_dir / "duplicate_urls.json",
        {u: url_counts[u] for u in duplicate_urls},
    )

    write_json(
        report_dir / "duplicate_content_hashes.json",
        {h: hash_counts[h] for h in duplicate_hashes},
    )

    write_json(
        report_dir / "duplicate_doc_ids.json",
        {d: doc_id_counts[d] for d in duplicate_doc_ids},
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Audit written to: {report_dir}")


if __name__ == "__main__":
    main()