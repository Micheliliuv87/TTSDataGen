# src/chunk_sources.py

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


TIMESTAMP_LINE_RE = re.compile(
    r"^\s*\[?(?P<ts>\d{1,2}:\d{2}(?::\d{2})?)\]?\s*(?P<rest>.*)$"
)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {e}") from e


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def get_doc_text(doc: Dict[str, Any]) -> str:
    for key in ["clean_text", "text_clean", "cleaned_text", "text", "content", "transcript"]:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def infer_podcast_slug(source_file: Path, input_dir: Path, doc: Dict[str, Any]) -> str:
    for key in ["podcast_slug", "show_slug", "series_slug"]:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    try:
        rel = source_file.relative_to(input_dir)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except ValueError:
        pass

    return source_file.parent.name


def split_long_text(text: str, max_chars: int) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    parts: List[str] = []
    current: List[str] = []
    current_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        if len(sent) > max_chars:
            if current:
                parts.append(" ".join(current).strip())
                current = []
                current_len = 0

            for i in range(0, len(sent), max_chars):
                parts.append(sent[i : i + max_chars].strip())
            continue

        if current and current_len + len(sent) + 1 > max_chars:
            parts.append(" ".join(current).strip())
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent) + 1

    if current:
        parts.append(" ".join(current).strip())

    return [p for p in parts if p]


def extract_timestamp_blocks(text: str, max_chars: int) -> Tuple[List[Dict[str, Optional[str]]], bool]:
    """
    Returns:
      blocks: [{"timestamp": str | None, "text": str}]
      timestamp_aware: True if timestamp boundaries were detected.
    """

    lines = text.splitlines()
    raw_blocks: List[Dict[str, Optional[str]]] = []

    current_ts: Optional[str] = None
    current_lines: List[str] = []
    timestamp_count = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current_lines:
                current_lines.append("")
            continue

        m = TIMESTAMP_LINE_RE.match(stripped)
        is_timestamp_line = False

        if m:
            ts = m.group("ts")
            rest = m.group("rest").strip()

            # Avoid treating normal prose like "12:30 pm" as transcript timestamp.
            if ":" in ts and not re.search(r"\b(am|pm)\b", rest.lower()):
                is_timestamp_line = True

        if is_timestamp_line:
            if current_lines:
                raw_blocks.append(
                    {
                        "timestamp": current_ts,
                        "text": "\n".join(current_lines).strip(),
                    }
                )

            current_ts = m.group("ts")
            current_lines = []
            timestamp_count += 1

            rest = m.group("rest").strip()
            if rest:
                current_lines.append(rest)
        else:
            current_lines.append(stripped)

    if current_lines:
        raw_blocks.append(
            {
                "timestamp": current_ts,
                "text": "\n".join(current_lines).strip(),
            }
        )

    timestamp_aware = timestamp_count >= 2

    # Fallback: paragraph blocks if timestamps are absent or unreliable.
    if not timestamp_aware:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
        raw_blocks = [{"timestamp": None, "text": p} for p in paragraphs]

    # Split unusually large blocks.
    blocks: List[Dict[str, Optional[str]]] = []
    for block in raw_blocks:
        block_text = str(block.get("text") or "").strip()
        if not block_text:
            continue

        for part in split_long_text(block_text, max_chars=max_chars):
            blocks.append(
                {
                    "timestamp": block.get("timestamp"),
                    "text": part,
                }
            )

    return blocks, timestamp_aware


def chunk_blocks(
    blocks: List[Dict[str, Optional[str]]],
    target_chars: int,
    max_chars: int,
    min_chunk_chars: int,
    overlap_blocks: int,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    n = len(blocks)
    i = 0

    while i < n:
        current: List[Dict[str, Optional[str]]] = []
        current_len = 0
        j = i

        while j < n:
            block_text = str(blocks[j].get("text") or "").strip()
            block_len = len(block_text)

            if current and current_len + block_len > max_chars:
                break

            current.append(blocks[j])
            current_len += block_len + 2
            j += 1

            if current_len >= target_chars:
                break

        if not current:
            i += 1
            continue

        chunk_text = "\n\n".join(str(b.get("text") or "").strip() for b in current).strip()

        chunks.append(
            {
                "text": chunk_text,
                "start_timestamp": current[0].get("timestamp"),
                "end_timestamp": current[-1].get("timestamp"),
                "block_count": len(current),
                "char_count": len(chunk_text),
            }
        )

        if j >= n:
            break

        next_i = max(i + 1, j - overlap_blocks)
        i = next_i

    # Merge or remove very small chunks.
    cleaned_chunks: List[Dict[str, Any]] = []

    for chunk in chunks:
        if chunk["char_count"] >= min_chunk_chars:
            cleaned_chunks.append(chunk)
            continue

        # Prefer merging tiny chunks into the previous chunk if it stays under max_chars.
        if cleaned_chunks and cleaned_chunks[-1]["char_count"] + chunk["char_count"] <= max_chars:
            prev = cleaned_chunks[-1]
            prev["text"] = (prev["text"] + "\n\n" + chunk["text"]).strip()
            prev["end_timestamp"] = chunk["end_timestamp"] or prev["end_timestamp"]
            prev["block_count"] += chunk["block_count"]
            prev["char_count"] = len(prev["text"])
            continue

        # If this is a leading tiny chunk, keep it temporarily.
        # It may be merged forward in the next pass.
        cleaned_chunks.append(chunk)

    # Second pass: merge leading or isolated tiny chunks forward when possible.
    final_chunks: List[Dict[str, Any]] = []
    i = 0

    while i < len(cleaned_chunks):
        chunk = cleaned_chunks[i]

        if chunk["char_count"] >= min_chunk_chars:
            final_chunks.append(chunk)
            i += 1
            continue

        # Try to merge tiny chunk into the next chunk.
        if i + 1 < len(cleaned_chunks):
            next_chunk = cleaned_chunks[i + 1]
            merged_text = (chunk["text"] + "\n\n" + next_chunk["text"]).strip()

            if len(merged_text) <= max_chars:
                merged = dict(next_chunk)
                merged["text"] = merged_text
                merged["start_timestamp"] = chunk["start_timestamp"] or next_chunk["start_timestamp"]
                merged["block_count"] = chunk["block_count"] + next_chunk["block_count"]
                merged["char_count"] = len(merged_text)
                final_chunks.append(merged)
                i += 2
                continue

        # Drop tiny isolated chunks unless the whole document only produced this one chunk.
        if len(cleaned_chunks) == 1:
            final_chunks.append(chunk)

        i += 1

    return final_chunks


def build_doc_chunks(
    doc: Dict[str, Any],
    source_file: Path,
    input_dir: Path,
    target_chars: int,
    max_chars: int,
    min_chunk_chars: int,
    overlap_blocks: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    text = get_doc_text(doc)
    if not text:
        return [], {"skip_reason": "missing_text"}

    blocks, timestamp_aware = extract_timestamp_blocks(text, max_chars=max_chars)

    if not blocks:
        return [], {"skip_reason": "no_blocks_after_split"}

    chunks = chunk_blocks(
        blocks=blocks,
        target_chars=target_chars,
        max_chars=max_chars,
        min_chunk_chars=min_chunk_chars,
        overlap_blocks=overlap_blocks,
    )

    if not chunks:
        return [], {"skip_reason": "no_chunks_after_chunking"}

    doc_id = doc.get("doc_id") or doc.get("id")
    url = doc.get("url") or doc.get("source_url")
    title = doc.get("title") or doc.get("episode_title") or doc.get("name")
    podcast_slug = infer_podcast_slug(source_file, input_dir, doc)

    if not doc_id:
        doc_id_seed = f"{podcast_slug}|{title}|{url}|{source_file}"
        doc_id = f"doc_{stable_hash(doc_id_seed)}"

    output_chunks: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}::chunk_{idx:04d}"

        output_chunks.append(
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": idx,
                "provider": "happyscribe",
                "source_type": "podcast_transcript",
                "podcast_slug": podcast_slug,
                "title": title,
                "url": url,
                "start_timestamp": chunk["start_timestamp"],
                "end_timestamp": chunk["end_timestamp"],
                "char_count": chunk["char_count"],
                "block_count": chunk["block_count"],
                "text": chunk["text"],
                "source_file": str(source_file),
                "timestamp_aware": timestamp_aware,
                "chunk_content_hash": stable_hash(normalize_space(chunk["text"])),
            }
        )

    doc_stats = {
        "skip_reason": None,
        "timestamp_aware": timestamp_aware,
        "block_count": len(blocks),
        "chunk_count": len(output_chunks),
        "doc_char_count": len(text),
    }

    return output_chunks, doc_stats


def iter_source_files(input_dir: Path, max_files: Optional[int]) -> List[Path]:
    files = sorted(input_dir.rglob("source_documents_clean.jsonl"))
    if max_files is not None:
        return files[:max_files]
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/processed/cleaned/podcasts/happyscribe"),
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("data/processed/rag/podcasts/happyscribe/chunks.jsonl"),
    )
    parser.add_argument(
        "--summary_path",
        type=Path,
        default=Path("outputs/evaluations/data_audit/happyscribe/chunk_summary.json"),
    )
    parser.add_argument("--target_chars", type=int, default=2800)
    parser.add_argument("--max_chars", type=int, default=3600)
    parser.add_argument("--min_chunk_chars", type=int, default=400)
    parser.add_argument("--overlap_blocks", type=int, default=1)
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--max_docs", type=int, default=None)

    args = parser.parse_args()

    if args.target_chars > args.max_chars:
        raise ValueError("--target_chars must be <= --max_chars")

    source_files = iter_source_files(args.input_dir, args.max_files)

    all_chunks: List[Dict[str, Any]] = []

    docs_seen = 0
    docs_written = 0
    skip_reasons: Counter[str] = Counter()
    timestamp_aware_docs = 0
    non_timestamp_docs = 0
    chunks_per_doc: List[int] = []
    chunk_char_counts: List[int] = []
    duplicate_chunk_hashes: Counter[str] = Counter()

    for source_file in source_files:
        for doc in read_jsonl(source_file):
            if args.max_docs is not None and docs_seen >= args.max_docs:
                break

            docs_seen += 1

            chunks, doc_stats = build_doc_chunks(
                doc=doc,
                source_file=source_file,
                input_dir=args.input_dir,
                target_chars=args.target_chars,
                max_chars=args.max_chars,
                min_chunk_chars=args.min_chunk_chars,
                overlap_blocks=args.overlap_blocks,
            )

            if doc_stats.get("skip_reason"):
                skip_reasons[doc_stats["skip_reason"]] += 1
                continue

            docs_written += 1

            if doc_stats.get("timestamp_aware"):
                timestamp_aware_docs += 1
            else:
                non_timestamp_docs += 1

            chunks_per_doc.append(len(chunks))

            for chunk in chunks:
                chunk_char_counts.append(chunk["char_count"])
                duplicate_chunk_hashes[chunk["chunk_content_hash"]] += 1

            all_chunks.extend(chunks)

        if args.max_docs is not None and docs_seen >= args.max_docs:
            break

    chunks_written = write_jsonl(args.output_path, all_chunks)

    repeated_hashes = {
        h: c for h, c in duplicate_chunk_hashes.items() if c > 1
    }

    summary = {
        "input_dir": str(args.input_dir),
        "output_path": str(args.output_path),
        "source_files_found": len(source_files),
        "documents_seen": docs_seen,
        "documents_written": docs_written,
        "documents_skipped": sum(skip_reasons.values()),
        "skip_reasons": dict(skip_reasons),
        "chunks_written": chunks_written,
        "timestamp_aware_docs": timestamp_aware_docs,
        "non_timestamp_docs": non_timestamp_docs,
        "duplicate_chunk_hashes": len(repeated_hashes),
        "params": {
            "target_chars": args.target_chars,
            "max_chars": args.max_chars,
            "min_chunk_chars": args.min_chunk_chars,
            "overlap_blocks": args.overlap_blocks,
            "max_files": args.max_files,
            "max_docs": args.max_docs,
        },
        "chunk_chars": {
            "min": min(chunk_char_counts) if chunk_char_counts else None,
            "max": max(chunk_char_counts) if chunk_char_counts else None,
            "avg": round(statistics.mean(chunk_char_counts), 2) if chunk_char_counts else None,
            "median": round(statistics.median(chunk_char_counts), 2) if chunk_char_counts else None,
        },
        "chunks_per_doc": {
            "min": min(chunks_per_doc) if chunks_per_doc else None,
            "max": max(chunks_per_doc) if chunks_per_doc else None,
            "avg": round(statistics.mean(chunks_per_doc), 2) if chunks_per_doc else None,
            "median": round(statistics.median(chunks_per_doc), 2) if chunks_per_doc else None,
        },
    }

    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    args.summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()