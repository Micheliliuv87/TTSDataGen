from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# Basic IO
# ---------------------------------------------------------------------


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


# ---------------------------------------------------------------------
# Dialogue parsing
# ---------------------------------------------------------------------


SOURCE_APPENDIX_RE = re.compile(
    r"\n+##\s*(?:Source Appendix|Source Notes|Sources)\b",
    flags=re.IGNORECASE,
)

ROUND_HEADING_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:Round\s+(\d+)|第\s*(\d+)\s*(?:轮|回合))\s*$",
    flags=re.IGNORECASE,
)

NUMBERED_DIALOGUE_RE = re.compile(
    r"^\s*(\d+)\.\s*([^:：\n]{1,40})\s*[:：]\s*(.*?)\s*$"
)


def split_body_and_appendix(markdown_text: str) -> Tuple[str, str, bool]:
    match = SOURCE_APPENDIX_RE.search(markdown_text)
    if not match:
        return markdown_text.rstrip(), "", False

    body = markdown_text[: match.start()].rstrip()
    appendix = markdown_text[match.start() :].strip()
    return body, appendix, True


def normalize_speaker_label(label: Any) -> str:
    label = clean_text(label)
    label = re.sub(r"^speaker\s+", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+", "", label)
    return label.lower()


def parse_round_headings(body: str) -> List[Dict[str, Any]]:
    headings: List[Dict[str, Any]] = []

    for line_index, raw_line in enumerate(body.splitlines(), start=1):
        match = ROUND_HEADING_RE.match(raw_line)
        if not match:
            continue

        round_num = safe_int(match.group(1) or match.group(2), -1)
        if round_num <= 0:
            continue

        headings.append(
            {
                "round": round_num,
                "line_index": line_index,
                "raw": raw_line.rstrip(),
            }
        )

    return headings


def parse_numbered_dialogue_lines(body: str) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    current_round: Optional[int] = None

    for line_index, raw_line in enumerate(body.splitlines(), start=1):
        round_match = ROUND_HEADING_RE.match(raw_line)
        if round_match:
            current_round = safe_int(round_match.group(1) or round_match.group(2), -1)
            continue

        match = NUMBERED_DIALOGUE_RE.match(raw_line)
        if not match:
            continue

        number = safe_int(match.group(1), -1)
        speaker = clean_text(match.group(2))
        text = clean_text(match.group(3))

        parsed.append(
            {
                "number": number,
                "speaker": speaker,
                "normalized_speaker": normalize_speaker_label(speaker),
                "text": text,
                "line_index": line_index,
                "round": current_round,
                "raw": raw_line.rstrip(),
                "char_count": count_content_chars(text),
                "cjk_char_count": count_cjk_chars(text),
            }
        )

    return parsed


# ---------------------------------------------------------------------
# Stats and checks
# ---------------------------------------------------------------------


def count_cjk_chars(text: Any) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", str(text or "")))


def count_content_chars(text: Any) -> int:
    t = clean_text(text)
    # Count non-space characters. This works reasonably for both Chinese and English.
    return len(re.sub(r"\s+", "", t))


def infer_rounds_from_meta(meta: Dict[str, Any]) -> Optional[int]:
    rounds = safe_int(meta.get("rounds"), -1)
    if rounds > 0:
        return rounds

    total_lines = safe_int(meta.get("total_dialogue_lines"), -1)
    if total_lines > 0 and total_lines % 2 == 0:
        return total_lines // 2

    return None


def infer_rounds_from_text(round_headings: List[Dict[str, Any]]) -> Optional[int]:
    if not round_headings:
        return None
    return max(item["round"] for item in round_headings)


def find_duplicates(items: List[int]) -> List[int]:
    seen = set()
    duplicates = set()
    for item in items:
        if item in seen:
            duplicates.add(item)
        seen.add(item)
    return sorted(duplicates)


def expected_range(n: int) -> List[int]:
    return list(range(1, n + 1))


def check_contiguous_sequence(values: List[int], expected_n: int) -> Dict[str, Any]:
    expected = set(expected_range(expected_n))
    actual = set(values)

    return {
        "missing": sorted(expected - actual),
        "unexpected": sorted(v for v in actual if v not in expected),
        "duplicates": find_duplicates(values),
    }


def detect_forbidden_leaks(body: str) -> List[Dict[str, Any]]:
    """
    Check only the dialogue body, not deterministic Source Appendix.

    This validator should detect pipeline artifacts, not ordinary topic language.
    A match should require clear evidence that the generated dialogue exposed
    retrieval/source-management machinery, citation scaffolding, metadata, IDs,
    or appendix/source-note instructions.
    """
    patterns = [
        # URLs and explicit machine-readable identifiers.
        ("url", r"https?://\S+"),
        ("chunk_id", r"\bchunk[_\s-]*id\b|\bchunk[_\s-]*\d+\b"),
        ("doc_id", r"\bdoc[_\s-]*id\b"),
        ("metadata", r"\bmetadata\b"),

        # English pipeline/source-management leakage.
        ("source_anchor_pack", r"\bsource[_\s-]*anchor[_\s-]*pack\b"),
        ("source_anchors", r"\bsource[_\s-]*anchors?\b"),
        ("source_appendix", r"\bsource\s+appendix\b"),
        ("source_notes", r"\bsource\s+notes?\b"),
        ("retrieval", r"\bretrieval\b|\bretrieved\b"),
        ("transcript_label", r"\btranscripts?\b"),
        ("citation_language", r"\bcitations?\b|\bcite\b|\bcited\b"),

        # Chinese pipeline/source-management leakage.
        # These patterns require system-like context rather than ordinary words.
        (
            "zh_source_leak",
            r"(?:根据|依据|参考|来自|使用|结合)"
            r"(?:上述|以下|这些|本次|提供的|检索到的)?"
            r"(?:来源材料|原始材料|检索结果|检索内容|转录稿|转写稿|附录材料)"
        ),
        ("zh_retrieval_leak", r"检索结果|检索到的内容|召回结果|相似度分数|向量检索"),
        ("zh_metadata_leak", r"元数据|材料编号|文档编号|片段编号|引用链接"),
        ("zh_appendix_leak", r"来源附录|材料附录|Source Appendix|Source Notes"),
        ("zh_anchor_leak", r"source anchor|source anchors|锚点材料|材料锚点"),
    ]

    leaks: List[Dict[str, Any]] = []

    for label, pattern in patterns:
        for match in re.finditer(pattern, body, flags=re.IGNORECASE):
            start = max(0, match.start() - 60)
            end = min(len(body), match.end() + 60)
            leaks.append(
                {
                    "type": label,
                    "match": match.group(0),
                    "context": clean_text(body[start:end]),
                }
            )

    return leaks


def check_round_line_mapping(
    dialogue_lines: List[Dict[str, Any]],
    *,
    expected_rounds: int,
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []

    for item in dialogue_lines:
        number = item["number"]
        actual_round = item.get("round")

        if number <= 0:
            continue

        expected_round = (number + 1) // 2

        if expected_round < 1 or expected_round > expected_rounds:
            continue

        if actual_round is None or actual_round <= 0:
            issues.append(
                {
                    "number": number,
                    "expected_round": expected_round,
                    "actual_round": actual_round,
                    "issue": "dialogue_line_not_under_round_heading",
                }
            )
            continue

        if actual_round != expected_round:
            issues.append(
                {
                    "number": number,
                    "expected_round": expected_round,
                    "actual_round": actual_round,
                    "issue": "dialogue_line_under_wrong_round_heading",
                }
            )

    return issues


def build_length_stats(
    dialogue_lines: List[Dict[str, Any]],
    *,
    short_line_chars: int,
    developed_line_chars: int,
) -> Dict[str, Any]:
    counts = [int(item.get("char_count", 0)) for item in dialogue_lines]
    cjk_counts = [int(item.get("cjk_char_count", 0)) for item in dialogue_lines]

    if not counts:
        return {
            "line_count": 0,
            "avg_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "short_line_count": 0,
            "short_line_ratio": 0.0,
            "developed_line_count": 0,
            "developed_line_ratio": 0.0,
            "avg_cjk_chars": 0,
        }

    short_count = sum(1 for x in counts if x < short_line_chars)
    developed_count = sum(1 for x in counts if x >= developed_line_chars)

    return {
        "line_count": len(counts),
        "avg_chars": round(mean(counts), 2),
        "min_chars": min(counts),
        "max_chars": max(counts),
        "short_line_threshold": short_line_chars,
        "short_line_count": short_count,
        "short_line_ratio": round(short_count / len(counts), 3),
        "developed_line_threshold": developed_line_chars,
        "developed_line_count": developed_count,
        "developed_line_ratio": round(developed_count / len(counts), 3),
        "avg_cjk_chars": round(mean(cjk_counts), 2),
    }


def detect_duplicate_dialogue_texts(dialogue_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Dict[str, int] = {}
    duplicates: List[Dict[str, Any]] = []

    for item in dialogue_lines:
        text_key = clean_text(item.get("text", "")).lower()
        if not text_key:
            continue

        # Ignore very short generic text; validator should not overreact.
        if len(text_key) < 20:
            continue

        if text_key in seen:
            duplicates.append(
                {
                    "first_number": seen[text_key],
                    "duplicate_number": item.get("number"),
                    "text": item.get("text", ""),
                }
            )
        else:
            seen[text_key] = item.get("number")

    return duplicates


def validate_dialogue(
    *,
    dialogue_text: str,
    expected_rounds: Optional[int],
    speaker_a: str,
    speaker_b: str,
    short_line_chars: int,
    developed_line_chars: int,
    max_short_line_ratio: float,
    min_developed_line_ratio: float,
    require_source_appendix: bool,
) -> Dict[str, Any]:
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    body, appendix, appendix_present = split_body_and_appendix(dialogue_text)
    round_headings = parse_round_headings(body)
    dialogue_lines = parse_numbered_dialogue_lines(body)

    inferred_rounds = expected_rounds or infer_rounds_from_text(round_headings)

    # Generic fallback for standalone validation:
    # If no meta/round headings are available but numbered A/B lines exist,
    # infer rounds from the largest numbered dialogue line.
    if inferred_rounds is None and dialogue_lines:
        max_line_number = max(
            [item["number"] for item in dialogue_lines if item.get("number", -1) > 0],
            default=-1,
        )
        if max_line_number > 0 and max_line_number % 2 == 0:
            inferred_rounds = max_line_number // 2

    if expected_rounds is None and inferred_rounds is None:
        warnings.append(
            {
                "type": "rounds_not_provided_or_inferred",
                "message": "No expected rounds were provided and no Round headings or complete numbered A/B line sequence could be used to infer rounds.",
            }
        )

    if inferred_rounds is not None and inferred_rounds > 0:
        expected_total_lines = inferred_rounds * 2
    else:
        expected_total_lines = None

    # Source appendix presence.
    if require_source_appendix and not appendix_present:
        errors.append(
            {
                "type": "missing_source_appendix",
                "message": "Source Appendix is required but was not found.",
            }
        )

    # Round headings.
    round_numbers = [item["round"] for item in round_headings]

    if inferred_rounds is not None and inferred_rounds > 0:
        round_seq = check_contiguous_sequence(round_numbers, inferred_rounds)

        if round_headings and len(round_numbers) != inferred_rounds:
            errors.append(
                {
                    "type": "round_heading_count_mismatch",
                    "expected": inferred_rounds,
                    "actual": len(round_numbers),
                }
            )

        if round_headings and round_seq["missing"]:
            errors.append(
                {
                    "type": "missing_round_headings",
                    "missing": round_seq["missing"],
                }
            )

        if round_seq["unexpected"]:
            errors.append(
                {
                    "type": "unexpected_round_headings",
                    "unexpected": round_seq["unexpected"],
                }
            )

        if round_seq["duplicates"]:
            errors.append(
                {
                    "type": "duplicate_round_headings",
                    "duplicates": round_seq["duplicates"],
                }
            )

    # Numbered dialogue lines.
    line_numbers = [item["number"] for item in dialogue_lines]

    if expected_total_lines is not None:
        line_seq = check_contiguous_sequence(line_numbers, expected_total_lines)

        if len(dialogue_lines) != expected_total_lines:
            errors.append(
                {
                    "type": "dialogue_line_count_mismatch",
                    "expected": expected_total_lines,
                    "actual": len(dialogue_lines),
                }
            )

        if line_seq["missing"]:
            errors.append(
                {
                    "type": "missing_dialogue_line_numbers",
                    "missing": line_seq["missing"],
                }
            )

        if line_seq["unexpected"]:
            errors.append(
                {
                    "type": "unexpected_dialogue_line_numbers",
                    "unexpected": line_seq["unexpected"],
                }
            )

        if line_seq["duplicates"]:
            errors.append(
                {
                    "type": "duplicate_dialogue_line_numbers",
                    "duplicates": line_seq["duplicates"],
                }
            )

    # Ordering.
    if line_numbers != sorted(line_numbers):
        errors.append(
            {
                "type": "dialogue_line_numbers_out_of_order",
                "actual_order": line_numbers,
            }
        )

    # Speaker alternation.
    norm_a = normalize_speaker_label(speaker_a)
    norm_b = normalize_speaker_label(speaker_b)

    speaker_errors: List[Dict[str, Any]] = []
    for item in dialogue_lines:
        number = item["number"]
        actual = item["normalized_speaker"]

        if number <= 0:
            continue

        expected_speaker = norm_a if number % 2 == 1 else norm_b
        expected_raw = speaker_a if number % 2 == 1 else speaker_b

        if actual != expected_speaker:
            speaker_errors.append(
                {
                    "number": number,
                    "actual_speaker": item["speaker"],
                    "expected_speaker": expected_raw,
                    "raw": item["raw"],
                }
            )

    if speaker_errors:
        errors.append(
            {
                "type": "speaker_alternation_errors",
                "count": len(speaker_errors),
                "items": speaker_errors[:30],
            }
        )

    # Round-line mapping.
    # Only enforce this when round headings exist. If a dialogue uses only
    # numbered A/B lines, line-count and speaker checks are sufficient.
    if round_headings and inferred_rounds is not None and inferred_rounds > 0:
        mapping_issues = check_round_line_mapping(
            dialogue_lines,
            expected_rounds=inferred_rounds,
        )
        if mapping_issues:
            errors.append(
                {
                    "type": "round_line_mapping_errors",
                    "count": len(mapping_issues),
                    "items": mapping_issues[:30],
                }
            )

    # Source/retrieval leakage in dialogue body only.
    leaks = detect_forbidden_leaks(body)
    if leaks:
        errors.append(
            {
                "type": "source_or_retrieval_leakage_in_dialogue_body",
                "count": len(leaks),
                "items": leaks[:30],
            }
        )

    # Duplicate dialogue content.
    duplicate_texts = detect_duplicate_dialogue_texts(dialogue_lines)
    if duplicate_texts:
        warnings.append(
            {
                "type": "duplicate_dialogue_texts",
                "count": len(duplicate_texts),
                "items": duplicate_texts[:20],
            }
        )

    # Length stats.
    length_stats = build_length_stats(
        dialogue_lines,
        short_line_chars=short_line_chars,
        developed_line_chars=developed_line_chars,
    )

    if dialogue_lines and length_stats["short_line_ratio"] > max_short_line_ratio:
        warnings.append(
            {
                "type": "high_short_line_ratio",
                "short_line_ratio": length_stats["short_line_ratio"],
                "max_short_line_ratio": max_short_line_ratio,
                "short_line_count": length_stats["short_line_count"],
                "line_count": length_stats["line_count"],
            }
        )

    if dialogue_lines and length_stats["developed_line_ratio"] < min_developed_line_ratio:
        warnings.append(
            {
                "type": "low_developed_line_ratio",
                "developed_line_ratio": length_stats["developed_line_ratio"],
                "min_developed_line_ratio": min_developed_line_ratio,
                "developed_line_count": length_stats["developed_line_count"],
                "line_count": length_stats["line_count"],
            }
        )

    # -----------------------------------------------------------------
    # Verdicts
    # -----------------------------------------------------------------
    # Keep "passed" backward-compatible: it means mechanical validation
    # passed, not that the dialogue is high quality.
    mechanical_passed = len(errors) == 0

    quality_blocking_warning_types = {
        "duplicate_dialogue_texts",
        "high_short_line_ratio",
        "low_developed_line_ratio",
    }

    quality_blocking_warnings = [
        item for item in warnings
        if item.get("type") in quality_blocking_warning_types
    ]

    quality_passed = mechanical_passed and not quality_blocking_warnings
    needs_rewrite = mechanical_passed and bool(quality_blocking_warnings)

    if not mechanical_passed:
        verdict = "failed_mechanical"
    elif needs_rewrite:
        verdict = "needs_rewrite"
    else:
        verdict = "passed"

    warning_type_counts: Dict[str, int] = {}
    for item in warnings:
        warning_type = str(item.get("type", "unknown_warning"))
        warning_type_counts[warning_type] = warning_type_counts.get(warning_type, 0) + 1

    error_type_counts: Dict[str, int] = {}
    for item in errors:
        error_type = str(item.get("type", "unknown_error"))
        error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

    return {
        "passed": mechanical_passed,
        "mechanical_passed": mechanical_passed,
        "quality_passed": quality_passed,
        "needs_rewrite": needs_rewrite,
        "verdict": verdict,
        "errors": errors,
        "warnings": warnings,
        "quality_blocking_warnings": quality_blocking_warnings,
        "summary": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "quality_blocking_warning_count": len(quality_blocking_warnings),
            "error_type_counts": error_type_counts,
            "warning_type_counts": warning_type_counts,
        },
        "stats": {
            "expected_rounds": expected_rounds,
            "inferred_rounds": inferred_rounds,
            "expected_total_dialogue_lines": expected_total_lines,
            "round_heading_count": len(round_headings),
            "dialogue_line_count": len(dialogue_lines),
            "source_appendix_present": appendix_present,
            "source_appendix_chars": len(appendix),
            "body_chars": len(body),
            "length": length_stats,
        },
        "parsed_preview": {
            "round_headings": round_headings[:10],
            "dialogue_lines": [
                {
                    "number": item["number"],
                    "speaker": item["speaker"],
                    "round": item["round"],
                    "char_count": item["char_count"],
                    "text": item["text"][:120],
                }
                for item in dialogue_lines[:10]
            ],
        },
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dialogue",
        type=Path,
        required=True,
        help="Path to generated dialogue Markdown file.",
    )
    parser.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="Optional generation meta JSON. Used to infer rounds if --rounds is omitted.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Expected number of A/B rounds. Overrides meta inference.",
    )
    parser.add_argument("--speaker_a", type=str, default="A")
    parser.add_argument("--speaker_b", type=str, default="B")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Output validation JSON path. Defaults to dialogue filename with .validation.json suffix.",
    )

    parser.add_argument(
        "--short_line_chars",
        type=int,
        default=50,
        help="Warn when many speaker lines are shorter than this many non-space characters.",
    )
    parser.add_argument(
        "--developed_line_chars",
        type=int,
        default=90,
        help="Warn when too few speaker lines reach this many non-space characters.",
    )
    parser.add_argument(
        "--max_short_line_ratio",
        type=float,
        default=0.35,
        help="Warning threshold for short-line ratio.",
    )
    parser.add_argument(
        "--min_developed_line_ratio",
        type=float,
        default=0.50,
        help="Warning threshold for developed-line ratio.",
    )
    parser.add_argument(
        "--require_source_appendix",
        action="store_true",
        help="Require deterministic Source Appendix to exist.",
    )

    args = parser.parse_args()

    if not args.dialogue.exists():
        raise FileNotFoundError(f"Dialogue file not found: {args.dialogue}")

    dialogue_text = args.dialogue.read_text(encoding="utf-8")

    expected_rounds = args.rounds
    meta: Dict[str, Any] = {}

    if expected_rounds is None and args.meta is not None and args.meta.exists():
        meta = load_json(args.meta)
        expected_rounds = infer_rounds_from_meta(meta)

    report = validate_dialogue(
        dialogue_text=dialogue_text,
        expected_rounds=expected_rounds,
        speaker_a=args.speaker_a,
        speaker_b=args.speaker_b,
        short_line_chars=args.short_line_chars,
        developed_line_chars=args.developed_line_chars,
        max_short_line_ratio=args.max_short_line_ratio,
        min_developed_line_ratio=args.min_developed_line_ratio,
        require_source_appendix=args.require_source_appendix,
    )

    report["_meta"] = {
        "dialogue": str(args.dialogue),
        "meta": str(args.meta) if args.meta else "",
        "speaker_a": args.speaker_a,
        "speaker_b": args.speaker_b,
        "rounds_source": "cli" if args.rounds is not None else "meta_or_inferred",
    }

    output_path = args.output_path or args.dialogue.with_suffix(".validation.json")
    write_json(output_path, report)

    status = "PASSED" if report["passed"] else "FAILED"
    print(f"Validation: {status}")
    print(f"Wrote validation report: {output_path}")
    print(f"Expected rounds: {report['stats'].get('expected_rounds')}")
    print(f"Inferred rounds: {report['stats'].get('inferred_rounds')}")
    print(f"Round headings: {report['stats'].get('round_heading_count')}")
    print(f"Dialogue lines: {report['stats'].get('dialogue_line_count')}")
    print(f"Errors: {len(report.get('errors', []))}")
    print(f"Warnings: {len(report.get('warnings', []))}")

    length = report["stats"].get("length", {})
    print(
        "Length stats: "
        f"avg_chars={length.get('avg_chars')} | "
        f"short_ratio={length.get('short_line_ratio')} | "
        f"developed_ratio={length.get('developed_line_ratio')}"
    )


if __name__ == "__main__":
    main()