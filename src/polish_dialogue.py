from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.lmstudio_utils import assert_lmstudio_model_available, make_lmstudio_client


# ---------------------------------------------------------------------
# Basic IO
# ---------------------------------------------------------------------


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def count_nonspace_chars(text: Any) -> int:
    return len(re.sub(r"\s+", "", str(text or "")))


def truncate_text(text: Any, limit: int) -> str:
    text = clean_text(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/object: {path}")

    return data


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}

    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")

    return data


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_rule_jsonl(path: Path, *, missing_ok: bool = True) -> List[Dict[str, Any]]:
    if not path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(f"Polish rules file not found: {path}")

    rules: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            try:
                item = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL rule at {path}:{line_num}: {exc}") from exc

            if isinstance(item, dict):
                rules.append(item)

    return rules


# ---------------------------------------------------------------------
# Markdown parsing
# ---------------------------------------------------------------------


SOURCE_APPENDIX_RE = re.compile(
    r"\n+##\s*(?:Source Appendix|Source Notes|Sources)\b",
    flags=re.IGNORECASE,
)

SPEAKER_LINE_RE = re.compile(r"^(\s*)(\d+)\.\s*([AB])\s*:\s*(.*?)\s*$")


@dataclass
class SpeakerLine:
    raw_index: int
    indent: str
    line_number: int
    speaker: str
    text: str

    @property
    def char_count(self) -> int:
        return count_nonspace_chars(self.text)

    def render(self) -> str:
        return f"{self.indent}{self.line_number}. {self.speaker}: {self.text}"


def split_body_and_appendix(markdown_text: str) -> Dict[str, Any]:
    match = SOURCE_APPENDIX_RE.search(markdown_text)
    if not match:
        return {
            "body": markdown_text.rstrip(),
            "appendix": "",
            "appendix_present": False,
        }

    return {
        "body": markdown_text[: match.start()].rstrip(),
        "appendix": markdown_text[match.start() :].strip(),
        "appendix_present": True,
    }


def parse_speaker_lines(body: str) -> Tuple[List[str], List[SpeakerLine]]:
    raw_lines = body.splitlines()
    speaker_lines: List[SpeakerLine] = []

    for idx, line in enumerate(raw_lines):
        match = SPEAKER_LINE_RE.match(line)
        if not match:
            continue

        indent, number, speaker, text = match.groups()
        speaker_lines.append(
            SpeakerLine(
                raw_index=idx,
                indent=indent,
                line_number=int(number),
                speaker=speaker,
                text=clean_text(text),
            )
        )

    return raw_lines, speaker_lines


def render_body(raw_lines: List[str]) -> str:
    return "\n".join(raw_lines).rstrip()


def strip_markdown_fence(text: str) -> str:
    text = str(text or "").strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json|markdown|md|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def strip_accidental_appendix(text: str) -> str:
    return re.sub(
        r"\n+##\s*(?:Source Appendix|Source Notes|Sources)\b.*$",
        "",
        str(text or "").rstrip(),
        flags=re.IGNORECASE | re.DOTALL,
    ).rstrip()


# ---------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------


def _rule_priority(rule: Dict[str, Any]) -> int:
    try:
        return int(rule.get("priority", 0))
    except Exception:
        return 0


def normalize_rules(
    rules: List[Dict[str, Any]],
    *,
    max_rules: int,
) -> List[Dict[str, Any]]:
    active: List[Dict[str, Any]] = []

    for rule in rules:
        if not isinstance(rule, dict):
            continue

        status = clean_text(rule.get("status", "active")).lower()
        if status and status not in {"active", "enabled"}:
            continue

        active.append(rule)

    active.sort(
        key=lambda item: (
            -_rule_priority(item),
            clean_text(item.get("rule_id", "")),
        )
    )

    if max_rules > 0:
        active = active[:max_rules]

    return active


def rule_ids(rules: List[Dict[str, Any]]) -> List[str]:
    return [
        clean_text(rule.get("rule_id", ""))
        for rule in rules
        if clean_text(rule.get("rule_id", ""))
    ]


def format_rules(
    rules: List[Dict[str, Any]],
    *,
    max_chars: int,
) -> str:
    if not rules:
        return "No active polish rules were loaded."

    max_chars = max(0, int(max_chars or 0))
    blocks: List[str] = []
    used_chars = 0

    for rule in rules:
        rule_id = clean_text(rule.get("rule_id", "unknown_rule"))
        priority = _rule_priority(rule)
        category = clean_text(rule.get("category", "uncategorized"))
        instruction = clean_text(
            rule.get("prompt_instruction")
            or rule.get("instruction")
            or rule.get("rule")
            or ""
        )

        if not instruction:
            continue

        block = (
            f"[{rule_id} | priority={priority} | category={category}]\n"
            f"{instruction}"
        )

        separator = "\n\n" if blocks else ""
        projected = used_chars + len(separator) + len(block)

        if max_chars > 0 and projected > max_chars:
            remaining = max_chars - used_chars - len(separator)
            if remaining >= 180:
                blocks.append(separator + block[:remaining].rstrip() + "...")
            break

        blocks.append(separator + block)
        used_chars = projected

    return "".join(blocks).strip() if blocks else "No active polish rules were loaded."


def load_polish_contract(
    config: Dict[str, Any],
    *,
    polish_rules_override: Optional[Path],
    max_polish_rules_override: Optional[int],
    max_polish_rule_chars_override: Optional[int],
    disable_polish_rules: bool,
) -> Dict[str, Any]:
    rules_cfg = config.get("rules", {}).get("polish", {})

    enabled = bool(rules_cfg.get("enabled", True)) and not disable_polish_rules

    rules_path = (
        polish_rules_override
        or Path(rules_cfg.get("path", "knowledge_base/rules/polish_rules.jsonl"))
    )

    max_rules = (
        max_polish_rules_override
        if max_polish_rules_override is not None
        else int(rules_cfg.get("max_rules", 20))
    )

    max_chars = (
        max_polish_rule_chars_override
        if max_polish_rule_chars_override is not None
        else int(rules_cfg.get("max_chars", 8000))
    )

    rules: List[Dict[str, Any]] = []
    rules_text = "No active polish rules were loaded."

    if enabled:
        raw_rules = load_rule_jsonl(rules_path, missing_ok=True)
        rules = normalize_rules(raw_rules, max_rules=max_rules)
        rules_text = format_rules(rules, max_chars=max_chars)

    return {
        "enabled": enabled,
        "path": rules_path,
        "count": len(rules),
        "max_rules": max_rules,
        "max_chars": max_chars,
        "rules_text": rules_text,
        "rule_ids": rule_ids(rules),
    }


# ---------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------


DEFAULT_POLISH_SYSTEM_PROMPT = """You are a source-free polish agent for a Chinese A/B dialogue pipeline.

Your job is to apply the active polish rules to an existing dialogue body.
Do not expand the dialogue substantially.
Do not use source anchors or external knowledge.
Preserve dialogue structure exactly.

Return strict JSON only.
"""

DEFAULT_POLISH_USER_PREAMBLE = """/no_think

Polish the selected speaker lines according to the active polish rules.
Return strict JSON only.
"""


def load_prompt_templates(config: Dict[str, Any]) -> Dict[str, str]:
    prompt_cfg = config.get("prompt_template", {}) or {}

    system_prompt = str(
        prompt_cfg.get("system_prompt")
        or prompt_cfg.get("system")
        or DEFAULT_POLISH_SYSTEM_PROMPT
    ).strip()

    user_preamble = str(
        prompt_cfg.get("user_preamble")
        or prompt_cfg.get("user_prompt_preamble")
        or DEFAULT_POLISH_USER_PREAMBLE
    ).strip()

    return {
        "system_prompt": system_prompt,
        "user_preamble": user_preamble,
    }


# ---------------------------------------------------------------------
# Validation report slimming
# ---------------------------------------------------------------------


def slim_validation_report(validation_report: Dict[str, Any]) -> Dict[str, Any]:
    if not validation_report:
        return {}

    stats = validation_report.get("stats", {}) or {}
    length = stats.get("length", {}) or {}

    return {
        "passed": validation_report.get("passed"),
        "mechanical_passed": validation_report.get("mechanical_passed", validation_report.get("passed")),
        "quality_passed": validation_report.get("quality_passed"),
        "needs_rewrite": validation_report.get("needs_rewrite"),
        "verdict": validation_report.get("verdict"),
        "summary": validation_report.get("summary", {}),
        "errors": validation_report.get("errors", [])[:8],
        "warnings": validation_report.get("warnings", [])[:12],
        "quality_blocking_warnings": validation_report.get("quality_blocking_warnings", [])[:12],
        "stats": {
            "expected_rounds": stats.get("expected_rounds"),
            "inferred_rounds": stats.get("inferred_rounds"),
            "expected_total_dialogue_lines": stats.get("expected_total_dialogue_lines"),
            "round_heading_count": stats.get("round_heading_count"),
            "dialogue_line_count": stats.get("dialogue_line_count"),
            "source_appendix_present": stats.get("source_appendix_present"),
            "body_chars": stats.get("body_chars"),
            "length": {
                "line_count": length.get("line_count"),
                "avg_chars": length.get("avg_chars"),
                "min_chars": length.get("min_chars"),
                "max_chars": length.get("max_chars"),
                "short_line_threshold": length.get("short_line_threshold"),
                "short_line_count": length.get("short_line_count"),
                "short_line_ratio": length.get("short_line_ratio"),
                "developed_line_threshold": length.get("developed_line_threshold"),
                "developed_line_count": length.get("developed_line_count"),
                "developed_line_ratio": length.get("developed_line_ratio"),
                "avg_cjk_chars": length.get("avg_cjk_chars"),
            },
        },
    }


# ---------------------------------------------------------------------
# Targeting / batching
# ---------------------------------------------------------------------


def parse_line_number_filter(raw: str) -> List[int]:
    raw = clean_text(raw)
    if not raw:
        return []

    values: List[int] = []
    for part in re.split(r"[,\s]+", raw):
        part = part.strip()
        if not part:
            continue

        if "-" in part:
            left, right = part.split("-", 1)
            try:
                start = int(left)
                end = int(right)
            except Exception:
                continue

            if start <= end:
                values.extend(range(start, end + 1))
            else:
                values.extend(range(end, start + 1))
            continue

        try:
            values.append(int(part))
        except Exception:
            continue

    return sorted(set(values))


def select_target_lines(
    speaker_lines: List[SpeakerLine],
    *,
    target_line_numbers: List[int],
) -> List[SpeakerLine]:
    if not target_line_numbers:
        return list(speaker_lines)

    allowed = set(target_line_numbers)
    return [line for line in speaker_lines if line.line_number in allowed]


def make_batches(items: List[SpeakerLine], batch_size: int) -> List[List[SpeakerLine]]:
    batch_size = max(1, int(batch_size or 1))
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def format_dialogue_snapshot(speaker_lines: List[SpeakerLine], *, max_line_chars: int) -> List[Dict[str, Any]]:
    return [
        {
            "line_number": line.line_number,
            "speaker": line.speaker,
            "char_count": line.char_count,
            "text": truncate_text(line.text, max_line_chars),
        }
        for line in speaker_lines
    ]


def format_target_lines(batch_lines: List[SpeakerLine], *, max_line_chars: int) -> List[Dict[str, Any]]:
    return [
        {
            "line_number": line.line_number,
            "speaker": line.speaker,
            "char_count": line.char_count,
            "current_text": truncate_text(line.text, max_line_chars),
        }
        for line in batch_lines
    ]


# ---------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------


def build_polish_batch_messages(
    *,
    batch_index: int,
    total_batches: int,
    batch_lines: List[SpeakerLine],
    all_speaker_lines: List[SpeakerLine],
    validation_report: Dict[str, Any],
    polish_rules_text: str,
    prompt_templates: Dict[str, str],
    max_line_chars_in_prompt: int,
) -> List[Dict[str, str]]:
    payload = {
        "task": "Apply active polish rules to selected speaker lines.",
        "batch": {
            "batch_index": batch_index,
            "total_batches": total_batches,
        },
        "stage_boundary": {
            "stage": "polish",
            "source_free": True,
            "content_expansion_stage_already_done": True,
            "instruction": (
                "Do not perform source-grounded expansion here. "
                "Only make local text edits required by active polish rules."
            ),
        },
        "hard_constraints": [
            "Return strict JSON only.",
            "Return an object with key 'replacements'.",
            "Each replacement must include line_number, speaker, and new_text.",
            "Only return replacements for lines that actually need polishing.",
            "If a line already satisfies the active polish rules, omit it from replacements.",
            "Do not include numbering or speaker prefix inside new_text.",
            "Do not add, remove, reorder, merge, or split dialogue lines.",
            "Do not change speaker labels.",
            "Do not write Markdown, Source Appendix, source notes, citations, URLs, retrieval notes, or explanations.",
            "Do not introduce new factual claims, new examples, new sources, or topic-specific assumptions.",
            "Preserve the original meaning, local dialogue function, and major content of each line.",
            "Do not substantially shorten or expand lines unless an active polish rule requires it.",
        ],
        "active_polish_rules": polish_rules_text,
        "validation_report": validation_report,
        "dialogue_snapshot": format_dialogue_snapshot(
            all_speaker_lines,
            max_line_chars=max_line_chars_in_prompt,
        ),
        "target_lines": format_target_lines(
            batch_lines,
            max_line_chars=max_line_chars_in_prompt,
        ),
        "required_output_schema": {
            "replacements": [
                {
                    "line_number": "integer",
                    "speaker": "A or B",
                    "new_text": "polished speaker text only, without numbering or speaker prefix",
                    "applied_rule_ids": ["optional list of polish rule IDs that motivated the change"],
                }
            ]
        },
    }

    user_prompt = (
        prompt_templates["user_preamble"].rstrip()
        + "\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )

    return [
        {"role": "system", "content": prompt_templates["system_prompt"]},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------
# JSON response parsing
# ---------------------------------------------------------------------


def parse_json_response(raw_text: str) -> Any:
    text = strip_markdown_fence(raw_text)
    text = text.strip()

    if not text:
        raise ValueError("Empty model response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()

    starts = [idx for idx, ch in enumerate(text) if ch in "[{"]
    for start in starts:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            return obj
        except json.JSONDecodeError:
            continue

    raise ValueError("Could not parse JSON from model response.")


def normalize_replacements(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, dict):
        replacements = parsed.get("replacements", [])
    elif isinstance(parsed, list):
        replacements = parsed
    else:
        replacements = []

    if not isinstance(replacements, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in replacements:
        if isinstance(item, dict):
            normalized.append(item)

    return normalized


def sanitize_new_text(text: Any, *, speaker: str) -> str:
    text = strip_accidental_appendix(str(text or ""))
    text = strip_markdown_fence(text)
    text = clean_text(text)

    text = re.sub(r"^\s*\d+\.\s*[AB]\s*:\s*", "", text)
    text = re.sub(rf"^\s*{re.escape(speaker)}\s*:\s*", "", text)
    text = re.sub(r"^\s*[AB]\s*:\s*", "", text)

    return clean_text(text)


# ---------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------


def make_lmstudio_runtime(polisher_cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_url = polisher_cfg.get("base_url", "http://localhost:1234/v1")
    api_key = polisher_cfg.get("api_key", "lm-studio")
    model = polisher_cfg.get("model", "qwen3-32b-mlx")
    timeout_seconds = int(polisher_cfg.get("timeout_seconds", 1200))

    assert_lmstudio_model_available(
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=30,
    )

    client = make_lmstudio_client(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )

    return {
        "client": client,
        "model": model,
        "temperature": float(polisher_cfg.get("temperature", 0.25)),
        "top_p": float(polisher_cfg.get("top_p", 0.85)),
        "max_tokens": int(polisher_cfg.get("max_tokens", 6000)),
    }


def call_lmstudio(
    *,
    runtime: Dict[str, Any],
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    response = runtime["client"].chat.completions.create(
        model=runtime["model"],
        messages=messages,
        temperature=runtime["temperature"],
        top_p=runtime["top_p"],
        max_tokens=runtime["max_tokens"],
    )

    choice = response.choices[0]
    content = choice.message.content or ""
    finish_reason = getattr(choice, "finish_reason", None)

    return {
        "content": content,
        "finish_reason": finish_reason,
    }


# ---------------------------------------------------------------------
# Applying replacements
# ---------------------------------------------------------------------


def apply_replacements(
    *,
    raw_lines: List[str],
    speaker_lines: List[SpeakerLine],
    replacements: List[Dict[str, Any]],
    min_accept_chars: int,
    min_length_ratio: float,
    max_length_ratio: float,
    disable_length_guard: bool,
) -> Dict[str, Any]:
    by_line_number: Dict[int, SpeakerLine] = {
        line.line_number: line for line in speaker_lines
    }

    applied: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    for replacement in replacements:
        try:
            line_number = int(replacement.get("line_number"))
        except Exception:
            rejected.append(
                {
                    "reason": "invalid_line_number",
                    "replacement": replacement,
                }
            )
            continue

        target = by_line_number.get(line_number)
        if not target:
            rejected.append(
                {
                    "line_number": line_number,
                    "reason": "line_number_not_found",
                    "replacement": replacement,
                }
            )
            continue

        speaker = clean_text(replacement.get("speaker"))
        if speaker and speaker != target.speaker:
            rejected.append(
                {
                    "line_number": line_number,
                    "reason": "speaker_mismatch",
                    "expected_speaker": target.speaker,
                    "replacement_speaker": speaker,
                    "replacement": replacement,
                }
            )
            continue

        new_text = sanitize_new_text(replacement.get("new_text"), speaker=target.speaker)
        old_text = target.text

        if not new_text:
            rejected.append(
                {
                    "line_number": line_number,
                    "reason": "empty_new_text",
                    "replacement": replacement,
                }
            )
            continue

        old_count = count_nonspace_chars(old_text)
        new_count = count_nonspace_chars(new_text)

        if min_accept_chars > 0 and new_count < min_accept_chars:
            rejected.append(
                {
                    "line_number": line_number,
                    "reason": "new_text_too_short",
                    "old_char_count": old_count,
                    "new_char_count": new_count,
                    "min_accept_chars": min_accept_chars,
                    "replacement": replacement,
                }
            )
            continue

        if not disable_length_guard and old_count > 0:
            too_short = old_count >= 50 and new_count < old_count * min_length_ratio
            too_long = new_count > old_count * max_length_ratio and new_count > old_count + 80

            if too_short:
                rejected.append(
                    {
                        "line_number": line_number,
                        "reason": "polish_replacement_too_short",
                        "old_char_count": old_count,
                        "new_char_count": new_count,
                        "min_length_ratio": min_length_ratio,
                        "replacement": replacement,
                    }
                )
                continue

            if too_long:
                rejected.append(
                    {
                        "line_number": line_number,
                        "reason": "polish_replacement_too_long",
                        "old_char_count": old_count,
                        "new_char_count": new_count,
                        "max_length_ratio": max_length_ratio,
                        "replacement": replacement,
                    }
                )
                continue

        target.text = new_text
        raw_lines[target.raw_index] = target.render()

        applied.append(
            {
                "line_number": line_number,
                "speaker": target.speaker,
                "old_char_count": old_count,
                "new_char_count": new_count,
                "old_text": old_text,
                "new_text": new_text,
                "applied_rule_ids": replacement.get("applied_rule_ids", []),
            }
        )

    return {
        "applied": applied,
        "rejected": rejected,
    }


# ---------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------


def default_output_path(dialogue_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = dialogue_path.stem

    if stem.startswith("polished_"):
        return output_dir / f"{stem}.md"

    return output_dir / f"polished_{stem}.md"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=Path, default=Path("configs/polisher.yaml"))
    parser.add_argument("--dialogue", type=Path, required=True)
    parser.add_argument("--validation_report", type=Path, default=None)
    parser.add_argument("--output_path", type=Path, default=None)

    parser.add_argument(
        "--polish_rules",
        type=Path,
        default=None,
        help="Optional override for polish-stage rules JSONL path.",
    )
    parser.add_argument("--max_polish_rules", type=int, default=None)
    parser.add_argument("--max_polish_rule_chars", type=int, default=None)
    parser.add_argument("--disable_polish_rules", action="store_true")

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument(
        "--target_line_numbers",
        type=str,
        default="",
        help="Optional line filter, e.g. '1,2,5-10'. If empty, all speaker lines are inspected.",
    )
    parser.add_argument("--max_line_chars_in_prompt", type=int, default=280)

    parser.add_argument("--min_accept_chars", type=int, default=8)
    parser.add_argument("--min_length_ratio", type=float, default=0.60)
    parser.add_argument("--max_length_ratio", type=float, default=1.60)
    parser.add_argument("--disable_length_guard", action="store_true")

    parser.add_argument(
        "--no_source_appendix",
        action="store_true",
        help="If set, do not preserve the original Source Appendix in the output.",
    )

    parser.add_argument("--save_prompt", action="store_true")
    parser.add_argument("--dry_run_prompt", action="store_true")

    args = parser.parse_args()

    config = load_yaml(args.config)
    polisher_cfg = config.get("polisher", config.get("rewriter", config.get("generator", {})))
    output_dir = Path(config.get("output_dir", "outputs/polishes"))
    prompt_templates = load_prompt_templates(config)

    dialogue_text = args.dialogue.read_text(encoding="utf-8")
    split = split_body_and_appendix(dialogue_text)
    body = split["body"]

    raw_lines, speaker_lines = parse_speaker_lines(body)

    if not speaker_lines:
        raise ValueError(f"No numbered A/B speaker lines found in dialogue body: {args.dialogue}")

    validation_report_raw = load_json(args.validation_report)
    validation_report = slim_validation_report(validation_report_raw)

    validation_stats = validation_report_raw.get("stats", {}) or {}
    if not isinstance(validation_stats, dict):
        validation_stats = {}

    inherited_rounds = (
        validation_stats.get("expected_rounds")
        or validation_stats.get("inferred_rounds")
    )

    inherited_total_dialogue_lines = (
        validation_stats.get("expected_total_dialogue_lines")
        or validation_stats.get("dialogue_line_count")
        or len(speaker_lines)
    )

    inherited_source_appendix_present = validation_stats.get("source_appendix_present")
    
    polish_contract = load_polish_contract(
        config,
        polish_rules_override=args.polish_rules,
        max_polish_rules_override=args.max_polish_rules,
        max_polish_rule_chars_override=args.max_polish_rule_chars,
        disable_polish_rules=args.disable_polish_rules,
    )

    target_line_numbers = parse_line_number_filter(args.target_line_numbers)
    target_lines = select_target_lines(
        speaker_lines,
        target_line_numbers=target_line_numbers,
    )

    batches = make_batches(target_lines, args.batch_size)

    output_path = args.output_path or default_output_path(args.dialogue, output_dir)

    prompt_records: List[Dict[str, Any]] = []
    batch_records: List[Dict[str, Any]] = []

    total_prompt_chars = 0

    for batch_index, batch in enumerate(batches, start=1):
        messages = build_polish_batch_messages(
            batch_index=batch_index,
            total_batches=len(batches),
            batch_lines=batch,
            all_speaker_lines=speaker_lines,
            validation_report=validation_report,
            polish_rules_text=polish_contract["rules_text"],
            prompt_templates=prompt_templates,
            max_line_chars_in_prompt=args.max_line_chars_in_prompt,
        )

        prompt_chars = sum(len(message["content"]) for message in messages)
        total_prompt_chars += prompt_chars

        prompt_records.append(
            {
                "batch_index": batch_index,
                "target_line_numbers": [line.line_number for line in batch],
                "prompt_chars": prompt_chars,
                "messages": messages,
            }
        )

    if args.save_prompt or args.dry_run_prompt:
        prompt_path = output_path.with_suffix(".prompt.json")
        prompt_payload = {
            "_meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dialogue": str(args.dialogue),
                "validation_report": str(args.validation_report) if args.validation_report else "",
                "rounds": inherited_rounds,
                "total_dialogue_lines": inherited_total_dialogue_lines,
                "input_expected_rounds": validation_stats.get("expected_rounds"),
                "input_inferred_rounds": validation_stats.get("inferred_rounds"),
                "input_expected_total_dialogue_lines": validation_stats.get("expected_total_dialogue_lines"),
                "input_dialogue_line_count": validation_stats.get("dialogue_line_count"),                
                "polish_rules_enabled": polish_contract["enabled"],
                "polish_rules_path": str(polish_contract["path"]),
                "polish_rule_count": polish_contract["count"],
                "polish_rule_ids": polish_contract["rule_ids"],
                "max_polish_rules": polish_contract["max_rules"],
                "max_polish_rule_chars": polish_contract["max_chars"],
                "target_line_count": len(target_lines),
                "target_line_numbers": [line.line_number for line in target_lines],
                "batch_size": args.batch_size,
                "batch_count": len(batches),
                "total_prompt_chars": total_prompt_chars,
                "source_appendix_present_in_original": bool(split["appendix_present"]),
                "source_appendix_preserved": bool(split["appendix_present"]) and not args.no_source_appendix,
                "prompt_template_from_config": True,
            },
            "prompts": prompt_records,
        }
        write_json(prompt_path, prompt_payload)
        print(f"Wrote polish prompt: {prompt_path}")

    if args.dry_run_prompt:
        return

    print("Polish mode: source_free_line_level")
    print(f"Dialogue: {args.dialogue}")
    if args.validation_report:
        print(f"Validation report: {args.validation_report}")
    print(f"Target lines: {len(target_lines)}")
    print(f"Batches: {len(batches)}")
    print(
        "Polish rules: "
        f"enabled={polish_contract['enabled']} | "
        f"count={polish_contract['count']} | "
        f"ids={', '.join(polish_contract['rule_ids'])}"
    )

    runtime = make_lmstudio_runtime(polisher_cfg)

    total_raw_response_chars = 0
    finish_reasons: List[Optional[str]] = []
    parse_errors: List[Dict[str, Any]] = []

    for record in prompt_records:
        batch_index = record["batch_index"]
        target_numbers = record["target_line_numbers"]
        messages = record["messages"]

        print(f"Polishing batch {batch_index}/{len(prompt_records)} | lines={target_numbers}")

        lm_result = call_lmstudio(runtime=runtime, messages=messages)
        raw_response = lm_result["content"]
        finish_reason = lm_result["finish_reason"]

        total_raw_response_chars += len(raw_response or "")
        finish_reasons.append(finish_reason)

        replacements: List[Dict[str, Any]] = []
        parse_error: Optional[Dict[str, Any]] = None

        try:
            parsed = parse_json_response(raw_response)
            replacements = normalize_replacements(parsed)
        except Exception as exc:
            parse_error = {
                "batch_index": batch_index,
                "target_line_numbers": target_numbers,
                "error": str(exc),
                "raw_response_preview": truncate_text(raw_response, 1000),
            }
            parse_errors.append(parse_error)

        apply_result = apply_replacements(
            raw_lines=raw_lines,
            speaker_lines=speaker_lines,
            replacements=replacements,
            min_accept_chars=args.min_accept_chars,
            min_length_ratio=args.min_length_ratio,
            max_length_ratio=args.max_length_ratio,
            disable_length_guard=args.disable_length_guard,
        )

        batch_records.append(
            {
                "batch_index": batch_index,
                "target_line_numbers": target_numbers,
                "finish_reason": finish_reason,
                "raw_response_chars": len(raw_response or ""),
                "replacement_count": len(replacements),
                "applied_count": len(apply_result["applied"]),
                "rejected_count": len(apply_result["rejected"]),
                "applied": apply_result["applied"],
                "rejected": apply_result["rejected"],
                "parse_error": parse_error,
            }
        )

    polished_body = render_body(raw_lines)

    if args.no_source_appendix or not split["appendix_present"]:
        final_output = polished_body.rstrip()
        source_appendix_chars = 0
    else:
        appendix = split["appendix"].strip()
        final_output = polished_body.rstrip() + "\n\n" + appendix
        source_appendix_chars = len(appendix or "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_output, encoding="utf-8")

    final_char_counts = [line.char_count for line in speaker_lines]
    short_count = sum(1 for count in final_char_counts if count < 50)
    developed_count = sum(1 for count in final_char_counts if count >= 90)

    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "polish_mode": "source_free_line_level",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dialogue": str(args.dialogue),
        "validation_report": str(args.validation_report) if args.validation_report else "",
        "output_path": str(output_path),
        "model": polisher_cfg.get("model", "qwen3-32b-mlx"),
        "rounds": inherited_rounds,
        "total_dialogue_lines": inherited_total_dialogue_lines,
        "input_expected_rounds": validation_stats.get("expected_rounds"),
        "input_inferred_rounds": validation_stats.get("inferred_rounds"),
        "input_expected_total_dialogue_lines": validation_stats.get("expected_total_dialogue_lines"),
        "input_dialogue_line_count": validation_stats.get("dialogue_line_count"),
        "input_source_appendix_present": inherited_source_appendix_present,
        "source_appendix_present_in_original": bool(split["appendix_present"]),
        "source_appendix_preserved": bool(split["appendix_present"]) and not args.no_source_appendix,
        "source_appendix_chars": source_appendix_chars,
        "polish_rules_enabled": polish_contract["enabled"],
        "polish_rules_path": str(polish_contract["path"]),
        "polish_rule_count": polish_contract["count"],
        "polish_rule_ids": polish_contract["rule_ids"],
        "max_polish_rules": polish_contract["max_rules"],
        "max_polish_rule_chars": polish_contract["max_chars"],
        "prompt_template_from_config": True,
        "target_line_count": len(target_lines),
        "target_line_numbers": [line.line_number for line in target_lines],
        "batch_size": args.batch_size,
        "batch_count": len(batch_records),
        "total_prompt_chars": total_prompt_chars,
        "total_raw_response_chars": total_raw_response_chars,
        "finish_reasons": finish_reasons,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors,
        "final_body_chars": len(polished_body or ""),
        "final_output_chars": len(final_output or ""),
        "final_avg_line_chars": round(sum(final_char_counts) / max(1, len(final_char_counts)), 2),
        "final_short_line_count_estimate": short_count,
        "final_short_line_ratio_estimate": round(short_count / max(1, len(final_char_counts)), 3),
        "final_developed_line_count_estimate": developed_count,
        "final_developed_line_ratio_estimate": round(developed_count / max(1, len(final_char_counts)), 3),
        "length_guard": {
            "min_accept_chars": args.min_accept_chars,
            "min_length_ratio": args.min_length_ratio,
            "max_length_ratio": args.max_length_ratio,
            "disable_length_guard": args.disable_length_guard,
        },
        "batch_records": batch_records,
    }

    write_json(meta_path, meta)

    print(f"Wrote polished dialogue: {output_path}")
    print(f"Wrote polish metadata: {meta_path}")
    print(f"Finish reasons: {finish_reasons}")
    print(f"Final body chars: {len(polished_body or '')}")
    print(
        "Estimated length stats: "
        f"avg_chars={meta['final_avg_line_chars']} | "
        f"short_ratio={meta['final_short_line_ratio_estimate']} | "
        f"developed_ratio={meta['final_developed_line_ratio_estimate']}"
    )


if __name__ == "__main__":
    main()