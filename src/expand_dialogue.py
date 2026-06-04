from __future__ import annotations

import argparse
import json
import math
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


def load_json(path: Path) -> Dict[str, Any]:
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
        raise FileNotFoundError(f"Expansion rules file not found: {path}")

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
# Markdown splitting / parsing
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


def strip_source_appendix_from_text(text: str) -> str:
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
        return "No active expansion rules were loaded."

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

    return "".join(blocks).strip() if blocks else "No active expansion rules were loaded."


def load_expand_contract(
    config: Dict[str, Any],
    *,
    expand_rules_override: Optional[Path],
    max_expand_rules_override: Optional[int],
    max_expand_rule_chars_override: Optional[int],
    disable_expand_rules: bool,
) -> Dict[str, Any]:
    rules_cfg = config.get("rules", {}).get("expand", {})

    enabled = bool(rules_cfg.get("enabled", True)) and not disable_expand_rules

    rules_path = (
        expand_rules_override
        or Path(rules_cfg.get("path", "knowledge_base/rules/expand_rules.jsonl"))
    )

    max_rules = (
        max_expand_rules_override
        if max_expand_rules_override is not None
        else int(rules_cfg.get("max_rules", 10))
    )

    max_chars = (
        max_expand_rule_chars_override
        if max_expand_rule_chars_override is not None
        else int(rules_cfg.get("max_chars", 5000))
    )

    rules: List[Dict[str, Any]] = []
    rules_text = "No active expansion rules were loaded."

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


DEFAULT_EXPAND_SYSTEM_PROMPT = """You are a source-aware expansion agent for a Chinese A/B dialogue pipeline.

Your job is not to freely rewrite the whole dialogue.
Your job is to expand selected numbered speaker lines while preserving the existing dialogue structure.

Important boundaries:
- Return strict JSON only.
- Replace only the requested numbered speaker lines.
- Preserve the same line numbers and speakers.
- Do not add, remove, reorder, merge, or split dialogue lines.
- Do not write Markdown, Source Appendix, source notes, citations, URLs, retrieval notes, or explanations.
- Use source anchors as private grounding material.
- Do not mention sources, anchors, retrieval, excerpts, metadata, transcript labels, URLs, or appendices inside speaker lines.
- Expansion should add concrete substance, not filler.
- This stage focuses on content expansion, source grounding, and major content repair.
- Fine-grained style rules belong to a later polish stage, unless they are included in the active expansion rules.
"""

DEFAULT_EXPAND_USER_PREAMBLE = """/no_think

Expand the selected speaker lines for a source-grounded Chinese A/B dialogue.
Return strict JSON only.
"""


def load_prompt_templates(config: Dict[str, Any]) -> Dict[str, str]:
    prompt_cfg = config.get("prompt_template", {}) or {}

    system_prompt = str(
        prompt_cfg.get("system_prompt")
        or prompt_cfg.get("system")
        or DEFAULT_EXPAND_SYSTEM_PROMPT
    ).strip()

    user_preamble = str(
        prompt_cfg.get("user_preamble")
        or prompt_cfg.get("user_prompt_preamble")
        or DEFAULT_EXPAND_USER_PREAMBLE
    ).strip()

    return {
        "system_prompt": system_prompt,
        "user_preamble": user_preamble,
    }


# ---------------------------------------------------------------------
# Input slimming
# ---------------------------------------------------------------------


def slim_validation_report(validation_report: Dict[str, Any]) -> Dict[str, Any]:
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


def slim_critique_report(critique_report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "critic_version": critique_report.get("critic_version"),
        "ready_for_rewrite": critique_report.get("ready_for_rewrite"),
        "overall_score": critique_report.get("overall_score"),
        "summary": critique_report.get("summary"),
        "mechanical_status": critique_report.get("mechanical_status", {}),
        "quality_status": critique_report.get("quality_status", {}),
        "source_usage": critique_report.get("source_usage", {}),
        "dialogue_quality": critique_report.get("dialogue_quality", {}),
        "major_issues": critique_report.get("major_issues", [])[:12],
        "rewrite_priorities": critique_report.get("rewrite_priorities", [])[:12],
        "rewrite_constraints": critique_report.get("rewrite_constraints", [])[:12],
    }


def get_critic_flagged_anchor_ids(critique_report: Dict[str, Any]) -> List[int]:
    """
    Return source anchor IDs that the critic explicitly marked as awkward,
    forced, noisy, or off-context.

    These anchors are excluded from the expander prompt so expansion does not
    amplify bad source material.
    """
    source_usage = critique_report.get("source_usage", {}) or {}
    raw_ids = source_usage.get("awkward_or_forced_anchor_ids", []) or []

    flagged: List[int] = []
    for item in raw_ids:
        try:
            flagged.append(int(item))
        except Exception:
            continue

    return sorted(set(flagged))


def _collect_strings(obj: Any) -> List[str]:
    values: List[str] = []

    if isinstance(obj, str):
        values.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            values.extend(_collect_strings(value))
    elif isinstance(obj, list):
        for value in obj:
            values.extend(_collect_strings(value))

    return values


def extract_critic_problem_terms(critique_report: Dict[str, Any]) -> List[str]:
    """
    Extract concrete quoted/problem terms from the critic report.

    This is intentionally generic. It does not hardcode any topic.
    """
    source_usage = critique_report.get("source_usage", {}) or {}
    major_issues = critique_report.get("major_issues", []) or []

    relevant_texts: List[str] = []
    relevant_texts.extend(_collect_strings(source_usage.get("unsupported_or_generic_expansion", [])))
    relevant_texts.extend(_collect_strings(source_usage.get("notes", "")))
    relevant_texts.extend(_collect_strings(major_issues))

    terms: List[str] = []
    quote_patterns = [
        r"‘([^’]{2,40})’",
        r"“([^”]{2,40})”",
        r"'([^']{2,40})'",
        r'"([^"]{2,40})"',
    ]

    for text in relevant_texts:
        text = clean_text(text)
        if not text:
            continue

        for pattern in quote_patterns:
            for match in re.finditer(pattern, text):
                term = clean_text(match.group(1))
                if term:
                    terms.append(term)

    blocked = {
        "source",
        "anchor",
        "anchors",
        "retrieval",
        "metadata",
        "锚点",
        "来源",
        "检索",
        "材料",
        "文本",
        "对话",
    }

    cleaned_terms: List[str] = []
    for term in terms:
        if term.lower() in blocked:
            continue
        if len(term) < 2:
            continue
        cleaned_terms.append(term)

    return sorted(set(cleaned_terms))


def find_critic_repair_line_numbers(
    speaker_lines: List[SpeakerLine],
    *,
    problem_terms: List[str],
    neighbor_rounds: int = 1,
) -> List[int]:
    """
    Find dialogue lines that appear to contain critic-flagged problematic content.

    If one line is directly matched, include the whole current round and a small
    neighboring window. This catches follow-up lines that continue the same bad
    material even if they do not repeat the exact quoted phrase.
    """
    if not problem_terms:
        return []

    direct_matches: List[int] = []

    for line in speaker_lines:
        text = clean_text(line.text)
        if any(term and term in text for term in problem_terms):
            direct_matches.append(line.line_number)

    if not direct_matches:
        return []

    affected: set[int] = set()
    existing_line_numbers = {line.line_number for line in speaker_lines}

    for line_number in direct_matches:
        round_start = ((line_number - 1) // 2) * 2 + 1

        start = round_start - (neighbor_rounds * 2)
        end = round_start + 1 + (neighbor_rounds * 2)

        for candidate in range(start, end + 1):
            if candidate in existing_line_numbers:
                affected.add(candidate)

    return sorted(affected)


def make_repair_hint_text(
    *,
    flagged_anchor_ids: List[int],
    problem_terms: List[str],
) -> str:
    parts: List[str] = []

    if flagged_anchor_ids:
        parts.append(
            "The critic marked some source anchors as awkward, forced, noisy, or off-context. "
            f"Do not use or preserve content derived from anchor IDs: {flagged_anchor_ids}."
        )

    if problem_terms:
        parts.append(
            "The critic also mentioned problematic phrase(s): "
            + ", ".join(problem_terms)
            + ". If a target line contains or continues this material, replace it rather than expanding it."
        )

    parts.append(
        "For repair-mode lines, preserve the local dialogue function, but do not preserve awkward/off-context content."
    )

    return " ".join(parts)


def slim_source_anchor_pack(
    source_anchor_pack: Dict[str, Any],
    *,
    max_anchors: int,
    max_chars_per_anchor: int,
    exclude_anchor_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    anchors: List[Dict[str, Any]] = []
    excluded_ids = set(int(x) for x in (exclude_anchor_ids or []))

    for anchor in source_anchor_pack.get("source_anchors", []) or []:
        if not isinstance(anchor, dict):
            continue

        try:
            anchor_id = int(anchor.get("anchor_id", len(anchors) + 1))
        except Exception:
            anchor_id = len(anchors) + 1

        if anchor_id in excluded_ids:
            continue

        excerpt = clean_text(anchor.get("selected_excerpt", ""))
        if not excerpt:
            continue

        anchors.append(
            {
                "anchor_id": anchor_id,
                "source_role": anchor.get("source_role", ""),
                "title": anchor.get("title", ""),
                "matched_terms": anchor.get("matched_terms", []),
                "strong_matched_terms": anchor.get("strong_matched_terms", []),
                "topic_axis_matches": anchor.get("topic_axis_matches", []),
                "topic_axis_count": anchor.get("topic_axis_count", 0),
                "anchor_score": anchor.get("anchor_score", ""),
                "why_useful": anchor.get("why_useful", ""),
                "suggested_use": anchor.get("suggested_use", ""),
                "selected_excerpt": truncate_text(excerpt, max_chars_per_anchor),
            }
        )

        if max_anchors > 0 and len(anchors) >= max_anchors:
            break

    query_profile = source_anchor_pack.get("query_profile", {}) or {}
    meta = source_anchor_pack.get("_meta", {}) or {}

    return {
        "pack_type": source_anchor_pack.get("pack_type", ""),
        "user_query": source_anchor_pack.get("user_query", ""),
        "query_profile": {
            "topic_axes": query_profile.get("topic_axes", []),
            "core_tokens": query_profile.get("core_tokens", []),
            "strong_terms": query_profile.get("strong_terms", []),
            "support_terms": query_profile.get("support_terms", []),
        },
        "source_anchor_count": len(anchors),
        "excluded_critic_flagged_anchor_ids": sorted(excluded_ids),
        "source_anchors": anchors,
        "_meta": {
            "source_count": meta.get("source_count", ""),
            "inspected_source_count": meta.get("inspected_source_count", ""),
            "accepted_candidate_count": meta.get("accepted_candidate_count", ""),
            "rejected_candidate_count": meta.get("rejected_candidate_count", ""),
            "selected_anchor_count": meta.get("selected_anchor_count", ""),
        },
    }


# ---------------------------------------------------------------------
# Expansion targeting
# ---------------------------------------------------------------------


def _find_warning(validation_report: Dict[str, Any], warning_type: str) -> Dict[str, Any]:
    for item in validation_report.get("warnings", []) or []:
        if isinstance(item, dict) and item.get("type") == warning_type:
            return item
    return {}


def derive_length_targets(
    validation_report: Dict[str, Any],
    *,
    fallback_developed_line_threshold: int,
    fallback_min_developed_line_ratio: float,
) -> Dict[str, Any]:
    stats = validation_report.get("stats", {}) or {}
    length = stats.get("length", {}) or {}

    line_count = int(length.get("line_count") or stats.get("expected_total_dialogue_lines") or 60)
    developed_line_threshold = int(length.get("developed_line_threshold") or fallback_developed_line_threshold)

    developed_warning = _find_warning(validation_report, "low_developed_line_ratio")
    min_developed_line_ratio = float(
        developed_warning.get("min_developed_line_ratio", fallback_min_developed_line_ratio)
    )

    min_developed_lines = int(math.ceil(line_count * min_developed_line_ratio))

    return {
        "line_count": line_count,
        "developed_line_threshold": developed_line_threshold,
        "min_developed_line_ratio": min_developed_line_ratio,
        "min_developed_lines": min_developed_lines,
    }


def choose_target_lines(
    speaker_lines: List[SpeakerLine],
    *,
    developed_line_threshold: int,
    min_developed_lines: int,
    target_mode: str,
    target_buffer: int,
    max_target_lines: int,
    min_target_lines: int,
) -> List[SpeakerLine]:
    current_developed = [
        line for line in speaker_lines
        if line.char_count >= developed_line_threshold
    ]

    candidates = [
        line for line in speaker_lines
        if line.char_count < developed_line_threshold
    ]

    if not candidates:
        return []

    if target_mode == "all_under_threshold":
        selected = candidates
    else:
        needed = max(0, min_developed_lines - len(current_developed))
        target_n = needed + max(0, target_buffer)

        if min_target_lines > 0:
            target_n = max(target_n, min_target_lines)

        if target_n <= 0:
            return []

        shortest = sorted(candidates, key=lambda line: (line.char_count, line.line_number))
        selected = shortest[:target_n]
        selected = sorted(selected, key=lambda line: line.line_number)

    if max_target_lines > 0:
        selected = selected[:max_target_lines]

    return selected


def make_batches(items: List[SpeakerLine], batch_size: int) -> List[List[SpeakerLine]]:
    batch_size = max(1, int(batch_size or 1))
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def format_dialogue_snapshot(speaker_lines: List[SpeakerLine], *, max_line_chars: int = 180) -> List[Dict[str, Any]]:
    return [
        {
            "line_number": line.line_number,
            "speaker": line.speaker,
            "char_count": line.char_count,
            "text": truncate_text(line.text, max_line_chars),
        }
        for line in speaker_lines
    ]


# ---------------------------------------------------------------------
# Source appendix
# ---------------------------------------------------------------------


def format_source_anchor_appendix(
    source_anchor_pack: Dict[str, Any],
    *,
    excerpt_chars: int = 1200,
) -> str:
    anchors = source_anchor_pack.get("source_anchors", []) or []

    lines: List[str] = []
    lines.append("\n\n## Source Appendix")
    lines.append("")
    lines.append(
        "This section is generated deterministically from the source anchor pack, not written by the language model."
    )
    lines.append("")

    if not anchors:
        lines.append("No source anchors were available.")
        return "\n".join(lines)

    for anchor in anchors:
        if not isinstance(anchor, dict):
            continue

        text = clean_text(anchor.get("selected_excerpt", ""))
        if len(text) > excerpt_chars:
            text = text[:excerpt_chars].rstrip() + "..."

        lines.append(f"### Anchor {anchor.get('anchor_id', '')}. {anchor.get('title', '')}")
        lines.append("")
        lines.append(f"- Role: `{anchor.get('source_role', '')}`")
        lines.append(f"- Source rank: `{anchor.get('source_rank', '')}`")
        lines.append(f"- Podcast: `{anchor.get('podcast_slug', '')}`")
        if anchor.get("url"):
            lines.append(f"- URL: {anchor.get('url')}")
        if anchor.get("matched_terms"):
            lines.append(f"- Matched terms: {', '.join(str(x) for x in anchor.get('matched_terms', []))}")
        if anchor.get("strong_matched_terms"):
            lines.append(
                f"- Strong matched terms: {', '.join(str(x) for x in anchor.get('strong_matched_terms', []))}"
            )
        if anchor.get("anchor_score") not in {"", None}:
            lines.append(f"- Anchor score: `{anchor.get('anchor_score')}`")
        if anchor.get("why_useful"):
            lines.append(f"- Why useful: {anchor.get('why_useful')}")
        lines.append("")
        lines.append("> " + text.replace("\n", "\n> "))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------


def build_expand_batch_messages(
    *,
    batch_index: int,
    total_batches: int,
    batch_lines: List[SpeakerLine],
    all_speaker_lines: List[SpeakerLine],
    source_anchor_pack: Dict[str, Any],
    validation_report: Dict[str, Any],
    critique_report: Dict[str, Any],
    expand_rules_text: str,
    developed_line_threshold: int,
    target_min_chars: int,
    target_max_chars: int,
    critic_repair_line_numbers: List[int],
    critic_repair_hint: str,
    prompt_templates: Dict[str, str],
) -> List[Dict[str, str]]:
    repair_line_set = set(critic_repair_line_numbers or [])

    batch_payload = []

    for line in batch_lines:
        repair_mode = "replace_critic_flagged_material" if line.line_number in repair_line_set else "expand_content"

        item = {
            "line_number": line.line_number,
            "speaker": line.speaker,
            "current_char_count": line.char_count,
            "target_nonspace_chars": f"{target_min_chars}-{target_max_chars}",
            "repair_mode": repair_mode,
            "current_text": line.text,
        }

        if repair_mode == "replace_critic_flagged_material":
            item["critic_repair_hint"] = critic_repair_hint

        batch_payload.append(item)

    payload = {
        "task": "Expand only the selected numbered speaker lines.",
        "batch": {
            "batch_index": batch_index,
            "total_batches": total_batches,
        },
        "expansion_goal": {
            "developed_line_threshold": developed_line_threshold,
            "target_nonspace_chars_per_replacement": f"{target_min_chars}-{target_max_chars}",
            "purpose": (
                "Turn thin or medium-length speaker lines into developed, source-grounded dialogue turns "
                "without changing line numbers, speakers, or dialogue structure."
            ),
        },
        "hard_constraints": [
            "Return strict JSON only.",
            "Return an object with key 'replacements'.",
            "Each replacement must include line_number, speaker, and new_text.",
            "Do not include numbering or speaker prefix inside new_text.",
            "Do not add, remove, reorder, merge, or split lines.",
            "Do not mention source, anchor, retrieval, transcript, metadata, URL, appendix, or citation.",
            "If a target line has repair_mode='replace_critic_flagged_material', do not preserve the awkward or off-context content.",
            "For repair-mode lines, preserve the local dialogue function while replacing the problematic material with relevant source-grounded content from non-excluded anchors.",
            "Do not introduce unrelated facts, unsupported examples, or topic-specific assumptions beyond the user request and source anchors.",
        ],
        "expansion_instructions": [
            "For normal expand_content lines, preserve the original meaning and local dialogue role while making the line more concrete and content-dense.",
            "For replace_critic_flagged_material lines, do not preserve awkward/off-context content. Preserve only the local dialogue function and replace the bad material with relevant source-grounded content.",
            "Add source-grounded details, sensory specificity, contrast, implication, clarification, or transition.",
            "Repair major critique issues when they appear in a target line.",
            "Do not use excluded critic-flagged anchors as expansion material.",
            "Avoid merely appending a generic phrase to the old line.",
            "Do not turn the dialogue into a lecture. Keep it conversational.",
        ],
        "critic_repair_policy": {
            "critic_repair_line_numbers": critic_repair_line_numbers,
            "critic_repair_hint": critic_repair_hint,
            "excluded_critic_flagged_anchor_ids": source_anchor_pack.get("excluded_critic_flagged_anchor_ids", []),
        },
        "active_expand_rules": expand_rules_text,
        "validation_report": validation_report,
        "critique_report": critique_report,
        "source_anchor_pack": source_anchor_pack,
        "dialogue_snapshot": format_dialogue_snapshot(all_speaker_lines),
        "target_lines": batch_payload,
        "required_output_schema": {
            "replacements": [
                {
                    "line_number": "integer",
                    "speaker": "A or B",
                    "new_text": "expanded speaker text only, without numbering or speaker prefix",
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
    text = strip_source_appendix_from_text(str(text or ""))
    text = strip_markdown_fence(text)
    text = clean_text(text)

    text = re.sub(r"^\s*\d+\.\s*[AB]\s*:\s*", "", text)
    text = re.sub(rf"^\s*{re.escape(speaker)}\s*:\s*", "", text)
    text = re.sub(r"^\s*[AB]\s*:\s*", "", text)

    return clean_text(text)


# ---------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------


def make_lmstudio_runtime(expander_cfg: Dict[str, Any]) -> Dict[str, Any]:
    base_url = expander_cfg.get("base_url", "http://localhost:1234/v1")
    api_key = expander_cfg.get("api_key", "lm-studio")
    model = expander_cfg.get("model", "qwen3-32b-mlx")
    timeout_seconds = int(expander_cfg.get("timeout_seconds", 1200))

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
        "temperature": float(expander_cfg.get("temperature", 0.35)),
        "top_p": float(expander_cfg.get("top_p", 0.9)),
        "max_tokens": int(expander_cfg.get("max_tokens", 8000)),
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
    reject_shorter: bool,
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

        if reject_shorter and new_count < old_count:
            rejected.append(
                {
                    "line_number": line_number,
                    "reason": "new_text_shorter_than_old_text",
                    "old_char_count": old_count,
                    "new_char_count": new_count,
                    "replacement": replacement,
                }
            )
            continue

        if min_accept_chars > 0 and new_count < min_accept_chars and new_count <= old_count + 8:
            rejected.append(
                {
                    "line_number": line_number,
                    "reason": "new_text_too_short_and_not_substantially_longer",
                    "old_char_count": old_count,
                    "new_char_count": new_count,
                    "min_accept_chars": min_accept_chars,
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

    if stem.startswith("expanded_"):
        return output_dir / f"{stem}.md"

    return output_dir / f"expanded_{stem}.md"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=Path, default=Path("configs/expander.yaml"))
    parser.add_argument("--dialogue", type=Path, required=True)
    parser.add_argument("--source_anchor_pack", type=Path, required=True)
    parser.add_argument("--validation_report", type=Path, required=True)
    parser.add_argument("--critique_report", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, default=None)

    parser.add_argument(
        "--expand_rules",
        type=Path,
        default=None,
        help="Optional override for expansion-stage rules JSONL path.",
    )
    parser.add_argument("--max_expand_rules", type=int, default=None)
    parser.add_argument("--max_expand_rule_chars", type=int, default=None)
    parser.add_argument("--disable_expand_rules", action="store_true")

    parser.add_argument(
        "--target_mode",
        choices=["needed", "all_under_threshold"],
        default="needed",
        help="needed expands enough low-developed lines to try to pass the validator; all_under_threshold expands every line below the developed threshold.",
    )
    parser.add_argument("--target_buffer", type=int, default=8)
    parser.add_argument("--max_target_lines", type=int, default=0)
    parser.add_argument("--min_target_lines", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--fallback_developed_line_threshold", type=int, default=90)
    parser.add_argument("--fallback_min_developed_line_ratio", type=float, default=0.50)
    parser.add_argument("--target_min_chars", type=int, default=95)
    parser.add_argument("--target_max_chars", type=int, default=135)
    parser.add_argument("--min_accept_chars", type=int, default=80)
    parser.add_argument("--reject_shorter", action="store_true")

    parser.add_argument("--max_anchors", type=int, default=8)
    parser.add_argument("--max_chars_per_anchor", type=int, default=1400)
    parser.add_argument("--source_appendix_excerpt_chars", type=int, default=1200)
    parser.add_argument("--no_source_appendix", action="store_true")

    parser.add_argument("--save_prompt", action="store_true")
    parser.add_argument("--dry_run_prompt", action="store_true")

    args = parser.parse_args()

    config = load_yaml(args.config)
    expander_cfg = config.get("expander", config.get("rewriter", config.get("generator", {})))
    output_dir = Path(config.get("output_dir", "outputs/expansions"))
    prompt_templates = load_prompt_templates(config)

    dialogue_text = args.dialogue.read_text(encoding="utf-8")
    split = split_body_and_appendix(dialogue_text)
    body = split["body"]

    raw_lines, speaker_lines = parse_speaker_lines(body)

    if not speaker_lines:
        raise ValueError(f"No numbered A/B speaker lines found in dialogue body: {args.dialogue}")

    source_anchor_pack_raw = load_json(args.source_anchor_pack)
    validation_report_raw = load_json(args.validation_report)
    critique_report_raw = load_json(args.critique_report)

    flagged_anchor_ids = get_critic_flagged_anchor_ids(critique_report_raw)
    critic_problem_terms = extract_critic_problem_terms(critique_report_raw)

    source_anchor_pack = slim_source_anchor_pack(
        source_anchor_pack_raw,
        max_anchors=args.max_anchors,
        max_chars_per_anchor=args.max_chars_per_anchor,
        exclude_anchor_ids=flagged_anchor_ids,
    )
    validation_report = slim_validation_report(validation_report_raw)
    critique_report = slim_critique_report(critique_report_raw)

    critic_repair_line_numbers = find_critic_repair_line_numbers(
        speaker_lines,
        problem_terms=critic_problem_terms,
        neighbor_rounds=1,
    )

    critic_repair_hint = make_repair_hint_text(
        flagged_anchor_ids=flagged_anchor_ids,
        problem_terms=critic_problem_terms,
    )

    expand_contract = load_expand_contract(
        config,
        expand_rules_override=args.expand_rules,
        max_expand_rules_override=args.max_expand_rules,
        max_expand_rule_chars_override=args.max_expand_rule_chars,
        disable_expand_rules=args.disable_expand_rules,
    )

    length_targets = derive_length_targets(
        validation_report,
        fallback_developed_line_threshold=args.fallback_developed_line_threshold,
        fallback_min_developed_line_ratio=args.fallback_min_developed_line_ratio,
    )

    developed_line_threshold = int(length_targets["developed_line_threshold"])
    min_developed_lines = int(length_targets["min_developed_lines"])

    target_lines = choose_target_lines(
        speaker_lines,
        developed_line_threshold=developed_line_threshold,
        min_developed_lines=min_developed_lines,
        target_mode=args.target_mode,
        target_buffer=args.target_buffer,
        max_target_lines=args.max_target_lines,
        min_target_lines=args.min_target_lines,
    )

    # Always include critic-repair lines, even if they were not selected by length.
    if critic_repair_line_numbers:
        by_number = {line.line_number: line for line in target_lines}

        for line in speaker_lines:
            if line.line_number in critic_repair_line_numbers:
                by_number[line.line_number] = line

        target_lines = sorted(by_number.values(), key=lambda line: line.line_number)

    batches = make_batches(target_lines, args.batch_size)

    output_path = args.output_path or default_output_path(args.dialogue, output_dir)

    prompt_records: List[Dict[str, Any]] = []
    batch_records: List[Dict[str, Any]] = []

    total_prompt_chars = 0

    for batch_index, batch in enumerate(batches, start=1):
        messages = build_expand_batch_messages(
            batch_index=batch_index,
            total_batches=len(batches),
            batch_lines=batch,
            all_speaker_lines=speaker_lines,
            source_anchor_pack=source_anchor_pack,
            validation_report=validation_report,
            critique_report=critique_report,
            expand_rules_text=expand_contract["rules_text"],
            developed_line_threshold=developed_line_threshold,
            target_min_chars=args.target_min_chars,
            target_max_chars=args.target_max_chars,
            critic_repair_line_numbers=critic_repair_line_numbers,
            critic_repair_hint=critic_repair_hint,
            prompt_templates=prompt_templates,
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
                "source_anchor_pack": str(args.source_anchor_pack),
                "validation_report": str(args.validation_report),
                "critique_report": str(args.critique_report),
                "expand_rules_enabled": expand_contract["enabled"],
                "expand_rules_path": str(expand_contract["path"]),
                "expand_rule_count": expand_contract["count"],
                "expand_rule_ids": expand_contract["rule_ids"],
                "max_expand_rules": expand_contract["max_rules"],
                "max_expand_rule_chars": expand_contract["max_chars"],
                "source_anchor_count": source_anchor_pack.get("source_anchor_count"),
                "critic_flagged_anchor_ids": flagged_anchor_ids,
                "critic_problem_terms": critic_problem_terms,
                "critic_repair_line_numbers": critic_repair_line_numbers,
                "prompt_template_from_config": True,
                "target_mode": args.target_mode,
                "target_line_count": len(target_lines),
                "batch_size": args.batch_size,
                "batch_count": len(batches),
                "total_prompt_chars": total_prompt_chars,
                "source_appendix_enabled": not args.no_source_appendix,
            },
            "prompts": prompt_records,
        }
        write_json(prompt_path, prompt_payload)
        print(f"Wrote expansion prompt: {prompt_path}")

    if args.dry_run_prompt:
        return

    print("Expansion mode: source_anchor_line_level")
    print(f"Dialogue: {args.dialogue}")
    print(f"Validation report: {args.validation_report}")
    print(f"Critique report: {args.critique_report}")
    print(f"Source anchor pack: {args.source_anchor_pack}")
    print(f"Target mode: {args.target_mode}")
    print(f"Target lines: {len(target_lines)}")
    print(f"Batches: {len(batches)}")
    print(f"Developed threshold: {developed_line_threshold}")
    print(f"Min developed lines target: {min_developed_lines}")
    print(f"Critic flagged anchor IDs: {flagged_anchor_ids}")
    print(f"Critic problem terms: {critic_problem_terms}")
    print(f"Critic repair lines: {critic_repair_line_numbers}")
    print(
        "Expand rules: "
        f"enabled={expand_contract['enabled']} | "
        f"count={expand_contract['count']} | "
        f"ids={', '.join(expand_contract['rule_ids'])}"
    )

    runtime = make_lmstudio_runtime(expander_cfg)

    total_raw_response_chars = 0
    finish_reasons: List[Optional[str]] = []
    parse_errors: List[Dict[str, Any]] = []

    for record in prompt_records:
        batch_index = record["batch_index"]
        target_line_numbers = record["target_line_numbers"]
        messages = record["messages"]

        print(f"Expanding batch {batch_index}/{len(prompt_records)} | lines={target_line_numbers}")

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
                "target_line_numbers": target_line_numbers,
                "error": str(exc),
                "raw_response_preview": truncate_text(raw_response, 1000),
            }
            parse_errors.append(parse_error)

        apply_result = apply_replacements(
            raw_lines=raw_lines,
            speaker_lines=speaker_lines,
            replacements=replacements,
            min_accept_chars=args.min_accept_chars,
            reject_shorter=args.reject_shorter,
        )

        batch_records.append(
            {
                "batch_index": batch_index,
                "target_line_numbers": target_line_numbers,
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

    expanded_body = render_body(raw_lines)

    if args.no_source_appendix:
        final_output = expanded_body.rstrip()
        source_appendix_chars = 0
    else:
        source_appendix = format_source_anchor_appendix(
            source_anchor_pack_raw,
            excerpt_chars=args.source_appendix_excerpt_chars,
        )
        final_output = expanded_body.rstrip() + source_appendix
        source_appendix_chars = len(source_appendix or "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_output, encoding="utf-8")

    stats = validation_report.get("stats", {}) or {}
    expected_rounds = stats.get("inferred_rounds") or stats.get("expected_rounds")
    expected_total_lines = stats.get("expected_total_dialogue_lines") or len(speaker_lines)

    final_char_counts = [line.char_count for line in speaker_lines]
    final_developed_count = sum(1 for count in final_char_counts if count >= developed_line_threshold)
    final_short_count = sum(1 for count in final_char_counts if count < 50)

    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "expand_mode": "source_anchor_line_level",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dialogue": str(args.dialogue),
        "source_anchor_pack": str(args.source_anchor_pack),
        "validation_report": str(args.validation_report),
        "critique_report": str(args.critique_report),
        "output_path": str(output_path),
        "model": expander_cfg.get("model", "qwen3-32b-mlx"),
        "rounds": expected_rounds,
        "total_dialogue_lines": expected_total_lines,
        "source_appendix_enabled": not args.no_source_appendix,
        "source_appendix_chars": source_appendix_chars,
        "source_appendix_present_in_original": bool(split["appendix_present"]),
        "source_anchor_count": source_anchor_pack.get("source_anchor_count"),
        "critic_flagged_anchor_ids": flagged_anchor_ids,
        "critic_problem_terms": critic_problem_terms,
        "critic_repair_line_numbers": critic_repair_line_numbers,
        "expand_rules_enabled": expand_contract["enabled"],
        "expand_rules_path": str(expand_contract["path"]),
        "expand_rule_count": expand_contract["count"],
        "expand_rule_ids": expand_contract["rule_ids"],
        "max_expand_rules": expand_contract["max_rules"],
        "max_expand_rule_chars": expand_contract["max_chars"],
        "prompt_template_from_config": True,
        "target_mode": args.target_mode,
        "target_buffer": args.target_buffer,
        "target_line_count": len(target_lines),
        "target_line_numbers": [line.line_number for line in target_lines],
        "batch_size": args.batch_size,
        "batch_count": len(batch_records),
        "total_prompt_chars": total_prompt_chars,
        "total_raw_response_chars": total_raw_response_chars,
        "finish_reasons": finish_reasons,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors,
        "developed_line_threshold": developed_line_threshold,
        "min_developed_lines_target": min_developed_lines,
        "target_min_chars": args.target_min_chars,
        "target_max_chars": args.target_max_chars,
        "min_accept_chars": args.min_accept_chars,
        "final_body_chars": len(expanded_body or ""),
        "final_output_chars": len(final_output or ""),
        "final_avg_line_chars": round(sum(final_char_counts) / max(1, len(final_char_counts)), 2),
        "final_developed_line_count_estimate": final_developed_count,
        "final_developed_line_ratio_estimate": round(final_developed_count / max(1, len(final_char_counts)), 3),
        "final_short_line_count_estimate": final_short_count,
        "final_short_line_ratio_estimate": round(final_short_count / max(1, len(final_char_counts)), 3),
        "batch_records": batch_records,
    }

    write_json(meta_path, meta)

    print(f"Wrote expanded dialogue: {output_path}")
    print(f"Wrote expansion metadata: {meta_path}")
    print(f"Finish reasons: {finish_reasons}")
    print(f"Final body chars: {len(expanded_body or '')}")
    print(
        "Estimated length stats: "
        f"avg_chars={meta['final_avg_line_chars']} | "
        f"developed_ratio={meta['final_developed_line_ratio_estimate']} | "
        f"short_ratio={meta['final_short_line_ratio_estimate']}"
    )


if __name__ == "__main__":
    main()