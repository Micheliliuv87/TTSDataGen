from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.lmstudio_utils import assert_lmstudio_model_available, make_lmstudio_client


# ---------------------------------------------------------------------
# Basic IO
# ---------------------------------------------------------------------


def load_rule_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Generation rules file not found: {path}")

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


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate_text(text: Any, limit: int) -> str:
    text = clean_text(text)
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 999.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


# ---------------------------------------------------------------------
# Rounds
# ---------------------------------------------------------------------


def extract_round_count(user_query: str, default_rounds: int) -> int:
    """
    Project convention:
    - Chinese "轮 / 回合" means one A+B exchange.
    - English "rounds" also means one A+B exchange.
    - English "turns" usually means individual speaker turns.
    """
    round_patterns = [
        r"(\d+)\s*轮",
        r"(\d+)\s*回合",
        r"(\d+)\s*rounds?",
    ]

    for pattern in round_patterns:
        match = re.search(pattern, user_query or "", flags=re.IGNORECASE)
        if match:
            rounds = int(match.group(1))
            return max(1, min(rounds, 80))

    turn_patterns = [
        r"(\d+)\s*turns?",
    ]

    for pattern in turn_patterns:
        match = re.search(pattern, user_query or "", flags=re.IGNORECASE)
        if match:
            turns = int(match.group(1))
            rounds = (turns + 1) // 2
            return max(1, min(rounds, 80))

    return default_rounds


# ---------------------------------------------------------------------
# Generation rules
# ---------------------------------------------------------------------


def _rule_priority(rule: Dict[str, Any]) -> int:
    try:
        return int(rule.get("priority", 0))
    except Exception:
        return 0


def normalize_generation_rules(
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


def generation_rule_ids(rules: List[Dict[str, Any]]) -> List[str]:
    return [
        clean_text(rule.get("rule_id", ""))
        for rule in rules
        if clean_text(rule.get("rule_id", ""))
    ]


def format_active_generation_rules(
    rules: List[Dict[str, Any]],
    *,
    max_chars: int,
    verbose: bool = False,
) -> str:
    if not rules:
        return "None"

    max_chars = max(0, int(max_chars or 0))
    blocks: List[str] = []
    used_chars = 0

    for rule in rules:
        rule_id = clean_text(rule.get("rule_id", "unknown_rule"))
        priority = _rule_priority(rule)
        category = clean_text(rule.get("category", "uncategorized"))
        rule_summary = clean_text(rule.get("rule", ""))
        instruction = clean_text(
            rule.get("prompt_instruction")
            or rule.get("instruction")
            or rule.get("rule")
            or ""
        )

        if not instruction:
            continue

        if verbose and rule_summary and rule_summary != instruction:
            block = (
                f"[{rule_id} | priority={priority} | category={category}]\n"
                f"Rule: {rule_summary}\n"
                f"Instruction: {instruction}"
            )
        else:
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

    return "".join(blocks).strip() if blocks else "None"


def format_generation_contract_text(rules: List[Dict[str, Any]]) -> str:
    if not rules:
        return "No active generation rules were loaded."

    ids = ", ".join(generation_rule_ids(rules))
    return (
        "The active generation rules below are binding content-generation instructions. "
        "The highest priority goal is multi-source grounded, content-dense dialogue generation. "
        "If generic prompt wording conflicts with active generation rules, the active rules win. "
        f"Loaded active rule IDs: {ids}"
    )

def build_generation_quality_contract(
    *,
    rounds: int,
    short_line_chars: int = 50,
    developed_line_chars: int = 90,
    max_short_line_ratio: float = 0.35,
    min_developed_line_ratio: float = 0.50,
) -> str:
    """
    Build dynamic, topic-independent density requirements for generation.

    This does not evaluate the output. It only gives the generator a clear,
    round-aware target that matches the validator's length-density checks.
    """
    total_lines = max(1, int(rounds)) * 2
    max_short_lines = int(total_lines * max_short_line_ratio)
    min_developed_lines = int(total_lines * min_developed_line_ratio + 0.999)

    return (
        "Dialogue density and progression requirements:\n"
        f"- Produce exactly {rounds} rounds and exactly {total_lines} numbered speaker lines.\n"
        f"- No more than {max_short_lines} of the {total_lines} speaker lines should be shorter than "
        f"{short_line_chars} non-space characters.\n"
        f"- At least {min_developed_lines} of the {total_lines} speaker lines should reach roughly "
        f"{developed_line_chars}+ non-space characters.\n"
        "- Do not fill later rounds by repeating earlier sentence frames, claims, examples, or transitions.\n"
        "- Each later round must add new substance: a concrete detail, contrast, implication, clarification, "
        "example, tension, transition, or source-grounded development.\n"
        "- Maintain natural dialogue flow while making the speaker turns content-dense."
    )

def load_generation_contract(
    config: Dict[str, Any],
    *,
    generation_rules_override: Optional[Path],
    max_generation_rules_override: Optional[int],
    max_generation_rule_chars_override: Optional[int],
    generation_rule_format_override: Optional[str],
    disable_generation_rules: bool,
) -> Dict[str, Any]:
    rules_cfg = config.get("rules", {}).get("generation", {})

    enabled = bool(rules_cfg.get("enabled", True)) and not disable_generation_rules

    rules_path = (
        generation_rules_override
        or Path(rules_cfg.get("path", "knowledge_base/rules/generation_rules.jsonl"))
    )

    max_rules = (
        max_generation_rules_override
        if max_generation_rules_override is not None
        else int(rules_cfg.get("max_rules", 10))
    )

    max_chars = (
        max_generation_rule_chars_override
        if max_generation_rule_chars_override is not None
        else int(rules_cfg.get("max_chars", 5200))
    )

    rule_format = (
        generation_rule_format_override
        or str(rules_cfg.get("format", "compact"))
    ).lower()

    rules: List[Dict[str, Any]] = []
    rules_text = "None"
    contract = "No active generation rules were loaded."

    if enabled:
        raw_rules = load_rule_jsonl(rules_path)
        rules = normalize_generation_rules(raw_rules, max_rules=max_rules)

        rules_text = format_active_generation_rules(
            rules,
            max_chars=max_chars,
            verbose=(rule_format == "verbose"),
        )
        contract = format_generation_contract_text(rules)

    return {
        "enabled": enabled,
        "path": rules_path,
        "format": rule_format,
        "count": len(rules),
        "max_rules": max_rules,
        "max_chars": max_chars,
        "contract": contract,
        "rules_text": rules_text,
        "rule_ids": generation_rule_ids(rules),
    }


# ---------------------------------------------------------------------
# Source-anchor pack
# ---------------------------------------------------------------------


def get_user_query(
    source_pack: Dict[str, Any],
    source_anchor_pack: Optional[Dict[str, Any]] = None,
) -> str:
    if source_anchor_pack:
        q = clean_text(source_anchor_pack.get("user_query", ""))
        if q:
            return q

    return clean_text(
        source_pack.get("user_query")
        or source_pack.get("query")
        or source_pack.get("original_query")
        or ""
    )


def source_sort_key(source: Dict[str, Any]) -> Tuple[int, float]:
    return safe_int(source.get("rank"), 999999), safe_float(source.get("distance"), 999.0)


def get_sources(source_pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources = source_pack.get("sources", []) or []
    if not isinstance(sources, list):
        return []
    return sorted([s for s in sources if isinstance(s, dict)], key=source_sort_key)


def fallback_anchor_pack_from_source_pack(
    source_pack: Dict[str, Any],
    *,
    max_anchors: int = 6,
    max_chars_per_anchor: int = 1400,
) -> Dict[str, Any]:
    """
    Optional raw-source fallback.

    This is intentionally NOT used by default because raw topK podcast chunks
    often contain ads, outros, platform promos, weakly related fragments, or
    transcript noise. Use only for debugging with --allow_raw_source_fallback.
    """
    sources = get_sources(source_pack)
    anchors: List[Dict[str, Any]] = []

    for source in sources[:max_anchors]:
        text = clean_text(source.get("text", ""))
        if not text:
            continue

        anchors.append(
            {
                "anchor_id": len(anchors) + 1,
                "source_rank": source.get("rank", ""),
                "source_role": "core" if len(anchors) < 2 else "supporting",
                "title": clean_text(source.get("title", "")),
                "podcast_slug": clean_text(source.get("podcast_slug", "")),
                "url": clean_text(source.get("url", "")),
                "doc_id": clean_text(source.get("doc_id", "")),
                "chunk_id": clean_text(source.get("chunk_id", "")),
                "chunk_index": source.get("chunk_index", ""),
                "retrieval_distance": source.get("distance", ""),
                "matched_queries": source.get("matched_queries", []),
                "anchor_score": "",
                "matched_terms": [],
                "strong_matched_terms": [],
                "selected_excerpt": truncate_text(text, max_chars_per_anchor),
                "why_useful": "Raw fallback source from top retrieved result. Use only for debugging.",
                "suggested_use": (
                    "Use cautiously. Raw fallback may contain ads, outros, weakly related transcript fragments, "
                    "or noisy podcast text."
                ),
            }
        )

    return {
        "pack_type": "source_anchor_pack",
        "user_query": get_user_query(source_pack),
        "query_profile": {},
        "source_anchors": anchors,
        "_meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "fallback_built_in_generate_dialogue": True,
            "selected_anchor_count": len(anchors),
            "warning": (
                "This pack was generated from raw topK source_pack entries inside generate_dialogue.py. "
                "Normal runs should use src.build_source_anchor_pack instead."
            ),
        },
    }


def validate_source_anchor_pack(
    source_anchor_pack: Dict[str, Any],
    *,
    source_anchor_pack_used: str,
    allow_empty: bool = False,
) -> None:
    anchors = source_anchor_pack.get("source_anchors", []) or []

    if not isinstance(anchors, list):
        raise ValueError(
            f"Invalid source anchor pack: source_anchors must be a list. "
            f"Path: {source_anchor_pack_used}"
        )

    usable = [
        anchor
        for anchor in anchors
        if isinstance(anchor, dict) and clean_text(anchor.get("selected_excerpt", ""))
    ]

    if not usable and not allow_empty:
        meta = source_anchor_pack.get("_meta", {}) or {}
        rejected_count = len(source_anchor_pack.get("rejected_anchor_candidates", []) or [])
        accepted_count = meta.get("accepted_candidate_count", "")
        inspected_count = meta.get("inspected_source_count", "")

        raise ValueError(
            "Source anchor pack contains zero usable source anchors.\n"
            f"Path: {source_anchor_pack_used}\n"
            f"Inspected sources: {inspected_count}\n"
            f"Accepted candidates: {accepted_count}\n"
            f"Rejected candidates: {rejected_count}\n\n"
            "This usually means build_source_anchor_pack.py filtered everything out. "
            "Inspect outputs/source_packs/latest_source_anchor_pack.json, especially "
            "`rejected_anchor_candidates`, then either loosen source_anchors thresholds in "
            "configs/generation.yaml or improve the retrieval query."
        )


def normalize_query_for_compat(text: Any) -> str:
    text = clean_text(text).lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[\"'“”‘’`，,。.!?？；;：:\-_/|()\[\]{}]+", "", text)
    return text


def validate_source_anchor_pack_compatibility(
    *,
    source_pack: Dict[str, Any],
    source_anchor_pack: Dict[str, Any],
    source_anchor_pack_used: str,
    allow_mismatch: bool = False,
) -> None:
    if allow_mismatch:
        return

    source_query = clean_text(
        source_pack.get("user_query")
        or source_pack.get("query")
        or source_pack.get("original_query")
        or ""
    )
    anchor_query = clean_text(source_anchor_pack.get("user_query", ""))

    if not source_query or not anchor_query:
        return

    if normalize_query_for_compat(source_query) != normalize_query_for_compat(anchor_query):
        raise ValueError(
            "Source anchor pack does not match the current source pack query.\n\n"
            f"Source pack query:\n  {source_query}\n\n"
            f"Source anchor pack query:\n  {anchor_query}\n\n"
            f"Source anchor pack path:\n  {source_anchor_pack_used}\n\n"
            "This usually means latest_source_anchor_pack.json is stale. "
            "Run build_source_anchor_pack.py again for the current latest_source_pack.json."
        )


def assign_anchor_rounds(anchors: List[Dict[str, Any]], *, rounds: int) -> List[Dict[str, Any]]:
    """
    Loose coverage guide.

    Important: do NOT duplicate selected_excerpt here. The source_anchors list
    already contains excerpts. Duplicating them in the plan wastes prompt space
    and may make the model treat the plan as a rigid script.
    """
    if not anchors or rounds <= 0:
        return []

    n = len(anchors)
    plan: List[Dict[str, Any]] = []

    for idx, anchor in enumerate(anchors, start=1):
        start_round = ((idx - 1) * rounds) // n + 1
        end_round = (idx * rounds) // n
        end_round = max(start_round, min(rounds, end_round))

        plan.append(
            {
                "anchor_id": anchor.get("anchor_id", idx),
                "assigned_rounds": f"Rounds {start_round}-{end_round}",
                "source_role": anchor.get("source_role", ""),
                "title": anchor.get("title", ""),
                "suggested_use": anchor.get("suggested_use", ""),
            }
        )

    return plan


def slim_source_anchor_pack(
    source_anchor_pack: Dict[str, Any],
    *,
    rounds: int,
    max_chars_per_anchor: int,
) -> Dict[str, Any]:
    anchors: List[Dict[str, Any]] = []

    for anchor in source_anchor_pack.get("source_anchors", []) or []:
        if not isinstance(anchor, dict):
            continue

        excerpt = clean_text(anchor.get("selected_excerpt", ""))
        if not excerpt:
            continue

        anchors.append(
            {
                "anchor_id": anchor.get("anchor_id", len(anchors) + 1),
                "source_rank": anchor.get("source_rank", ""),
                "source_role": anchor.get("source_role", ""),
                "title": anchor.get("title", ""),
                "matched_terms": anchor.get("matched_terms", []),
                "strong_matched_terms": anchor.get("strong_matched_terms", []),
                "support_matched_terms": anchor.get("support_matched_terms", []),
                "weak_matched_terms": anchor.get("weak_matched_terms", []),
                "topic_axis_matches": anchor.get("topic_axis_matches", []),
                "topic_axis_count": anchor.get("topic_axis_count", 0),
                "anchor_score": anchor.get("anchor_score", ""),
                "selected_excerpt": truncate_text(excerpt, max_chars_per_anchor),
                "why_useful": anchor.get("why_useful", ""),
                "suggested_use": anchor.get("suggested_use", ""),
            }
        )

    query_profile = source_anchor_pack.get("query_profile", {}) or {}

    return {
        "pack_type": "source_anchor_pack",
        "user_query": source_anchor_pack.get("user_query", ""),
        "source_anchor_count": len(anchors),
        "query_profile": {
            "topic_axes": query_profile.get("topic_axes", []),
            "core_tokens": query_profile.get("core_tokens", []),
            "strong_terms": query_profile.get("strong_terms", []),
            "support_terms": query_profile.get("support_terms", []),
        },
        "source_anchors": anchors,
        "anchor_round_plan": assign_anchor_rounds(anchors, rounds=rounds),
        "generation_note": (
            "Use source_anchors as cleaned retrieved source excerpts, not as line-by-line translation. "
            "Cover the main topic axes when possible. "
            "The anchor_round_plan is only a loose coverage guide, not a rigid outline."
        ),
    }


def get_anchor_pack_summary(source_anchor_pack: Dict[str, Any]) -> Dict[str, Any]:
    meta = source_anchor_pack.get("_meta", {}) or {}
    return {
        "pack_type": source_anchor_pack.get("pack_type", ""),
        "builder_created_at": meta.get("created_at", ""),
        "source_count": meta.get("source_count", ""),
        "inspected_source_count": meta.get("inspected_source_count", ""),
        "accepted_candidate_count": meta.get("accepted_candidate_count", ""),
        "rejected_candidate_count": meta.get("rejected_candidate_count", ""),
        "selected_anchor_count": meta.get("selected_anchor_count", ""),
        "fallback_built_in_generate_dialogue": meta.get("fallback_built_in_generate_dialogue", False),
    }


# ---------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------


def build_generation_messages_from_source_anchors(
    *,
    source_pack: Dict[str, Any],
    source_anchor_pack: Dict[str, Any],
    templates: Dict[str, Any],
    rounds: int,
    language: str,
    speaker_a: str,
    speaker_b: str,
    generation_contract: str,
    generation_rules: str,
    max_chars_per_anchor: int,
    extra_instructions: str = "",
) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    total_dialogue_lines = rounds * 2
    user_query = get_user_query(source_pack, source_anchor_pack)

    quality_contract = build_generation_quality_contract(rounds=rounds)

    base_generation_rules = clean_text(generation_rules)
    if not base_generation_rules or base_generation_rules.lower() == "none":
        effective_generation_rules = quality_contract
    else:
        effective_generation_rules = (
            base_generation_rules
            + "\n\n"
            + quality_contract
        )

    generation_templates = templates.get("source_anchor_generation", {})
    system_template = str(generation_templates.get("system", "")).strip()
    user_template = str(generation_templates.get("user_template", "")).strip()

    if not system_template or not user_template:
        raise ValueError(
            "Missing source_anchor_generation.system or source_anchor_generation.user_template "
            "in prompt templates."
        )

    slim_pack = slim_source_anchor_pack(
        source_anchor_pack,
        rounds=rounds,
        max_chars_per_anchor=max_chars_per_anchor,
    )

    user = user_template.format(
        user_query=user_query,
        rounds=rounds,
        total_dialogue_lines=total_dialogue_lines,
        language=language,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        source_anchor_pack=json.dumps(slim_pack, ensure_ascii=False, indent=2),
        generation_contract=generation_contract if generation_contract else "None",
        generation_rules=effective_generation_rules if effective_generation_rules else "None",
        extra_instructions=extra_instructions if extra_instructions else "None",
        last_a_line=total_dialogue_lines - 1,
        last_b_line=total_dialogue_lines,
    )

    return [
        {"role": "system", "content": system_template},
        {"role": "user", "content": user},
    ], slim_pack


# ---------------------------------------------------------------------
# Source appendix
# ---------------------------------------------------------------------


def strip_model_source_notes(dialogue: str) -> str:
    patterns = [
        r"\n+##\s*Source Notes\b.*$",
        r"\n+##\s*Sources\b.*$",
        r"\n+##\s*Source Appendix\b.*$",
    ]

    cleaned = dialogue.rstrip()
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL).rstrip()

    return cleaned


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
# LM Studio
# ---------------------------------------------------------------------


def call_lmstudio(
    messages: List[Dict[str, str]],
    generator_cfg: Dict[str, Any],
) -> str:
    base_url = generator_cfg.get("base_url", "http://localhost:1234/v1")
    api_key = generator_cfg.get("api_key", "lm-studio")
    model = generator_cfg.get("model", "qwen3-32b-mlx")
    temperature = float(generator_cfg.get("temperature", 0.55))
    top_p = float(generator_cfg.get("top_p", 0.9))
    max_tokens = int(generator_cfg.get("max_tokens", 12000))
    timeout_seconds = int(generator_cfg.get("timeout_seconds", 1200))

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

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content or ""


def default_output_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"dialogue_{timestamp}.md"


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=Path, default=Path("configs/generation.yaml"))
    parser.add_argument(
        "--prompt_templates",
        type=Path,
        default=None,
        help="Optional override for prompt template YAML path.",
    )
    parser.add_argument(
        "--source_pack",
        type=Path,
        default=Path("outputs/source_packs/latest_source_pack.json"),
    )
    parser.add_argument(
        "--source_anchor_pack",
        type=Path,
        default=None,
        help="Path to source anchor pack JSON. If omitted, latest_source_anchor_pack.json is used when available.",
    )
    parser.add_argument(
        "--allow_raw_source_fallback",
        action="store_true",
        help=(
            "Allow generate_dialogue.py to build a temporary raw-source fallback pack if source_anchor_pack "
            "is missing. Not recommended for normal runs because raw topK sources may include ads/outros."
        ),
    )
    parser.add_argument(
        "--allow_empty_source_anchor_pack",
        action="store_true",
        help=(
            "Allow generation even if the source anchor pack has zero usable anchors. "
            "Not recommended except for debugging prompt formatting."
        ),
    )
    parser.add_argument("--output_path", type=Path, default=None)

    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--extra_instructions", type=str, default="")
    parser.add_argument("--dry_run_prompt", action="store_true")
    parser.add_argument("--save_prompt", action="store_true")

    parser.add_argument(
        "--generation_rules",
        type=Path,
        default=None,
        help="Optional override for generation-stage rules JSONL path.",
    )
    parser.add_argument("--max_generation_rules", type=int, default=None)
    parser.add_argument("--max_generation_rule_chars", type=int, default=None)
    parser.add_argument(
        "--generation_rule_format",
        type=str,
        choices=["compact", "verbose"],
        default=None,
    )
    parser.add_argument("--disable_generation_rules", action="store_true")

    parser.add_argument(
        "--allow_mismatched_source_anchor_pack",
        action="store_true",
        help=(
            "Allow generation even if source_pack.user_query and source_anchor_pack.user_query differ. "
            "Use only for debugging."
        ),
    )
    
    args = parser.parse_args()

    config = load_yaml(args.config)
    generator_cfg = config.get("generator", {})
    dialogue_cfg = config.get("dialogue", {})
    source_anchor_cfg = config.get("source_anchors", {}) or {}

    templates_path = (
        args.prompt_templates
        or Path(config.get("prompt_templates", {}).get("path", "configs/prompt_templates.yaml"))
    )
    templates = load_yaml(templates_path)

    source_pack = load_json(args.source_pack)

    # -----------------------------------------------------------------
    # Load source anchor pack.
    #
    # Important:
    # - Normal path: run src.build_source_anchor_pack first.
    # - Raw fallback is disabled by default because it can inject ads/outros.
    # -----------------------------------------------------------------

    source_anchor_pack_path = args.source_anchor_pack
    if source_anchor_pack_path is None:
        default_anchor_path = Path(
            source_anchor_cfg.get(
                "latest_path",
                "outputs/source_packs/latest_source_anchor_pack.json",
            )
        )
        if default_anchor_path.exists():
            source_anchor_pack_path = default_anchor_path

    if source_anchor_pack_path is not None and source_anchor_pack_path.exists():
        source_anchor_pack = load_json(source_anchor_pack_path)
        source_anchor_pack_used = str(source_anchor_pack_path)
        anchor_pack_fallback = False
    else:
        if not args.allow_raw_source_fallback:
            missing_path = str(source_anchor_pack_path) if source_anchor_pack_path else (
                source_anchor_cfg.get(
                    "latest_path",
                    "outputs/source_packs/latest_source_anchor_pack.json",
                )
            )

            raise FileNotFoundError(
                "No source anchor pack found.\n\n"
                f"Expected path: {missing_path}\n\n"
                "Run this first:\n"
                "  python -m src.build_source_anchor_pack "
                "--source_pack outputs/source_packs/latest_source_pack.json\n\n"
                "Then run generation:\n"
                "  python -m src.generate_dialogue "
                "--source_pack outputs/source_packs/latest_source_pack.json "
                "--source_anchor_pack outputs/source_packs/latest_source_anchor_pack.json "
                "--save_prompt\n\n"
                "Raw fallback is disabled by default because raw topK podcast chunks may include ads, "
                "outros, platform promos, and weakly related transcript fragments. "
                "For debugging only, you can add --allow_raw_source_fallback."
            )

        source_anchor_pack = fallback_anchor_pack_from_source_pack(
            source_pack,
            max_anchors=int(source_anchor_cfg.get("max_anchors", 6)),
            max_chars_per_anchor=int(source_anchor_cfg.get("max_chars_per_anchor", 1400)),
        )
        source_anchor_pack_used = "generated_from_source_pack_in_generate_dialogue.py"
        anchor_pack_fallback = True

    validate_source_anchor_pack(
        source_anchor_pack,
        source_anchor_pack_used=source_anchor_pack_used,
        allow_empty=args.allow_empty_source_anchor_pack,
    )

    if not anchor_pack_fallback:
        validate_source_anchor_pack_compatibility(
            source_pack=source_pack,
            source_anchor_pack=source_anchor_pack,
            source_anchor_pack_used=source_anchor_pack_used,
            allow_mismatch=args.allow_mismatched_source_anchor_pack,
        )

    user_query = get_user_query(source_pack, source_anchor_pack)

    default_rounds = int(
        dialogue_cfg.get("default_rounds", dialogue_cfg.get("default_turns", 30))
    )

    if args.rounds is not None:
        rounds = args.rounds
    elif args.turns is not None:
        rounds = (args.turns + 1) // 2
    else:
        rounds = extract_round_count(user_query, default_rounds)

    rounds = max(1, min(int(rounds), 80))

    contract_info = load_generation_contract(
        config,
        generation_rules_override=args.generation_rules,
        max_generation_rules_override=args.max_generation_rules,
        max_generation_rule_chars_override=args.max_generation_rule_chars,
        generation_rule_format_override=args.generation_rule_format,
        disable_generation_rules=args.disable_generation_rules,
    )

    generation_contract = contract_info["contract"]
    generation_rules = contract_info["rules_text"]

    language = args.language or dialogue_cfg.get("language", "Chinese")
    speaker_a = dialogue_cfg.get("speaker_a", "A")
    speaker_b = dialogue_cfg.get("speaker_b", "B")

    output_dir = Path(dialogue_cfg.get("output_dir", "outputs/dialogues"))
    output_path = args.output_path or default_output_path(output_dir)

    messages, slim_pack_for_debug = build_generation_messages_from_source_anchors(
        source_pack=source_pack,
        source_anchor_pack=source_anchor_pack,
        templates=templates,
        rounds=rounds,
        language=language,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        generation_contract=generation_contract,
        generation_rules=generation_rules,
        max_chars_per_anchor=int(source_anchor_cfg.get("max_chars_per_anchor", 1600)),
        extra_instructions=args.extra_instructions,
    )
    generation_quality_contract = build_generation_quality_contract(rounds=rounds)
    anchor_pack_summary = get_anchor_pack_summary(source_anchor_pack)

    if args.save_prompt or args.dry_run_prompt:
        prompt_path = output_path.with_suffix(".prompt.json")
        prompt_payload = {
            "_meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "generation_mode": "source_anchor",
                "prompt_templates": str(templates_path),
                "source_pack": str(args.source_pack),
                "source_anchor_pack": source_anchor_pack_used,
                "anchor_pack_fallback": anchor_pack_fallback,
                "raw_source_fallback_allowed": args.allow_raw_source_fallback,
                "source_anchor_count": slim_pack_for_debug.get("source_anchor_count", None),
                "anchor_pack_summary": anchor_pack_summary,
                "generation_rules_enabled": contract_info["enabled"],
                "generation_rules_path": str(contract_info["path"]),
                "generation_rule_format": contract_info["format"],
                "generation_rule_count": contract_info["count"],
                "generation_rule_ids": contract_info["rule_ids"],
                "max_generation_rules": contract_info["max_rules"],
                "max_generation_rule_chars": contract_info["max_chars"],
                "generation_quality_contract_chars": len(generation_quality_contract),
                "generation_quality_contract": generation_quality_contract,
            },
            "messages": messages,
        }
        write_json(prompt_path, prompt_payload)
        print(f"Wrote prompt: {prompt_path}")

    if args.dry_run_prompt:
        return

    prompt_chars = sum(len(message["content"]) for message in messages)
    print("Generation mode: source_anchor")
    print(f"Prompt chars: {prompt_chars}")
    print(f"Requested rounds: {rounds}")
    print(f"Expected dialogue lines: {rounds * 2}")
    print(f"Source anchor pack: {source_anchor_pack_used}")
    print(f"Source anchors: {slim_pack_for_debug.get('source_anchor_count')}")
    print(f"Anchor pack fallback: {anchor_pack_fallback}")
    if anchor_pack_fallback:
        print("WARNING: raw source fallback is enabled. This is not recommended for normal generation.")
    print(
        "Generation rules: "
        f"enabled={contract_info['enabled']} | "
        f"count={contract_info['count']} | "
        f"ids={', '.join(contract_info['rule_ids'])}"
    )

    raw_dialogue = call_lmstudio(
        messages=messages,
        generator_cfg=generator_cfg,
    )

    raw_dialogue_chars = len(raw_dialogue or "")
    dialogue = strip_model_source_notes(raw_dialogue)
    dialogue_after_strip_chars = len(dialogue or "")

    appendix_excerpt_chars = int(dialogue_cfg.get("source_appendix_excerpt_chars", 1200))
    source_appendix = format_source_anchor_appendix(
        source_anchor_pack,
        excerpt_chars=appendix_excerpt_chars,
    )

    final_output = dialogue.rstrip() + source_appendix
    
    source_appendix_chars = len(source_appendix or "")
    final_output_chars = len(final_output or "")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_output, encoding="utf-8")

    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "generation_mode": "source_anchor",
        "source_pack": str(args.source_pack),
        "source_anchor_pack": source_anchor_pack_used,
        "anchor_pack_fallback": anchor_pack_fallback,
        "raw_source_fallback_allowed": args.allow_raw_source_fallback,
        "output_path": str(output_path),
        "model": generator_cfg.get("model", "qwen3-32b-mlx"),
        "rounds": rounds,
        "total_dialogue_lines": rounds * 2,
        "language": language,
        "coverage": source_pack.get("coverage", {}),
        "user_query": user_query,
        "source_anchor_count": slim_pack_for_debug.get("source_anchor_count", None),
        "anchor_pack_summary": anchor_pack_summary,
        "generation_rules_enabled": contract_info["enabled"],
        "generation_rules_path": str(contract_info["path"]),
        "generation_rule_format": contract_info["format"],
        "generation_rule_count": contract_info["count"],
        "max_generation_rules": contract_info["max_rules"],
        "max_generation_rule_chars": contract_info["max_chars"],
        "generation_rule_ids": contract_info["rule_ids"],
        "generation_rules_text_chars": len(generation_rules),
        "generation_quality_contract_chars": len(generation_quality_contract),
        "prompt_chars": prompt_chars,
        "raw_dialogue_chars": raw_dialogue_chars,
        "dialogue_after_strip_chars": dialogue_after_strip_chars,
        "source_appendix_chars": source_appendix_chars,
        "final_output_chars": final_output_chars,
    }

    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote dialogue: {output_path}")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()