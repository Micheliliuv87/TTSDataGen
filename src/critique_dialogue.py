from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.lmstudio_utils import assert_lmstudio_model_available, make_lmstudio_client


# ---------------------------------------------------------------------
# Basic IO
# ---------------------------------------------------------------------


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


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
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_rule_jsonl(path: Path, *, missing_ok: bool = True) -> List[Dict[str, Any]]:
    if not path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(f"Critique rules file not found: {path}")

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
# Dialogue preprocessing
# ---------------------------------------------------------------------


SOURCE_APPENDIX_RE = re.compile(
    r"\n+##\s*(?:Source Appendix|Source Notes|Sources)\b",
    flags=re.IGNORECASE,
)


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


def default_output_path(dialogue_path: Path) -> Path:
    return dialogue_path.with_suffix(".critique.json")


# ---------------------------------------------------------------------
# Critique rules
# ---------------------------------------------------------------------


def _rule_priority(rule: Dict[str, Any]) -> int:
    try:
        return int(rule.get("priority", 0))
    except Exception:
        return 0


def normalize_critique_rules(
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


def critique_rule_ids(rules: List[Dict[str, Any]]) -> List[str]:
    return [
        clean_text(rule.get("rule_id", ""))
        for rule in rules
        if clean_text(rule.get("rule_id", ""))
    ]


def format_critique_rules(
    rules: List[Dict[str, Any]],
    *,
    max_chars: int,
) -> str:
    if not rules:
        return "No active critique rules were loaded."

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

    return "".join(blocks).strip() if blocks else "No active critique rules were loaded."


def load_critique_contract(
    config: Dict[str, Any],
    *,
    critique_rules_override: Optional[Path],
    max_critique_rules_override: Optional[int],
    max_critique_rule_chars_override: Optional[int],
    disable_critique_rules: bool,
) -> Dict[str, Any]:
    rules_cfg = config.get("rules", {}).get("critique", {})

    enabled = bool(rules_cfg.get("enabled", True)) and not disable_critique_rules

    rules_path = (
        critique_rules_override
        or Path(rules_cfg.get("path", "knowledge_base/rules/critique_rules.jsonl"))
    )

    max_rules = (
        max_critique_rules_override
        if max_critique_rules_override is not None
        else int(rules_cfg.get("max_rules", 12))
    )

    max_chars = (
        max_critique_rule_chars_override
        if max_critique_rule_chars_override is not None
        else int(rules_cfg.get("max_chars", 5000))
    )

    rules: List[Dict[str, Any]] = []
    rules_text = "No active critique rules were loaded."

    if enabled:
        raw_rules = load_rule_jsonl(rules_path, missing_ok=True)
        rules = normalize_critique_rules(raw_rules, max_rules=max_rules)
        rules_text = format_critique_rules(rules, max_chars=max_chars)

    return {
        "enabled": enabled,
        "path": rules_path,
        "count": len(rules),
        "max_rules": max_rules,
        "max_chars": max_chars,
        "rules_text": rules_text,
        "rule_ids": critique_rule_ids(rules),
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


def slim_source_anchor_pack(
    source_anchor_pack: Dict[str, Any],
    *,
    max_anchors: int,
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
# Prompt building
# ---------------------------------------------------------------------


CRITIQUE_SYSTEM_PROMPT = """You are a strict but practical dialogue critic for a source-grounded dialogue generation pipeline.

Your job is to evaluate a generated Chinese A/B dialogue and produce actionable rewrite guidance.

Important boundaries:
- Do not rewrite the dialogue.
- Do not add topic-specific assumptions.
- Do not reward generic fluency if the dialogue is short, repetitive, weakly grounded, or poorly developed.
- Do not require every supporting source anchor to be used. Supporting anchors are optional when weak, noisy, awkward, or off-context.
- Mechanical formatting is mostly handled by the validator. Use the validation report instead of re-checking everything manually.
- Focus on problems that a rewrite agent can fix.

Return strict JSON only. Do not wrap it in Markdown.
"""


def build_critique_prompt(
    *,
    dialogue_body: str,
    source_anchor_pack: Dict[str, Any],
    validation_report: Dict[str, Any],
    critique_rules_text: str,
    max_dialogue_chars: int,
) -> str:
    dialogue_body = truncate_text(dialogue_body, max_dialogue_chars)

    payload = {
        "task": "Critique the generated dialogue and produce structured rewrite instructions.",
        "principles": [
            "Evaluate general generation quality, not a single topic-specific case.",
            "Prefer source-grounded, content-dense, non-repetitive dialogue.",
            "Identify weak progression, thin lines, generic drift, unsupported expansion, awkward source use, and poor source coverage.",
            "Do not demand equal coverage of all supporting anchors. Supporting anchors should be omitted when they hurt relevance or naturalness.",
            "Return only JSON matching the requested schema.",
        ],
        "validation_report": validation_report,
        "source_anchor_pack": source_anchor_pack,
        "active_critique_rules": critique_rules_text,
        "dialogue_body": dialogue_body,
        "required_output_schema": {
            "critic_version": "v0",
            "ready_for_rewrite": "boolean",
            "overall_score": "integer 1-10",
            "summary": "string",
            "mechanical_status": {
                "passed": "boolean",
                "blocking_errors": ["string"],
                "notes": "string",
            },
            "quality_status": {
                "passed": "boolean",
                "needs_rewrite": "boolean",
                "main_failure_modes": ["string"],
            },
            "source_usage": {
                "status": "good | mixed | weak | problematic",
                "core_anchor_usage": "string",
                "supporting_anchor_usage": "string",
                "underused_anchor_ids": ["integer"],
                "awkward_or_forced_anchor_ids": ["integer"],
                "unsupported_or_generic_expansion": ["string"],
                "notes": "string",
            },
            "dialogue_quality": {
                "density": "good | mixed | weak",
                "progression": "good | mixed | weak",
                "repetition": "none | minor | major",
                "naturalness": "good | mixed | weak",
                "speaker_balance": "good | mixed | weak",
                "notes": "string",
            },
            "major_issues": [
                {
                    "type": "string",
                    "severity": "low | medium | high",
                    "evidence": "string",
                    "fix_instruction": "string",
                }
            ],
            "rewrite_priorities": ["string"],
            "rewrite_constraints": ["string"],
        },
    }

    return (
        "/no_think\n\n"
        "Critique the following generated dialogue for a source-grounded dialogue pipeline.\n"
        "Return strict JSON only.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def build_critique_messages(
    *,
    dialogue_body: str,
    source_anchor_pack: Dict[str, Any],
    validation_report: Dict[str, Any],
    critique_rules_text: str,
    max_dialogue_chars: int,
) -> List[Dict[str, str]]:
    user_prompt = build_critique_prompt(
        dialogue_body=dialogue_body,
        source_anchor_pack=source_anchor_pack,
        validation_report=validation_report,
        critique_rules_text=critique_rules_text,
        max_dialogue_chars=max_dialogue_chars,
    )

    return [
        {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# ---------------------------------------------------------------------
# LM Studio + JSON parsing
# ---------------------------------------------------------------------


def call_lmstudio(
    messages: List[Dict[str, str]],
    critic_cfg: Dict[str, Any],
) -> str:
    base_url = critic_cfg.get("base_url", "http://localhost:1234/v1")
    api_key = critic_cfg.get("api_key", "lm-studio")
    model = critic_cfg.get("model", "qwen3-32b-mlx")
    temperature = float(critic_cfg.get("temperature", 0.2))
    top_p = float(critic_cfg.get("top_p", 0.8))
    max_tokens = int(critic_cfg.get("max_tokens", 5000))
    timeout_seconds = int(critic_cfg.get("timeout_seconds", 1200))

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


def strip_json_fence(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    text = strip_json_fence(text)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data

    raise ValueError("Critic response did not contain a valid JSON object.")


def fallback_parse_error_report(
    *,
    raw_response: str,
    parse_error: Exception,
) -> Dict[str, Any]:
    return {
        "critic_version": "v0",
        "ready_for_rewrite": True,
        "overall_score": 0,
        "summary": "The critic model returned invalid JSON. The raw response was saved for debugging.",
        "mechanical_status": {
            "passed": None,
            "blocking_errors": [],
            "notes": "Unable to parse critic response.",
        },
        "quality_status": {
            "passed": False,
            "needs_rewrite": True,
            "main_failure_modes": ["critic_json_parse_error"],
        },
        "source_usage": {
            "status": "unknown",
            "core_anchor_usage": "",
            "supporting_anchor_usage": "",
            "underused_anchor_ids": [],
            "awkward_or_forced_anchor_ids": [],
            "unsupported_or_generic_expansion": [],
            "notes": "Unable to parse critic response.",
        },
        "dialogue_quality": {
            "density": "weak",
            "progression": "weak",
            "repetition": "minor",
            "naturalness": "mixed",
            "speaker_balance": "mixed",
            "notes": "Unable to parse critic response.",
        },
        "major_issues": [
            {
                "type": "critic_json_parse_error",
                "severity": "high",
                "evidence": str(parse_error),
                "fix_instruction": "Inspect the saved raw critic response and tighten the JSON-only prompt if needed.",
            }
        ],
        "rewrite_priorities": [
            "Do not run rewrite from this critique until the parse error is inspected.",
        ],
        "rewrite_constraints": [
            "Preserve exact round count and A/B numbering.",
        ],
        "_raw_response": raw_response,
        "_parse_error": str(parse_error),
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=Path, default=Path("configs/critic.yaml"))
    parser.add_argument("--dialogue", type=Path, required=True)
    parser.add_argument("--source_anchor_pack", type=Path, required=True)
    parser.add_argument("--validation_report", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, default=None)

    parser.add_argument(
        "--critique_rules",
        type=Path,
        default=None,
        help="Optional override for critique-stage rules JSONL path.",
    )
    parser.add_argument("--max_critique_rules", type=int, default=None)
    parser.add_argument("--max_critique_rule_chars", type=int, default=None)
    parser.add_argument("--disable_critique_rules", action="store_true")

    parser.add_argument("--max_dialogue_chars", type=int, default=16000)
    parser.add_argument("--max_anchors", type=int, default=8)
    parser.add_argument("--max_chars_per_anchor", type=int, default=900)

    parser.add_argument("--save_prompt", action="store_true")
    parser.add_argument("--dry_run_prompt", action="store_true")

    args = parser.parse_args()

    config = load_yaml(args.config)
    critic_cfg = config.get("critic", config.get("generator", {}))

    dialogue_text = args.dialogue.read_text(encoding="utf-8")
    split = split_body_and_appendix(dialogue_text)
    dialogue_body = split["body"]

    source_anchor_pack_raw = load_json(args.source_anchor_pack)
    validation_report_raw = load_json(args.validation_report)

    source_anchor_pack = slim_source_anchor_pack(
        source_anchor_pack_raw,
        max_anchors=args.max_anchors,
        max_chars_per_anchor=args.max_chars_per_anchor,
    )
    validation_report = slim_validation_report(validation_report_raw)

    critique_contract = load_critique_contract(
        config,
        critique_rules_override=args.critique_rules,
        max_critique_rules_override=args.max_critique_rules,
        max_critique_rule_chars_override=args.max_critique_rule_chars,
        disable_critique_rules=args.disable_critique_rules,
    )

    messages = build_critique_messages(
        dialogue_body=dialogue_body,
        source_anchor_pack=source_anchor_pack,
        validation_report=validation_report,
        critique_rules_text=critique_contract["rules_text"],
        max_dialogue_chars=args.max_dialogue_chars,
    )

    output_path = args.output_path or default_output_path(args.dialogue)

    prompt_chars = sum(len(message["content"]) for message in messages)

    if args.save_prompt or args.dry_run_prompt:
        prompt_path = output_path.with_suffix(".prompt.json")
        prompt_payload = {
            "_meta": {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dialogue": str(args.dialogue),
                "source_anchor_pack": str(args.source_anchor_pack),
                "validation_report": str(args.validation_report),
                "critique_rules_enabled": critique_contract["enabled"],
                "critique_rules_path": str(critique_contract["path"]),
                "critique_rule_count": critique_contract["count"],
                "critique_rule_ids": critique_contract["rule_ids"],
                "max_critique_rules": critique_contract["max_rules"],
                "max_critique_rule_chars": critique_contract["max_chars"],
                "source_anchor_count": source_anchor_pack.get("source_anchor_count"),
                "prompt_chars": prompt_chars,
            },
            "messages": messages,
        }
        write_json(prompt_path, prompt_payload)
        print(f"Wrote critique prompt: {prompt_path}")

    if args.dry_run_prompt:
        return

    print("Critique mode: source_anchor")
    print(f"Dialogue: {args.dialogue}")
    print(f"Validation report: {args.validation_report}")
    print(f"Source anchor pack: {args.source_anchor_pack}")
    print(f"Prompt chars: {prompt_chars}")
    print(
        "Critique rules: "
        f"enabled={critique_contract['enabled']} | "
        f"count={critique_contract['count']} | "
        f"ids={', '.join(critique_contract['rule_ids'])}"
    )

    raw_response = call_lmstudio(messages=messages, critic_cfg=critic_cfg)

    try:
        critique = extract_json_object(raw_response)
    except Exception as exc:
        critique = fallback_parse_error_report(
            raw_response=raw_response,
            parse_error=exc,
        )

    critique["_meta"] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dialogue": str(args.dialogue),
        "source_anchor_pack": str(args.source_anchor_pack),
        "validation_report": str(args.validation_report),
        "config": str(args.config),
        "model": critic_cfg.get("model", "qwen3-32b-mlx"),
        "prompt_chars": prompt_chars,
        "raw_response_chars": len(raw_response or ""),
        "critique_rules_enabled": critique_contract["enabled"],
        "critique_rules_path": str(critique_contract["path"]),
        "critique_rule_count": critique_contract["count"],
        "critique_rule_ids": critique_contract["rule_ids"],
        "source_anchor_count": source_anchor_pack.get("source_anchor_count"),
        "dialogue_body_chars": len(dialogue_body),
        "source_appendix_present": bool(split["appendix_present"]),
    }

    write_json(output_path, critique)

    print(f"Wrote critique: {output_path}")
    print(f"Ready for rewrite: {critique.get('ready_for_rewrite')}")
    print(f"Overall score: {critique.get('overall_score')}")


if __name__ == "__main__":
    main()