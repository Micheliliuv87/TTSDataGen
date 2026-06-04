# src/rule_pack.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


Rule = Dict[str, Any]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def read_jsonl(path: Path) -> List[Rule]:
    """
    Read a JSONL file where each non-empty line is one JSON object.

    Missing files return [] so the pipeline can still run without rules.
    """
    if not path.exists():
        return []

    rules: List[Rule] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL in {path} at line {line_no}: {exc}"
                ) from exc

            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid rule in {path} at line {line_no}: expected JSON object."
                )

            item.setdefault("_source_path", str(path))
            item.setdefault("_line_no", line_no)
            rules.append(item)

    return rules


def normalize_rule(rule: Rule) -> Rule:
    """
    Normalize optional fields so downstream formatters remain stable.

    Recommended fields:
    - rule_id
    - status
    - priority
    - category
    - rule
    - prompt_instruction
    - generation_instruction
    - critic_instruction
    - revision_instruction
    - validator_check
    """
    normalized = dict(rule)

    normalized.setdefault("rule_id", "")
    normalized.setdefault("status", "active")
    normalized.setdefault("priority", 0)
    normalized.setdefault("category", "general")

    # Human-facing / debug fields.
    normalized.setdefault("rule", "")
    normalized.setdefault("description", "")

    # Prompt-facing fields.
    normalized.setdefault("prompt_instruction", "")
    normalized.setdefault("generation_instruction", "")
    normalized.setdefault("critic_instruction", "")
    normalized.setdefault("revision_instruction", "")
    normalized.setdefault("validator_check", "")

    # Optional future fields.
    normalized.setdefault("always_include", False)
    normalized.setdefault("trigger_terms", [])

    normalized["priority"] = _safe_int(normalized.get("priority"), 0)

    return normalized


def load_rules(
    path: Path,
    *,
    active_only: bool = True,
    max_rules: Optional[int] = None,
) -> List[Rule]:
    """
    Load rules from a JSONL file.

    Default behavior:
    - Only status == active rules are returned.
    - Rules are sorted by priority descending.
    """
    rules = [normalize_rule(rule) for rule in read_jsonl(path)]

    if active_only:
        rules = [
            rule
            for rule in rules
            if _safe_str(rule.get("status", "active")).lower() == "active"
        ]

    rules.sort(
        key=lambda rule: (
            _safe_int(rule.get("priority"), 0),
            _safe_str(rule.get("rule_id")),
        ),
        reverse=True,
    )

    if max_rules is not None:
        rules = rules[:max_rules]

    return rules


def select_rules(
    rules: Iterable[Rule],
    *,
    categories: Optional[List[str]] = None,
    max_rules: Optional[int] = None,
) -> List[Rule]:
    """
    Lightweight selector.

    For now:
    - filter by category if provided
    - sort by priority
    - keep top N

    Later this can become semantic Control RAG.
    """
    selected = [normalize_rule(rule) for rule in rules]

    if categories:
        allowed = {category.lower() for category in categories}
        selected = [
            rule
            for rule in selected
            if _safe_str(rule.get("category")).lower() in allowed
        ]

    selected.sort(
        key=lambda rule: (
            _safe_int(rule.get("priority"), 0),
            _safe_str(rule.get("rule_id")),
        ),
        reverse=True,
    )

    if max_rules is not None:
        selected = selected[:max_rules]

    return selected


def _truncate_block(text: str, max_chars: Optional[int]) -> str:
    if max_chars is None or max_chars <= 0:
        return text.strip()

    text = text.strip()
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars].rstrip()
    last_newline = truncated.rfind("\n")
    if last_newline > max_chars * 0.65:
        truncated = truncated[:last_newline].rstrip()

    return truncated + "\n- Additional lower-priority rules were omitted due to the rule budget."


def _format_rule_header(index: int, rule: Rule) -> str:
    rule_id = _safe_str(rule.get("rule_id")) or f"rule_{index}"
    category = _safe_str(rule.get("category")) or "general"
    priority = _safe_int(rule.get("priority"), 0)
    return f"{index}. [{rule_id}] category={category}; priority={priority}"


def format_generation_contract(
    rules: List[Rule],
    *,
    max_rules: Optional[int] = 6,
    max_chars: Optional[int] = 1200,
) -> str:
    """
    Compact production formatter for generation prompts.

    This is the preferred formatter for generate_dialogue.py.

    It intentionally avoids:
    - rule_id
    - category
    - priority
    - verbose debug text

    It uses the shortest prompt-facing instruction available:
    prompt_instruction > generation_instruction > rule
    """
    selected = select_rules(rules, max_rules=max_rules)

    instructions: List[str] = []

    for rule in selected:
        rule = normalize_rule(rule)

        instruction = (
            _safe_str(rule.get("prompt_instruction"))
            or _safe_str(rule.get("generation_instruction"))
            or _safe_str(rule.get("rule"))
        )

        if not instruction:
            continue

        if not instruction.endswith((".", "。", "!", "！", "?", "？")):
            instruction += "."

        instructions.append(f"- {instruction}")

    if not instructions:
        return ""

    block = "\n".join(instructions)
    return _truncate_block(block, max_chars=max_chars)


def format_generation_rules(
    rules: List[Rule],
    *,
    max_rules: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> str:
    """
    Verbose debug formatter for generation rules.

    Useful for checking which rules are selected, but do not use this
    as the default production prompt format once the rule set grows.
    """
    selected = select_rules(rules, max_rules=max_rules)

    lines: List[str] = []

    for index, rule in enumerate(selected, start=1):
        rule = normalize_rule(rule)

        base_rule = _safe_str(rule.get("rule"))
        prompt_instruction = _safe_str(rule.get("prompt_instruction"))
        generation_instruction = _safe_str(rule.get("generation_instruction"))

        if not base_rule and not prompt_instruction and not generation_instruction:
            continue

        lines.append(_format_rule_header(index, rule))

        if base_rule:
            lines.append(f"- Rule: {base_rule}")

        if prompt_instruction:
            lines.append(f"- Prompt instruction: {prompt_instruction}")

        if generation_instruction:
            lines.append(f"- Apply during generation: {generation_instruction}")

        lines.append("")

    return _truncate_block("\n".join(lines), max_chars=max_chars)


def format_critique_rules(
    rules: List[Rule],
    *,
    max_rules: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> str:
    selected = select_rules(rules, max_rules=max_rules)

    lines: List[str] = []

    for index, rule in enumerate(selected, start=1):
        rule = normalize_rule(rule)

        base_rule = _safe_str(rule.get("rule"))
        critic_instruction = _safe_str(rule.get("critic_instruction"))

        if not base_rule and not critic_instruction:
            continue

        lines.append(_format_rule_header(index, rule))

        if base_rule:
            lines.append(f"- Rule: {base_rule}")

        if critic_instruction:
            lines.append(f"- Critic should check: {critic_instruction}")

        lines.append("")

    return _truncate_block("\n".join(lines), max_chars=max_chars)


def format_revision_rules(
    rules: List[Rule],
    *,
    max_rules: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> str:
    selected = select_rules(rules, max_rules=max_rules)

    lines: List[str] = []

    for index, rule in enumerate(selected, start=1):
        rule = normalize_rule(rule)

        base_rule = _safe_str(rule.get("rule"))
        revision_instruction = _safe_str(rule.get("revision_instruction"))

        if not base_rule and not revision_instruction:
            continue

        lines.append(_format_rule_header(index, rule))

        if base_rule:
            lines.append(f"- Rule: {base_rule}")

        if revision_instruction:
            lines.append(f"- Revise by: {revision_instruction}")

        lines.append("")

    return _truncate_block("\n".join(lines), max_chars=max_chars)


def format_validation_rules(
    rules: List[Rule],
    *,
    max_rules: Optional[int] = None,
    max_chars: Optional[int] = None,
) -> str:
    selected = select_rules(rules, max_rules=max_rules)

    lines: List[str] = []

    for index, rule in enumerate(selected, start=1):
        rule = normalize_rule(rule)

        description = _safe_str(rule.get("description"))
        validator_check = _safe_str(rule.get("validator_check"))
        base_rule = _safe_str(rule.get("rule"))

        if not description and not validator_check and not base_rule:
            continue

        lines.append(_format_rule_header(index, rule))

        if base_rule:
            lines.append(f"- Rule: {base_rule}")

        if description:
            lines.append(f"- Description: {description}")

        if validator_check:
            lines.append(f"- Validator check: {validator_check}")

        lines.append("")

    return _truncate_block("\n".join(lines), max_chars=max_chars)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rules",
        type=Path,
        required=True,
        help="Path to a JSONL rule file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="generation",
        choices=[
            "generation",
            "generation_verbose",
            "critique",
            "revision",
            "validation",
        ],
    )
    parser.add_argument("--max_rules", type=int, default=None)
    parser.add_argument("--max_chars", type=int, default=None)
    parser.add_argument("--include_inactive", action="store_true")
    args = parser.parse_args()

    rules = load_rules(
        args.rules,
        active_only=not args.include_inactive,
        max_rules=args.max_rules,
    )

    if args.mode == "generation":
        print(
            format_generation_contract(
                rules,
                max_rules=args.max_rules,
                max_chars=args.max_chars,
            )
        )
    elif args.mode == "generation_verbose":
        print(
            format_generation_rules(
                rules,
                max_rules=args.max_rules,
                max_chars=args.max_chars,
            )
        )
    elif args.mode == "critique":
        print(
            format_critique_rules(
                rules,
                max_rules=args.max_rules,
                max_chars=args.max_chars,
            )
        )
    elif args.mode == "revision":
        print(
            format_revision_rules(
                rules,
                max_rules=args.max_rules,
                max_chars=args.max_chars,
            )
        )
    elif args.mode == "validation":
        print(
            format_validation_rules(
                rules,
                max_rules=args.max_rules,
                max_chars=args.max_chars,
            )
        )


if __name__ == "__main__":
    main()