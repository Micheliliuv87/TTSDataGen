# src/generate_dialogue.py

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.lmstudio_utils import assert_lmstudio_model_available, make_lmstudio_client


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def extract_round_count(user_query: str, default_rounds: int) -> int:
    """
    Project convention:
    - Chinese "轮 / 回合" means one A+B exchange.
      Example: 30轮 = 30 rounds = 60 dialogue lines.
    - English "rounds" also means one A+B exchange.
    - English "turns" usually means individual speaker turns.
      Example: 30 turns = 15 rounds = 30 dialogue lines.
    """
    round_patterns = [
        r"(\d+)\s*轮",
        r"(\d+)\s*回合",
        r"(\d+)\s*rounds?",
    ]

    for pattern in round_patterns:
        match = re.search(pattern, user_query, flags=re.IGNORECASE)
        if match:
            rounds = int(match.group(1))
            return max(1, min(rounds, 80))

    turn_patterns = [
        r"(\d+)\s*turns?",
    ]

    for pattern in turn_patterns:
        match = re.search(pattern, user_query, flags=re.IGNORECASE)
        if match:
            turns = int(match.group(1))
            rounds = (turns + 1) // 2
            return max(1, min(rounds, 80))

    return default_rounds


def build_source_block(
    sources: List[Dict[str, Any]],
    max_source_chars: int,
    max_sources: int,
    max_chars_per_source: int,
) -> str:
    if not sources:
        return "No retrieved sources were provided."

    ranked_sources = sorted(
        sources,
        key=lambda s: float(s.get("distance", 999.0)),
    )[:max_sources]

    blocks: List[str] = []
    used_chars = 0

    for source in ranked_sources:
        text = clean_text(source.get("text", ""))
        if not text:
            continue

        remaining = max_source_chars - used_chars
        if remaining <= 0:
            break

        text_limit = min(max_chars_per_source, remaining)
        text = text[:text_limit].rstrip()
        used_chars += len(text)

        block = (
            f"[S{source.get('rank')}]\n"
            f"title: {source.get('title', '')}\n"
            f"distance: {source.get('distance', '')}\n"
            f"text: {text}"
        )
        blocks.append(block)

    return "\n\n".join(blocks)


def format_source_appendix(
    source_pack: Dict[str, Any],
    excerpt_chars: int = 1200,
) -> str:
    sources = source_pack.get("sources", [])
    if not sources:
        return "\n\n## Source Appendix\n\nNo retrieved sources were available.\n"

    lines: List[str] = []
    lines.append("\n\n## Source Appendix")
    lines.append("")
    lines.append(
        "This section is generated deterministically from the retrieved source pack, not written by the language model."
    )
    lines.append("")

    for source in sources:
        rank = source.get("rank", "")
        title = source.get("title", "")
        podcast_slug = source.get("podcast_slug", "")
        url = source.get("url", "")
        start_timestamp = source.get("start_timestamp", "")
        end_timestamp = source.get("end_timestamp", "")
        chunk_id = source.get("chunk_id", "")
        doc_id = source.get("doc_id", "")
        chunk_index = source.get("chunk_index", "")
        distance = source.get("distance", "")
        matched_queries = source.get("matched_queries", [])
        text = clean_text(source.get("text", ""))

        if len(text) > excerpt_chars:
            text = text[:excerpt_chars].rstrip() + "..."

        lines.append(f"### S{rank}. {title}")
        lines.append("")
        lines.append(f"- Podcast: `{podcast_slug}`")
        lines.append(f"- URL: {url if url else 'N/A'}")
        lines.append(f"- Timestamp: {start_timestamp} - {end_timestamp}")
        lines.append(f"- Chunk ID: `{chunk_id}`")
        lines.append(f"- Doc ID: `{doc_id}`")
        lines.append(f"- Chunk index: `{chunk_index}`")
        lines.append(f"- Retrieval distance: `{distance}`")

        if matched_queries:
            lines.append("- Matched queries:")
            for query in matched_queries:
                lines.append(f"  - {query}")

        lines.append("")
        lines.append("Retrieved excerpt:")
        lines.append("")
        lines.append("> " + text.replace("\n", "\n> "))
        lines.append("")

    return "\n".join(lines)


def strip_model_source_notes(dialogue: str) -> str:
    """
    Remove model-generated source notes so the final source appendix
    is controlled by Python and cannot hallucinate URLs or source IDs.
    """
    patterns = [
        r"\n+##\s*Source Notes\b.*$",
        r"\n+##\s*Sources\b.*$",
        r"\n+##\s*Source Appendix\b.*$",
    ]

    cleaned = dialogue.rstrip()
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL).rstrip()

    return cleaned

def build_generation_messages(
    source_pack: Dict[str, Any],
    rounds: int,
    language: str,
    speaker_a: str,
    speaker_b: str,
    max_source_chars: int,
    max_sources: int,
    max_chars_per_source: int,
    include_source_notes: bool,
    extra_instructions: str = "",
) -> List[Dict[str, str]]:
    total_dialogue_lines = rounds * 2

    user_query = source_pack.get("user_query", "")

    query_rewrite = source_pack.get("query_rewrite", {})
    if not isinstance(query_rewrite, dict):
        query_rewrite = {}

    query_rewrite_brief = {
        "core_query": query_rewrite.get("core_query", ""),
        "canonical_terms": query_rewrite.get("canonical_terms", []),
    }

    coverage = source_pack.get("coverage", {})
    sources = source_pack.get("sources", [])

    source_block = build_source_block(
        sources=sources,
        max_source_chars=max_source_chars,
        max_sources=max_sources,
        max_chars_per_source=max_chars_per_source,
    )

    source_note_instruction = (
        "Do not write Source Notes, Sources, citations, URLs, or a source appendix. "
        "Python will append the exact source appendix after generation."
    )

    system = (
        "You are a dialogue script generator for a local RAG system.\n"
        "You write natural two-speaker dialogue scripts based on retrieved podcast transcript sources.\n"
        "Use the retrieved sources as grounding and inspiration, but do not copy long passages.\n"
        "Do not invent fake citations, fake episode details, fake source titles, or unsupported factual claims.\n"
        "You may add connective language, interpretation, and conversational flow, but concrete examples must be grounded in the source pack.\n"
        "If coverage is weak, avoid overclaiming source support.\n"
        "Follow the user's requested topic, language, tone, and constraints.\n"
        "Return polished Markdown only."
    )

    user = f"""
/no_think

User request:
{user_query}

Query rewrite metadata:
{json.dumps(query_rewrite_brief, ensure_ascii=False, indent=2)}

Coverage metadata:
{json.dumps(coverage, ensure_ascii=False, indent=2)}

Task:
Create a {rounds}-round dialogue between Speaker {speaker_a} and Speaker {speaker_b}.

Important definition:
One round means Speaker {speaker_a} speaks once and Speaker {speaker_b} replies once.
Therefore, {rounds} rounds must contain exactly {total_dialogue_lines} numbered dialogue lines.
Do not treat one numbered line as one full round.

Language:
{language}

Dialogue requirements:
1. Use exactly {rounds} rounds.
2. Use exactly {total_dialogue_lines} numbered dialogue lines total.
3. Alternate speakers strictly: {speaker_a}, {speaker_b}, {speaker_a}, {speaker_b}, ...
4. Every odd-numbered line must be Speaker {speaker_a}; every even-numbered line must be Speaker {speaker_b}.
5. Organize the dialogue by round labels: Round 1, Round 2, ... Round {rounds}.
6. Make the dialogue natural, modern, and listenable.
7. Preserve the user's constraints. For example, if the user asks "not too religious", use religious source material as background but avoid a sermon-like tone.
8. Prefer strong sources over weak or loosely related sources.
9. If sources are weak or only loosely related, use them carefully and do not force them.
10. Avoid long direct quotations.
11. Stay grounded in the retrieved source pack. Do not add unsupported specific facts.
12. Before finalizing, silently verify that the output has exactly {rounds} rounds and exactly {total_dialogue_lines} numbered dialogue lines. Do not include the verification.

Output format:

# Title

## Dialogue

Round 1

1. {speaker_a}: ...
2. {speaker_b}: ...

Round 2

3. {speaker_a}: ...
4. {speaker_b}: ...

Continue until:

Round {rounds}

{total_dialogue_lines - 1}. {speaker_a}: ...
{total_dialogue_lines}. {speaker_b}: ...

{source_note_instruction}

Extra instructions:
{extra_instructions if extra_instructions else "None"}

Retrieved source pack:
{source_block}
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def call_lmstudio(
    messages: List[Dict[str, str]],
    generator_cfg: Dict[str, Any],
) -> str:
    base_url = generator_cfg.get("base_url", "http://localhost:1234/v1")
    api_key = generator_cfg.get("api_key", "lm-studio")
    model = generator_cfg.get("model", "qwen3-32b-mlx")
    temperature = float(generator_cfg.get("temperature", 0.7))
    top_p = float(generator_cfg.get("top_p", 0.9))
    max_tokens = int(generator_cfg.get("max_tokens", 8000))
    timeout_seconds = int(generator_cfg.get("timeout_seconds", 900))

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/generation.yaml"))
    parser.add_argument(
        "--source_pack",
        type=Path,
        default=Path("outputs/source_packs/latest_source_pack.json"),
    )
    parser.add_argument("--output_path", type=Path, default=None)

    # Preferred new argument.
    parser.add_argument("--rounds", type=int, default=None)

    # Backward-compatible old argument.
    # In this project, --turns is treated as rounds for compatibility with existing run_pipeline.py.
    parser.add_argument("--turns", type=int, default=None)

    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--extra_instructions", type=str, default="")
    parser.add_argument("--dry_run_prompt", action="store_true")
    parser.add_argument("--save_prompt", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    generator_cfg = config.get("generator", {})
    dialogue_cfg = config.get("dialogue", {})

    source_pack = load_json(args.source_pack)

    user_query = source_pack.get("user_query", "")

    default_rounds = int(
        dialogue_cfg.get("default_rounds", dialogue_cfg.get("default_turns", 30))
    )

    rounds = (
        args.rounds
        or args.turns
        or extract_round_count(user_query, default_rounds)
    )
    rounds = max(1, min(int(rounds), 80))

    language = args.language or dialogue_cfg.get("language", "Chinese")
    speaker_a = dialogue_cfg.get("speaker_a", "A")
    speaker_b = dialogue_cfg.get("speaker_b", "B")

    max_source_chars = int(dialogue_cfg.get("max_source_chars", 18000))
    max_sources = int(dialogue_cfg.get("max_sources", 6))
    max_chars_per_source = int(dialogue_cfg.get("max_chars_per_source", 3000))
    include_source_notes = bool(dialogue_cfg.get("include_source_notes", True))

    output_dir = Path(dialogue_cfg.get("output_dir", "outputs/dialogues"))
    output_path = args.output_path or default_output_path(output_dir)

    messages = build_generation_messages(
        source_pack=source_pack,
        rounds=rounds,
        language=language,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        max_source_chars=max_source_chars,
        max_sources=max_sources,
        max_chars_per_source=max_chars_per_source,
        include_source_notes=include_source_notes,
        extra_instructions=args.extra_instructions,
    )

    if args.save_prompt or args.dry_run_prompt:
        prompt_path = output_path.with_suffix(".prompt.json")
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(
            json.dumps(messages, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote prompt: {prompt_path}")

    if args.dry_run_prompt:
        return

    prompt_chars = sum(len(message["content"]) for message in messages)
    print(f"Prompt chars: {prompt_chars}")
    print(f"Requested rounds: {rounds}")
    print(f"Expected dialogue lines: {rounds * 2}")

    dialogue = call_lmstudio(
        messages=messages,
        generator_cfg=generator_cfg,
    )

    dialogue = strip_model_source_notes(dialogue)

    source_appendix = format_source_appendix(
        source_pack=source_pack,
        excerpt_chars=int(dialogue_cfg.get("source_appendix_excerpt_chars", 1200)),
    )

    final_output = dialogue.rstrip() + source_appendix

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_output, encoding="utf-8")

    meta_path = output_path.with_suffix(".meta.json")
    meta = {
        "source_pack": str(args.source_pack),
        "output_path": str(output_path),
        "model": generator_cfg.get("model", "qwen3-32b-mlx"),
        "rounds": rounds,
        "total_dialogue_lines": rounds * 2,
        "language": language,
        "coverage": source_pack.get("coverage", {}),
        "user_query": user_query,
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote dialogue: {output_path}")
    print(f"Wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()