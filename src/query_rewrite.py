# src/query_rewrite.py

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from src.lmstudio_utils import assert_lmstudio_model_available, make_lmstudio_client


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

SANITIZER_VERSION = "topic_consistency_v1"

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'’]*|[\u4e00-\u9fff]+")
JOINER_RE = re.compile(r"\b(?:and|or|vs|versus)\b|与|和|以及|对比", re.IGNORECASE)

GENERIC_QUERY_TOKENS: Set[str] = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "about",
    "from",
    "into",
    "by",
    "as",
    "and",
    "or",
    "vs",
    "versus",
    "topic",
    "topics",
    "theme",
    "themes",
    "concept",
    "concepts",
    "idea",
    "ideas",
    "story",
    "stories",
    "narrative",
    "narratives",
    "book",
    "books",
    "event",
    "events",
    "history",
    "historical",
    "modern",
    "framing",
    "context",
    "contexts",
    "discussion",
    "discussions",
    "podcast",
    "podcasts",
    "transcript",
    "transcripts",
    "interview",
    "interviews",
    "dialogue",
    "conversation",
    "conversations",
    "analysis",
    "content",
    "generation",
    "writing",
    "script",
    "scripts",
    "language",
    "languages",
}

GENERATION_INSTRUCTION_PATTERNS = [
    r"\bwrite\b",
    r"\bgenerate\b",
    r"\bcreate\b",
    r"\bdialogue\b",
    r"\bscript\b",
    r"\btraining\s+text\b",
    r"\b\d+\s*(rounds?|turns?|lines?)\b",
    r"生成",
    r"写",
    r"对话",
    r"语音训练",
    r"\d+\s*轮",
    r"a\s*与\s*b",
    r"a\s*和\s*b",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip()


def dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        item = normalize_query(str(item))
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def build_canonical_query(canonical_terms: List[str]) -> Optional[str]:
    terms = dedupe_keep_order([str(t) for t in canonical_terms if str(t).strip()])

    if not terms:
        return None

    return " ".join(terms)


def _normalize_token(token: str) -> str:
    token = token.lower().strip(" '’.-_")
    token = token.replace("’", "'")

    if len(token) > 4 and token.endswith("s"):
        token = token[:-1]

    return token


def _meaningful_tokens(text: str) -> List[str]:
    tokens: List[str] = []

    for raw in TOKEN_RE.findall(text or ""):
        token = _normalize_token(raw)

        if not token:
            continue
        if token in GENERIC_QUERY_TOKENS:
            continue
        if len(token) <= 1:
            continue

        tokens.append(token)

    return dedupe_keep_order(tokens)


def _token_overlap_count(a: List[str], b: List[str]) -> int:
    return len(set(a) & set(b))


def _text_supports_tokens(text: str, tokens: List[str]) -> bool:
    if not tokens:
        return False

    text_tokens = _meaningful_tokens(text)
    if not text_tokens:
        return False

    overlap = _token_overlap_count(tokens, text_tokens)

    if len(tokens) == 1:
        return overlap >= 1

    return overlap >= min(2, len(tokens))


def _support_count(tokens: List[str], texts: List[str]) -> int:
    return sum(1 for text in texts if _text_supports_tokens(text, tokens))


def _token_document_frequency(texts: List[str]) -> Dict[str, int]:
    freq: Dict[str, int] = {}

    for text in texts:
        for token in set(_meaningful_tokens(text)):
            freq[token] = freq.get(token, 0) + 1

    return freq


def _looks_like_generation_instruction(text: str) -> bool:
    normalized = normalize_query(text).lower()
    return any(re.search(pattern, normalized, flags=re.IGNORECASE) for pattern in GENERATION_INSTRUCTION_PATTERNS)


def _contains_removed_term(text: str, removed_terms: List[str]) -> Optional[str]:
    text_norm = normalize_query(text).lower()

    for term in removed_terms:
        term_norm = normalize_query(term).lower()
        if term_norm and term_norm in text_norm:
            return term

    return None


def _has_unanchored_joined_part(query: str, anchor_tokens: List[str]) -> bool:
    """
    Detect a joined retrieval query that contains one anchored part and one
    unanchored substantive part.

    This is intentionally generic. It does not know any domain-specific topics.
    """
    if not anchor_tokens:
        return False

    parts = [p.strip() for p in JOINER_RE.split(query or "") if p.strip()]
    if len(parts) <= 1:
        return False

    anchored_parts = 0
    unanchored_substantive_parts = 0

    for part in parts:
        part_tokens = _meaningful_tokens(part)
        if not part_tokens:
            continue

        overlap = _token_overlap_count(part_tokens, anchor_tokens)

        if overlap > 0:
            anchored_parts += 1
        elif len(part_tokens) >= 1:
            unanchored_substantive_parts += 1

    return anchored_parts >= 1 and unanchored_substantive_parts >= 1


def _choose_anchor_tokens(
    canonical_terms: List[str],
    core_query: str,
    retrieval_queries: List[str],
) -> List[str]:
    """
    Build a generic topic anchor from the model's own internally consistent output.

    Priority:
    1. core_query tokens
    2. tokens repeated across rewritten retrieval queries
    3. first canonical term tokens as a last resort
    """
    support_texts = []
    if core_query:
        support_texts.append(core_query)
    support_texts.extend(retrieval_queries)

    core_tokens = _meaningful_tokens(core_query)
    freq = _token_document_frequency(support_texts)
    repeated_tokens = [token for token, count in freq.items() if count >= 2]

    anchor_tokens = dedupe_keep_order(core_tokens + repeated_tokens)

    if not anchor_tokens and canonical_terms:
        anchor_tokens = _meaningful_tokens(canonical_terms[0])

    return anchor_tokens


def sanitize_rewrite_result(parsed: Dict[str, Any], original_query: str) -> Dict[str, Any]:
    """
    Generic post-filter for query rewrite drift.

    The goal is not to make the rewrite perfect. The goal is to prevent a local
    rewrite model from expanding a specific request into loosely related side
    topics before retrieval.

    This sanitizer is intentionally topic-agnostic:
    - no hand-written topic blacklist
    - no domain-specific examples
    - no special cases for one test query
    """
    if not isinstance(parsed, dict):
        return parsed

    raw_terms = parsed.get("canonical_terms", [])
    if not isinstance(raw_terms, list):
        raw_terms = []

    canonical_terms = [str(t).strip() for t in raw_terms if str(t).strip()]

    core_query = parsed.get("core_query", "")
    core_query = core_query.strip() if isinstance(core_query, str) else ""

    raw_queries = parsed.get("retrieval_queries", [])
    if not isinstance(raw_queries, list):
        raw_queries = []

    retrieval_queries = [str(q).strip() for q in raw_queries if str(q).strip()]

    if not canonical_terms and not core_query and not retrieval_queries:
        return parsed

    warnings: List[Dict[str, Any]] = []

    support_texts = []
    if core_query:
        support_texts.append(core_query)
    support_texts.extend(retrieval_queries)

    anchor_tokens = _choose_anchor_tokens(
        canonical_terms=canonical_terms,
        core_query=core_query,
        retrieval_queries=retrieval_queries,
    )

    kept_terms: List[str] = []
    removed_terms: List[str] = []

    for idx, term in enumerate(canonical_terms):
        term_tokens = _meaningful_tokens(term)

        if not term_tokens:
            removed_terms.append(term)
            warnings.append(
                {
                    "type": "removed_generic_or_empty_term",
                    "field": "canonical_terms",
                    "text": term,
                }
            )
            continue

        support = _support_count(term_tokens, support_texts)
        overlap = _token_overlap_count(term_tokens, anchor_tokens)
        unique_term_token_count = max(1, len(set(term_tokens)))
        overlap_ratio = overlap / unique_term_token_count

        # Keep the first canonical term because it usually names the main topic.
        # For later terms, require stronger consistency than a single partial overlap.
        if idx == 0:
            keep = True
        elif unique_term_token_count == 1:
            keep = overlap >= 1 or support >= 2
        else:
            keep = overlap_ratio >= 0.67 or support >= 2

        if keep:
            kept_terms.append(term)
        else:
            removed_terms.append(term)
            warnings.append(
                {
                    "type": "removed_low_support_side_topic",
                    "field": "canonical_terms",
                    "text": term,
                    "support_count": support,
                    "overlap_ratio": round(overlap_ratio, 3),
                }
            )
            warnings.append(
                {
                    "type": "removed_low_support_side_topic",
                    "field": "canonical_terms",
                    "text": term,
                    "support_count": support,
                    "overlap_ratio": round(overlap_ratio, 3),
                }
            )

    anchor_tokens = _choose_anchor_tokens(
        canonical_terms=kept_terms,
        core_query=core_query,
        retrieval_queries=retrieval_queries,
    )

    kept_queries: List[str] = []

    for query in retrieval_queries:
        if _looks_like_generation_instruction(query):
            warnings.append(
                {
                    "type": "removed_generation_instruction_query",
                    "field": "retrieval_queries",
                    "text": query,
                }
            )
            continue

        removed_match = _contains_removed_term(query, removed_terms)
        if removed_match:
            warnings.append(
                {
                    "type": "removed_query_containing_removed_term",
                    "field": "retrieval_queries",
                    "text": query,
                    "removed_term": removed_match,
                }
            )
            continue

        query_tokens = _meaningful_tokens(query)

        if query_tokens and anchor_tokens and _token_overlap_count(query_tokens, anchor_tokens) == 0:
            warnings.append(
                {
                    "type": "removed_query_without_topic_anchor",
                    "field": "retrieval_queries",
                    "text": query,
                }
            )
            continue

        if _has_unanchored_joined_part(query, anchor_tokens):
            warnings.append(
                {
                    "type": "removed_query_with_unanchored_joined_part",
                    "field": "retrieval_queries",
                    "text": query,
                }
            )
            continue

        kept_queries.append(query)

    if core_query:
        removed_match = _contains_removed_term(core_query, removed_terms)
        if removed_match or _looks_like_generation_instruction(core_query):
            warnings.append(
                {
                    "type": "rebuilt_core_query",
                    "field": "core_query",
                    "text": core_query,
                    "reason": "contains_removed_or_instruction_term",
                }
            )
            rebuilt_core = build_canonical_query(kept_terms)
            parsed["core_query"] = rebuilt_core or ""
        else:
            parsed["core_query"] = core_query

    parsed["canonical_terms"] = kept_terms
    parsed["retrieval_queries"] = kept_queries

    if warnings:
        parsed["rewrite_warnings"] = warnings

    return parsed


def read_cache(cache_path: Path, cache_key: str) -> Optional[Dict[str, Any]]:
    if not cache_path.exists():
        return None

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("cache_key") == cache_key:
                return row.get("result")

    return None


def append_cache(cache_path: Path, cache_key: str, result: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "cache_key": cache_key,
        "result": result,
    }
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = JSON_OBJECT_RE.search(text)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text[:500]}")

    return json.loads(match.group(0))


def fallback_result(query: str, reason: str) -> Dict[str, Any]:
    return {
        "original_query": query,
        "retrieval_queries": [query],
        "rewrite_used": False,
        "fallback_reason": reason,
    }


def build_rewrite_prompt(query: str, max_queries: int) -> List[Dict[str, str]]:
    system = (
        "You are a retrieval query rewriting assistant for a local RAG system.\n"
        "Your job is to convert the user's request into search queries that retrieve relevant podcast transcript chunks.\n"
        "The source corpus is mostly English podcast transcripts.\n"
        "Return strict JSON only. Do not include markdown. Do not include explanations.\n\n"
        "Important rules:\n"
        "1. Separate content topic from generation style.\n"
        "2. The first English query must focus on the content topic, not the requested output format.\n"
        "3. Use standard English canonical names for known entities, works, events, and concepts.\n"
        "4. Do not create sound-alike translations. Do not guess by phonetics.\n"
        "5. If uncertain about an entity translation, keep the original term and add the most likely canonical English phrase.\n"
        "6. Queries should be useful for semantic retrieval, not final generation.\n"
        "7. Return only the JSON object requested by the user message.\n"
    )

    user = f"""
/no_think

User request:
{query}

Create up to {max_queries} retrieval queries for podcast transcript retrieval.

The output must be strict JSON with this schema:

{{
  "canonical_terms": [
    "standard entity or concept term 1",
    "standard entity or concept term 2"
  ],
  "core_query": "one concise English retrieval query focused on the main content topic",
  "retrieval_queries": [
    "query 1",
    "query 2",
    "query 3",
    "query 4"
  ]
}}

Rules for retrieval_queries:
1. Do not repeat the user's original wording; the system will add it automatically.
2. The first rewritten query must be a concise English topic query.
3. The rewritten queries should retrieve transcript chunks about the same main content topic.
4. Do not include output-format instructions.
5. Avoid malformed phrases or nonstandard translations.
6. Prefer common English phrases used in books, podcasts, and transcripts.
""".strip()

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def rewrite_query(query: str, config: Dict[str, Any]) -> Dict[str, Any]:
    rewrite_cfg = config.get("query_rewrite", {})

    enabled = bool(rewrite_cfg.get("enabled", True))
    if not enabled:
        return fallback_result(query, "query_rewrite_disabled")

    provider = rewrite_cfg.get("provider", "lmstudio")
    if provider != "lmstudio":
        return fallback_result(query, f"unsupported_provider:{provider}")

    base_url = rewrite_cfg.get("base_url", "http://localhost:1234/v1")
    api_key = rewrite_cfg.get("api_key", "lm-studio")
    model = rewrite_cfg.get("model", "qwen3-4b")
    max_queries = int(rewrite_cfg.get("max_queries", 4))
    timeout_seconds = int(rewrite_cfg.get("timeout_seconds", 60))
    include_original_query = bool(rewrite_cfg.get("include_original_query", True))
    fallback_to_original = bool(rewrite_cfg.get("fallback_to_original", True))
    cache_enabled = bool(rewrite_cfg.get("cache_enabled", True))
    cache_path = Path(
        rewrite_cfg.get(
            "cache_path",
            "outputs/source_packs/query_rewrite_cache.jsonl",
        )
    )

    cache_key_payload = {
        "query": query,
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "max_queries": max_queries,
        "prompt_version": rewrite_cfg.get("prompt_version", "v1"),
        "sanitizer_version": SANITIZER_VERSION,
    }
    cache_key = stable_hash(json.dumps(cache_key_payload, ensure_ascii=False, sort_keys=True))

    if cache_enabled:
        cached = read_cache(cache_path, cache_key)
        if cached:
            return cached

    try:
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

        messages = build_rewrite_prompt(query=query, max_queries=max_queries)

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.1,
            max_tokens=600,
        )

        content = response.choices[0].message.content or ""
        parsed = extract_json_object(content)
        parsed = sanitize_rewrite_result(parsed, query)

        canonical_terms = parsed.get("canonical_terms", [])
        if isinstance(canonical_terms, list):
            canonical_terms = [str(t).strip() for t in canonical_terms if str(t).strip()]
        else:
            canonical_terms = []

        retrieval_queries = parsed.get("retrieval_queries", [])
        if not isinstance(retrieval_queries, list):
            raise ValueError("retrieval_queries is not a list")

        cleaned_queries: List[str] = []

        if include_original_query:
            cleaned_queries.append(query)

        core_query = parsed.get("core_query", "")
        if isinstance(core_query, str) and core_query.strip():
            cleaned_queries.append(core_query.strip())
        else:
            canonical_query = build_canonical_query(canonical_terms)
            if canonical_query:
                cleaned_queries.append(canonical_query)

        cleaned_queries.extend(str(q).strip() for q in retrieval_queries if str(q).strip())
        cleaned_queries = dedupe_keep_order(cleaned_queries)[:max_queries]

        if not cleaned_queries:
            raise ValueError("No valid retrieval queries returned")

        result = {
            "original_query": query,
            "canonical_terms": canonical_terms,
            "core_query": core_query.strip() if isinstance(core_query, str) else "",
            "retrieval_queries": cleaned_queries,
            "rewrite_used": True,
            "provider": provider,
            "model": model,
            "sanitizer_version": SANITIZER_VERSION,
        }

        rewrite_warnings = parsed.get("rewrite_warnings", [])
        if rewrite_warnings:
            result["rewrite_warnings"] = rewrite_warnings

        if cache_enabled:
            append_cache(cache_path, cache_key, result)

        return result

    except Exception as e:
        if fallback_to_original:
            return fallback_result(query, f"rewrite_failed:{type(e).__name__}:{e}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    config = load_yaml(args.config)
    result = rewrite_query(args.query, config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()