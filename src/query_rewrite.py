# src/query_rewrite.py

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.lmstudio_utils import assert_lmstudio_model_available, make_lmstudio_client
import yaml
from openai import OpenAI


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


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
        item = normalize_query(item)
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

    # Prefer a compact query that anchors the core topic.
    # This is model-derived, not a hand-written translation dictionary.
    return " ".join(terms)

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
        "1. Separate CONTENT TOPIC from GENERATION STYLE.\n"
        "2. The first English query must focus on the core topic, not the requested output style.\n"
        "3. Use standard English canonical names for known entities, books, myths, historical events, and concepts.\n"
        "4. Do not create sound-alike translations. Do not guess by phonetics.\n"
        "5. If uncertain about an entity translation, keep the original term and add the most likely canonical English phrase.\n"
        "6. Preserve negative constraints, but do not let them erase the core topic.\n"
        "7. Queries should be useful for semantic retrieval, not final generation.\n"
        "8. For named stories, myths, books, historical events, or cultural concepts, prefer the most specific conventional English phrase over a shorter generic term.\n"
        "9. Do not shorten a specific canonical phrase into a broader word. For example, prefer \"Tower of Babel\" over \"Babel\" when the topic is the tower story.\n"
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
  "core_query": "one canonical English retrieval query focused on the main topic",
  "retrieval_queries": [
    "query 1",
    "query 2",
    "query 3",
    "query 4"
  ]
}}

Rules for retrieval_queries:
1. Do not repeat the user's original wording; the system will add it automatically.
2. The first rewritten query must be a canonical English topic query focused on the core subject.
3. The first rewritten query should include the most important canonical terms.
4. Query 3 may add related names, events, concepts, or interpretive keywords.
5. Query 4 may capture modern framing, analogies, or style-relevant concepts.
6. Do not include output-format instructions like "write 30 rounds" unless useful for retrieval.
7. Avoid malformed phrases or nonstandard translations.
8. Prefer common English phrases used in books, podcasts, and transcripts.
9. The core_query must be a concise English query using the most specific canonical phrase available.
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

        canonical_terms = parsed.get("canonical_terms", [])
        if isinstance(canonical_terms, list):
            canonical_terms = [str(t) for t in canonical_terms if str(t).strip()]
        else:
            canonical_terms = []
    
        retrieval_queries = parsed.get("retrieval_queries", [])

        if not isinstance(retrieval_queries, list):
            raise ValueError("retrieval_queries is not a list")

        cleaned_queries = []

        if include_original_query:
            cleaned_queries.append(query)

        core_query = parsed.get("core_query", "")
        if isinstance(core_query, str) and core_query.strip():
            cleaned_queries.append(core_query.strip())
        else:
            canonical_query = build_canonical_query(canonical_terms)
            if canonical_query:
                cleaned_queries.append(canonical_query)

        cleaned_queries.extend(str(q) for q in retrieval_queries)

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
        }

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