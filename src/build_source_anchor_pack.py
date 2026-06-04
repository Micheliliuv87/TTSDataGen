from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import yaml


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


DEFAULT_SOURCE_ANCHOR_CFG: Dict[str, Any] = {
    "output_dir": "outputs/source_packs",
    "latest_path": "outputs/source_packs/latest_source_anchor_pack.json",

    # Inspect topK retrieved podcast chunks.
    "max_sources": 10,

    # Upper bound only. Do not force-fill weak anchors.
    "max_anchors": 6,

    # Excerpt sizes.
    "anchor_excerpt_chars": 1800,
    "top_scored_sentence_preview_chars": 220,

    # Sentence/source scoring.
    # A sentence normally needs at least one strong topic match or multiple
    # support matches to become an anchor sentence.
    "min_sentence_score": 1.6,
    "min_anchor_score": 3.0,
    "min_core_anchor_score": 6.0,
    "min_clean_chars": 120,
    "require_topic_sentence_evidence": True,
    "require_strong_match": True,

    # Context window around selected sentences.
    "window_before": 0,
    "window_after": 1,

    # Cleaning / rejection.
    "drop_noise": True,
    "drop_ads": True,
    "reject_ad_like_sources": True,
    "max_ad_sentence_ratio": 0.30,
    "max_noise_sentence_ratio": 0.50,
    "max_promo_marker_count": 1,

    # Fallback behavior.
    # Keep this off by default. The previous source-level fallback could select
    # weak excerpts when sentence-level evidence was zero.
    "allow_clean_source_fallback": False,
    "allow_trimmed_source_fallback": False,  # backward-compatible old name

    # Diversity controls.
    "enable_diversity_filter": True,
    "max_anchors_per_doc_id": 1,
    "max_anchors_per_podcast_slug": 2,
    "near_duplicate_jaccard_threshold": 0.72,

    # Roles.
    "allow_context_anchors": False,
    "min_core_topic_axis_count": 2,
    "allow_core_with_single_axis_and_metadata_match": False,
}


# ---------------------------------------------------------------------
# Basic IO
# ---------------------------------------------------------------------


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/object: {path}")
    return data


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


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = 999.0) -> float:
    try:
        x = float(value)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def get_source_anchor_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return deep_merge(DEFAULT_SOURCE_ANCHOR_CFG, config.get("source_anchors", {}) or {})


def truncate_text_nicely(text: Any, limit: int) -> str:
    text = clean_text(text)
    if limit <= 0 or len(text) <= limit:
        return text

    truncated = text[:limit].rstrip()
    best = -1
    for marker in [". ", "? ", "! ", "。", "？", "！", "\n"]:
        idx = truncated.rfind(marker)
        if idx > best:
            best = idx + len(marker)

    if best > int(limit * 0.68):
        truncated = truncated[:best].rstrip()

    return truncated + "..."


def unique_preserve_order(items: Iterable[Any]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        s = clean_text(item)
        key = s.lower()
        if s and key not in seen:
            seen.add(key)
            out.append(s)
    return out


# ---------------------------------------------------------------------
# Source helpers
# ---------------------------------------------------------------------


def get_user_query(source_pack: Dict[str, Any]) -> str:
    return clean_text(
        source_pack.get("user_query")
        or source_pack.get("query")
        or source_pack.get("original_query")
        or ""
    )


def normalize_query_rewrite(source_pack: Dict[str, Any]) -> Dict[str, Any]:
    raw = source_pack.get("query_rewrite", {})
    return raw if isinstance(raw, dict) else {}


def source_sort_key(source: Dict[str, Any]) -> Tuple[int, float]:
    return safe_int(source.get("rank"), 999999), safe_float(source.get("distance"), 999.0)


def get_sources(source_pack: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources = source_pack.get("sources", []) or []
    if not isinstance(sources, list):
        return []
    return sorted([s for s in sources if isinstance(s, dict)], key=source_sort_key)


def flatten_query_field(value: Any) -> List[str]:
    out: List[str] = []
    if value is None:
        return out
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            out.extend(flatten_query_field(item))
    elif isinstance(value, dict):
        for item in value.values():
            out.extend(flatten_query_field(item))
    else:
        out.append(str(value))
    return out


# Generic task/format words. This is intentionally not a topic blacklist.
EN_STOPWORDS: Set[str] = {
    "the", "a", "an", "of", "to", "in", "on", "for", "with", "about", "from", "by", "as",
    "and", "or", "vs", "versus", "into", "over", "under", "through", "between", "among",
    "this", "that", "these", "those", "it", "its", "their", "your", "our", "my", "his", "her",
    "is", "are", "was", "were", "be", "being", "been", "can", "could", "should", "would", "will",
    "do", "does", "did", "have", "has", "had", "not", "but", "if", "then", "than", "so",
}

TASK_OR_CORPUS_TOKENS: Set[str] = {
    "generate", "write", "create", "make", "draft", "produce", "output",
    "dialogue", "conversation", "script", "text", "training", "voice", "round", "rounds", "turn", "turns",
    "speaker", "speakers", "line", "lines", "format", "content", "topic", "theme", "related", "relevant",
    "podcast", "podcasts", "transcript", "transcripts", "episode", "episodes", "interview", "interviews",
    "analysis", "analyze", "discussion", "discussions", "source", "sources", "chunk", "chunks", "retrieval",
    "slightly", "expand", "expanded", "extension", "develop", "development", "moderate", "creative",
}

TASK_OR_CORPUS_TOKENS.update(
    {
        "discuss",
        "discusses",
        "discussed",
        "discussing",
        "conversation",
        "conversations",
        "focusing",
        "focused",
        "featuring",
        "feature",
        "features",
        # Retrieval wrapper / functional verbs.
        # These should not become standalone topic evidence.
        "getting",
        "covering",
        "covered",
        "cover",
        "covers",
        "looking",
        "trying",
        "using",
    }
)

WEAK_SEMANTIC_TOKENS: Set[str] = {
    "human", "humans", "people", "person", "life", "world", "society", "social", "culture", "cultural",
    "history", "historical", "story", "stories", "narrative", "narratives", "idea", "ideas", "concept", "concepts",
    "meaning", "context", "contexts", "challenge", "challenges", "change", "changes", "impact", "impacts",
    "future", "modern", "traditional", "personal", "public", "private", "problem", "problems", "issue", "issues",
    "communication", "relationship", "relationships", "experience", "experiences", "memory", "memories",
    "boundary", "boundaries", "question", "questions", "example", "examples",
    "different", "role", "roles", "country", "countries", "gathering", "gatherings", "tradition", "traditions",
    "machine", "machines", "family", "families", "understanding", "language", "languages",
}

# Single tokens created only by rewritten retrieval-query expansion are not
# trusted as primary topic evidence. They may help as support, but they should
# not make a source core by themselves.
UNHELPFUL_EXPANSION_SINGLE_TOKENS: Set[str] = {
    "different", "role", "roles", "future", "modern", "context", "challenge", "challenges",
    "issue", "issues", "problem", "problems", "example", "examples", "country", "countries",
    "culture", "cultural", "human", "humans", "people", "person", "machine", "machines",
}

AMBIGUOUS_SINGLE_TOPIC_TOKENS: Set[str] = WEAK_SEMANTIC_TOKENS | {
    "ai", "language", "languages", "understanding", "human", "humans", "communication",
    "connection", "connections", "family", "families", "memory", "memories",
}

CN_TASK_PATTERNS = [
    r"我想(要)?生成", r"帮我生成", r"请生成", r"生成", r"写", r"输出", r"制作",
    r"\d+\s*轮", r"\d+\s*回合", r"a\s*与\s*b", r"a\s*和\s*b", r"A\s*与\s*B", r"A\s*和\s*B",
    r"对话", r"语音训练文本", r"训练文本", r"文本", r"内容", r"主题", r"相关", r"围绕", r"关于",
    r"可以", r"稍微", r"适度", r"拓展", r"扩展", r"展开", r"发散", r"结合", r"不要", r"太",
]

CN_GENERIC_TERMS: Set[str] = {
    "一个", "一段", "一些", "这个", "那个", "内容", "主题", "相关", "关于", "围绕", "可以", "稍微", "适度",
    "拓展", "扩展", "展开", "发散", "结合", "分析", "讨论", "对话", "语音", "训练", "文本", "生成",
    "历史", "文化", "社会", "人类", "人们", "故事", "问题", "影响", "意义", "背景", "例子", "案例",
}

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'’\-]*|[\u4e00-\u9fff]+")
EN_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9'’\-]*")
CN_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,24}")


def normalize_term(term: Any) -> str:
    term = clean_text(term).lower().replace("’", "'")
    term = re.sub(r"^[\"'“”‘’\[\]{}()]+|[\"'“”‘’\[\]{}()]+$", "", term)
    term = re.sub(r"\s+", " ", term)
    return term.strip()


def normalize_token(token: str) -> str:
    token = normalize_term(token)
    token = token.strip("'’-_")

    # Better lightweight English normalization.
    # This avoids bad forms like memories -> memorie, countries -> countrie.
    if re.fullmatch(r"[a-z][a-z0-9'\-]*", token):
        if len(token) > 5 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]

    return token

def is_task_like_text(text: str) -> bool:
    t = clean_text(text)
    lower = t.lower()

    if any(re.search(pattern, t, flags=re.IGNORECASE) for pattern in CN_TASK_PATTERNS):
        # A text can contain one task marker and still contain a useful topic,
        # but if it is mostly instruction-shaped it should not become a term.
        if len(t) <= 40 or re.search(r"\d+\s*(轮|回合|rounds?|turns?)", lower):
            return True

    words = [normalize_token(w) for w in EN_WORD_RE.findall(lower)]
    if not words:
        return False

    task_count = sum(1 for w in words if w in TASK_OR_CORPUS_TOKENS)
    return task_count >= max(2, len(words) // 2)


def strip_generation_instructions(text: str) -> str:
    t = clean_text(text)
    if not t:
        return ""

    # Prefer content after common topic markers, without depending on any domain.
    marker_match = re.search(r"(?:主题是|内容要与|内容与|围绕|关于)\s*([^。.!?\n]+)", t, flags=re.IGNORECASE)
    if marker_match:
        t = marker_match.group(1)

    for pattern in CN_TASK_PATTERNS:
        t = re.sub(pattern, " ", t, flags=re.IGNORECASE)

    t = re.sub(r"\b\d+\s*(rounds?|turns?|lines?)\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"[。.!?].*$", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" ，,;；:：。.!? ")

def strip_query_wrappers(text: str) -> str:
    """
    Remove generic retrieval-query wrappers without using any topic-specific list.

    Examples of wrappers:
    - discuss ...
    - podcast transcript on ...
    - podcast episode about ...
    - transcript focusing on ...
    - conversations on ...
    """
    t = clean_text(text)
    if not t:
        return ""

    wrapper_patterns = [
        r"^\s*(?:please\s+)?discuss(?:ing)?\s+",
        r"^\s*discussion\s+of\s+",
        r"^\s*conversations?\s+on\s+",
        r"^\s*podcast\s+transcripts?\s+(?:on|about|focusing\s+on|focused\s+on)\s+",
        r"^\s*podcast\s+episodes?\s+(?:on|about|featuring)\s+",
        r"^\s*transcripts?\s+(?:on|about|focusing\s+on|focused\s+on)\s+",
        r"^\s*episodes?\s+(?:on|about|featuring)\s+",
        r"^\s*the\s+science\s+of\s+",
        r"^\s*interviews?\s+(?:on|about|with|covering|focused\s+on|focusing\s+on)\s+",
        r"^\s*stories?\s+(?:on|about|covering)\s+",
    ]

    changed = True
    while changed:
        changed = False
        before = t
        for pattern in wrapper_patterns:
            t = re.sub(pattern, " ", t, flags=re.IGNORECASE)
        t = clean_text(t)
        changed = before != t

    # Remove wrapper words that survived inside a phrase.
    t = re.sub(
        r"\b(?:podcast|podcasts|transcript|transcripts|episode|episodes|interview|interviews|story|stories)\b",
        " ",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r"\b(?:discuss|discusses|discussed|discussing)\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(?:focusing|focused)\s+on\b", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)

    return t.strip(" ,;:，；：")

def term_tokens(term: str) -> List[str]:
    tokens: List[str] = []
    for raw in TOKEN_RE.findall(term or ""):
        token = normalize_token(raw)
        if not token:
            continue
        if token in EN_STOPWORDS or token in TASK_OR_CORPUS_TOKENS:
            continue
        if token in CN_GENERIC_TERMS:
            continue
        if len(token) <= 1:
            continue
        tokens.append(token)
    return unique_preserve_order(tokens)


def term_specificity(term: str) -> str:
    tokens = term_tokens(term)
    if not tokens:
        return "drop"

    en_tokens = [t for t in tokens if re.fullmatch(r"[a-z][a-z0-9'\-]*", t)]
    cn_tokens = [t for t in tokens if re.fullmatch(r"[\u4e00-\u9fff]+", t)]

    weak_count = sum(1 for t in en_tokens if t in WEAK_SEMANTIC_TOKENS)

    # Phrases with at least one specific token are usually useful.
    if len(en_tokens) >= 2:
        if weak_count == len(en_tokens):
            return "weak"
        return "strong"

    if len(cn_tokens) >= 1:
        # Chinese topic hints are useful, but English transcripts may not match them.
        # Keep them as support unless they are very short/generic.
        if all(t in CN_GENERIC_TERMS for t in cn_tokens):
            return "drop"
        return "support"

    if len(en_tokens) == 1:
        tok = en_tokens[0]
        if tok in WEAK_SEMANTIC_TOKENS:
            return "weak"
        if len(tok) >= 4:
            return "strong"

    return "drop"


def split_candidate_terms(text: str, *, source: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    if source == "user_query":
        text = strip_generation_instructions(text)
    else:
        text = strip_query_wrappers(text)

    # Remove obvious query writing instructions, but keep topic phrases.
    if is_task_like_text(text) and source != "user_query":
        return []

    candidates: List[str] = []

    # Full phrase first.
    phrase = normalize_term(text)
    if 2 <= len(phrase) <= 100 and not is_task_like_text(phrase):
        candidates.append(phrase)

    # Split on generic separators/connectors.
    pieces = re.split(
        r"[,;；，、/|]|\b(?:and|or|vs|versus)\b|与|和|以及|及|对比",
        text,
        flags=re.IGNORECASE,
    )

    for piece in pieces:
        p = normalize_term(piece)
        if not p or is_task_like_text(p):
            continue

        if 2 <= len(p) <= 80:
            candidates.append(p)

        words = [normalize_token(w) for w in EN_WORD_RE.findall(p)]
        words = [
            w
            for w in words
            if w
            and w not in EN_STOPWORDS
            and w not in TASK_OR_CORPUS_TOKENS
        ]

        # Keep phrase-level English candidates.
        if len(words) >= 2:
            candidates.append(" ".join(words))

        # Important:
        # Single words from retrieval_queries are often expansion noise.
        # They should not become independent topic evidence unless they also
        # appear through canonical_terms/core_query.
        allow_single_words = source in {"user_query", "canonical_terms", "core_query"}

        if allow_single_words:
            for word in words:
                if len(word) >= 4:
                    candidates.append(word)

        # Chinese topic chunks. No domain-specific segmentation is used.
        for run in CN_RUN_RE.findall(p):
            run = normalize_term(run)
            if run and run not in CN_GENERIC_TERMS and not is_task_like_text(run):
                candidates.append(run)

    return unique_preserve_order(candidates)


def token_variants(token: str) -> List[str]:
    t = normalize_token(token)
    if not t:
        return []

    variants = [t]
    if re.fullmatch(r"[a-z][a-z0-9'\-]*", t):
        if t.endswith("y") and len(t) > 4:
            variants.append(t[:-1] + "ies")
        elif len(t) >= 4 and not t.endswith(("s", "ss")):
            variants.append(t + "s")
    return unique_preserve_order(variants)


def term_variants(term: str) -> List[str]:
    t = normalize_term(term)
    if not t:
        return []

    variants = [t]

    if not re.fullmatch(r"[a-z][a-z0-9'\- ]+", t):
        return unique_preserve_order(variants)

    # Hyphen/space variants, e.g. self-control <-> self control.
    if "-" in t:
        variants.append(t.replace("-", " "))

    words = t.split()

    # Phrase plural support, e.g. autonomous vehicle -> autonomous vehicles.
    if len(words) >= 2:
        last = words[-1]
        if last.endswith("y") and len(last) > 3:
            plural_last = last[:-1] + "ies"
        elif not last.endswith("s"):
            plural_last = last + "s"
        else:
            plural_last = last

        if plural_last != last:
            variants.append(" ".join(words[:-1] + [plural_last]))

    # Single-word singular/plural support.
    if len(words) == 1:
        w = words[0]

        if len(w) >= 4:
            if w.endswith("ies"):
                variants.append(w[:-3] + "y")
            elif w.endswith("y"):
                variants.append(w[:-1] + "ies")
            elif w.endswith("s") and not w.endswith("ss"):
                variants.append(w[:-1])
            else:
                variants.append(w + "s")

        # Lightweight derivational variants.
        # This is generic, not topic-specific:
        # procrastination -> procrastinate / procrastinating
        # translation -> translate / translating
        # communication -> communicate / communicating
        if len(w) > 7 and w.endswith("ation"):
            base = w[:-5] + "ate"
            variants.append(base)
            variants.append(base + "d")
            if base.endswith("e"):
                variants.append(base[:-1] + "ing")
            else:
                variants.append(base + "ing")

    return unique_preserve_order(variants)

def _stronger_specificity(a: str, b: str) -> str:
    order = {"drop": 0, "weak": 1, "support": 2, "strong": 3}
    return a if order.get(a, 0) >= order.get(b, 0) else b


def adjust_specificity_for_origin(
    *,
    term: str,
    tokens: List[str],
    base_specificity: str,
    origin: str,
    core_tokens: Set[str],
) -> str:
    """
    Keep query expansion useful without letting it dominate relevance.

    Generic rule:
    - canonical/core/user-topic terms can define strong topic evidence;
    - retrieval-query expansions can support retrieval, but single expansion
      tokens should not become strong unless they are already part of the core
      topic seed;
    - broad single tokens from retrieval expansion are dropped or weakened.
    """
    if base_specificity == "drop" or not tokens:
        return "drop"

    token_set = set(tokens)
    core_overlap = len(token_set & core_tokens)

    if origin == "retrieval_queries":
        if len(tokens) == 1:
            tok = tokens[0]
            if tok in UNHELPFUL_EXPANSION_SINGLE_TOKENS:
                return "drop"
            if tok in core_tokens:
                if tok in AMBIGUOUS_SINGLE_TOPIC_TOKENS:
                    return "support"
                return "strong"
            return "weak"

        if core_overlap >= 2:
            return "support"
        if core_overlap == 1:
            return "support" if base_specificity == "strong" else "weak"
        return "weak"

    # For terms that come from the user query, canonical terms, or core query,
    # single broad words should usually support, not dominate.
    if len(tokens) == 1 and tokens[0] in AMBIGUOUS_SINGLE_TOPIC_TOKENS:
        return "support"

    return base_specificity


def add_term(
    term_map: Dict[str, Dict[str, Any]],
    term: str,
    *,
    origin: str,
    base_weight: float,
    core_tokens: Optional[Set[str]] = None,
) -> None:
    t = normalize_term(term)
    if not t:
        return

    tokens = term_tokens(t)
    if not tokens:
        return

    core_tokens = core_tokens or set()
    specificity = term_specificity(t)

    # Rescue short but explicit topic words from canonical/core query.
    # Example: "map" is only 3 letters but can be the actual user topic.
    # This remains generic because it only applies to model/user topic fields,
    # not arbitrary retrieval-query expansion words.
    if specificity == "drop":
        if (
            origin in {"canonical_terms", "core_query"}
            and len(tokens) == 1
            and re.fullmatch(r"[a-z][a-z0-9'\-]*", tokens[0])
            and len(tokens[0]) >= 3
            and tokens[0] not in EN_STOPWORDS
            and tokens[0] not in TASK_OR_CORPUS_TOKENS
            and tokens[0] not in WEAK_SEMANTIC_TOKENS
        ):
            specificity = "strong" if origin == "canonical_terms" else "support"
        else:
            return

    specificity = adjust_specificity_for_origin(
        term=t,
        tokens=tokens,
        base_specificity=specificity,
        origin=origin,
        core_tokens=core_tokens,
    )

    if specificity == "drop":
        return

    # If wrappers were removed by term_tokens(), rebuild the display term from
    # remaining tokens so terms like "getting lost" do not keep "getting" in the
    # debug fields or phrase variants.
    if re.fullmatch(r"[a-z][a-z0-9'\- ]+", t):
        display_term = " ".join(tokens)
    else:
        display_term = t

    key = display_term.lower()
    current = term_map.get(key)

    if current is None:
        term_map[key] = {
            "term": display_term,
            "tokens": tokens,
            "specificity": specificity,
            "origins": [origin],
            "weight": base_weight,
            "variants": term_variants(display_term),
        }
        return

    current["weight"] = max(float(current.get("weight", 0.0)), base_weight)
    current["specificity"] = _stronger_specificity(
        str(current.get("specificity", "weak")),
        specificity,
    )
    current["tokens"] = unique_preserve_order(list(current.get("tokens", [])) + tokens)
    current["variants"] = unique_preserve_order(
        list(current.get("variants", [])) + term_variants(display_term)
    )

    origins = list(current.get("origins", []))
    if origin not in origins:
        origins.append(origin)
    current["origins"] = origins

def core_seed_terms(user_query: str, query_rewrite: Dict[str, Any]) -> List[str]:
    seed_terms: List[str] = []

    seed_terms.extend(split_candidate_terms(user_query, source="user_query"))

    for term in flatten_query_field(query_rewrite.get("canonical_terms")):
        seed_terms.extend(split_candidate_terms(term, source="canonical_terms"))

    for term in flatten_query_field(query_rewrite.get("core_query")):
        seed_terms.extend(split_candidate_terms(term, source="core_query"))

    return unique_preserve_order(seed_terms)


def build_core_token_set(seed_terms: List[str]) -> Set[str]:
    tokens: Set[str] = set()
    for term in seed_terms:
        tokens.update(term_tokens(term))
    return tokens


def build_topic_axes(user_query: str, query_rewrite: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build broad topic axes from the user's topic and canonical/core rewrite.

    Axes are not domain-specific lists. They represent the main components of
    the request, such as [waffle], [breakfast, culture], [family, memory].
    Later role inference uses axis coverage so one repeated broad word does not
    make a source look core.
    """
    axis_terms: List[str] = []

    canonical_terms = flatten_query_field(query_rewrite.get("canonical_terms"))
    for term in canonical_terms:
        term = clean_text(term)
        if term:
            axis_terms.append(term)

    # Use user topic hints as fallback or complement. Splitting is generic.
    topic_hint = strip_generation_instructions(user_query)
    for part in re.split(r"[,;；，、/|]|与|和|以及|及|对比", topic_hint):
        part = clean_text(part)
        if part:
            axis_terms.append(part)

    if not axis_terms:
        axis_terms.extend(flatten_query_field(query_rewrite.get("core_query")))

    axes: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, ...]] = set()

    for term in axis_terms:
        candidates = split_candidate_terms(term, source="canonical_terms")
        if not candidates:
            candidates = [term]

        # Prefer the full candidate, not every split single word.
        candidate = normalize_term(candidates[0])
        tokens = term_tokens(candidate)
        if not tokens:
            continue
        if len(tokens) > 6:
            continue

        key = tuple(tokens)
        if key in seen:
            continue
        seen.add(key)

        axes.append(
            {
                "axis_id": len(axes) + 1,
                "term": candidate,
                "tokens": tokens,
            }
        )

    return axes


def build_query_profile(source_pack: Dict[str, Any]) -> Dict[str, Any]:
    user_query = get_user_query(source_pack)
    query_rewrite = normalize_query_rewrite(source_pack)

    seed_terms = core_seed_terms(user_query, query_rewrite)
    core_tokens = build_core_token_set(seed_terms)
    topic_axes = build_topic_axes(user_query, query_rewrite)

    term_map: Dict[str, Dict[str, Any]] = {}

    # User query is used only after stripping task/format instructions.
    for term in split_candidate_terms(user_query, source="user_query"):
        add_term(term_map, term, origin="user_query_topic_hint", base_weight=1.4, core_tokens=core_tokens)

    # Model rewrite fields are already sanitized upstream, but we still filter them.
    for term in flatten_query_field(query_rewrite.get("canonical_terms")):
        for candidate in split_candidate_terms(term, source="canonical_terms"):
            add_term(term_map, candidate, origin="canonical_terms", base_weight=2.0, core_tokens=core_tokens)

    for term in flatten_query_field(query_rewrite.get("core_query")):
        for candidate in split_candidate_terms(term, source="core_query"):
            add_term(term_map, candidate, origin="core_query", base_weight=1.8, core_tokens=core_tokens)

    for term in flatten_query_field(query_rewrite.get("retrieval_queries")):
        # Skip the exact original user query. It contains task instructions.
        if clean_text(term) == user_query:
            continue
        for candidate in split_candidate_terms(term, source="retrieval_queries"):
            add_term(term_map, candidate, origin="retrieval_queries", base_weight=1.2, core_tokens=core_tokens)

    all_terms = list(term_map.values())

    # Sort by specificity/weight/length for stable debug output.
    specificity_order = {"strong": 0, "support": 1, "weak": 2}
    all_terms.sort(
        key=lambda x: (
            specificity_order.get(str(x.get("specificity", "weak")), 9),
            -float(x.get("weight", 0.0)),
            -len(str(x.get("term", ""))),
            str(x.get("term", "")),
        )
    )

    strong_terms = [t["term"] for t in all_terms if t.get("specificity") == "strong"]
    support_terms = [t["term"] for t in all_terms if t.get("specificity") == "support"]
    weak_terms = [t["term"] for t in all_terms if t.get("specificity") == "weak"]

    avoid_terms = [
        "advertisement",
        "sponsor",
        "sponsored",
        "promo code",
        "discount",
        "subscription",
        "special deal",
        "rules and restrictions apply",
        "download the app",
        "wherever you get your podcasts",
        "outro",
        "book promotion",
        "apple podcasts",
        "iheartradio",
        "youtube comments",
        "instagram",
        "tiktok",
    ]

    return {
        "user_query": user_query,
        "query_rewrite": query_rewrite,
        "term_objects": all_terms,
        "topic_axes": topic_axes,
        "core_tokens": sorted(core_tokens),
        # positive_terms is kept for backward/debug compatibility.
        "positive_terms": [t["term"] for t in all_terms],
        "strong_terms": strong_terms,
        "support_terms": support_terms,
        "weak_terms": weak_terms,
        "avoid_terms": unique_preserve_order(avoid_terms),
    }


# ---------------------------------------------------------------------
# Cleaning and scoring
# ---------------------------------------------------------------------


PODCAST_NOISE_PATTERNS: List[str] = [
    r"\baudio transcribed by\b",
    r"\byou'?re listening to\b",
    r"\bsubscribe\b.*\b(apple|spotify|stitcher|podcast|podcasts)\b",
    r"\bwherever you (listen|get) (to )?(your )?podcasts\b",
    r"\bleave a review\b",
    r"\bdownload .*\bapp\b",
    r"\bavailable right now\b",
    r"\bstay with us\b",
    r"\bnext time\b.*\bwe'?ll hear\b",
    r"\bthank you for listening\b",
    r"\bthanks so much\b",
    r"\bmake sure you share\b",
    r"\btag both of us\b",
    r"\byoutube comments\b",
    r"\bif you love this episode\b",
    r"\byou'?ll enjoy my conversation\b",
    r"\btune in to\b",
    r"\bnew podcast\b",
    r"\bwe'?re here to introduce you\b",
    r"\bhello sunshine\b",
    r"\biheart ?radio\b",
    r"\bapple podcasts\b",
    r"\bpray\.?\s*com\b",
    r"\bjackgraham\.?\s*org\b",
    r"\bgo to .{0,40}\.org\b",
    r"\bgo to .{0,40}\.com\b",
]

AD_LIKE_PATTERNS: List[str] = [
    r"\bthis episode is brought to you by\b",
    r"\bthis podcast is brought to you by\b",
    r"\bbrought to you by\b",
    r"\bsponsored by\b",
    r"\bpaid advertisement\b",
    r"\bpromo code\b",
    r"\buse code\b",
    r"\bfree shipping\b",
    r"\bterms (and conditions )?apply\b",
    r"\brules and restrictions may apply\b",
    r"\b\d+%\s+off\b",
    r"\b\d+\s+percent\s+off\b",
    r"\bspecial deal\b",
    r"\bsubscription\b",
    r"\bsubscriptions sold\b",
    r"\bget up to\b.{0,40}\boff\b",
    r"\bgo to\s+[a-z0-9.-]+\s*\.com\b",
    r"\bvisit\s+[a-z0-9.-]+\s*\.com\b",
    r"\b[a-z0-9.-]+\.com\s*/\s*\w+\b",
    r"\bmember fdic\b",
    r"\bquick ten minute lessons\b",
    r"\bhandcrafted by over\b.{0,30}\blanguage experts\b",
    r"\bstart speaking a new language in as little as\b",
    r"\bscience backed language learning app\b",
]

PROMO_MARKERS: List[str] = [
    "special deal",
    "get up to",
    "off your",
    "subscription",
    "rules and restrictions",
    "wherever you get your podcasts",
    "download the app",
    "apple podcasts",
    "iheartradio",
    "promo code",
    "use code",
]


def regex_any(text: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def split_into_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    pieces = re.split(r"(?<=[.!?。！？])\s+", text)
    pieces = [p.strip() for p in pieces if p.strip()]

    out: List[str] = []
    for piece in pieces:
        if len(piece) <= 900:
            out.append(piece)
            continue
        subpieces = re.split(r"(?<=[,，;；:：])\s+", piece)
        out.extend([s.strip() for s in subpieces if s.strip()])

    return out


def classify_sentence(sentence: str) -> str:
    s = clean_text(sentence)
    if not s:
        return "empty"
    if regex_any(s, AD_LIKE_PATTERNS):
        return "ad"
    if regex_any(s, PODCAST_NOISE_PATTERNS):
        return "noise"
    return "content"


def clean_source_text(text: str, *, drop_noise: bool, drop_ads: bool) -> Tuple[str, Dict[str, Any]]:
    original = clean_text(text)
    sentences = split_into_sentences(original)

    stats = {
        "raw_chars": len(original),
        "raw_sentence_count": len(sentences),
        "kept_sentences": 0,
        "removed_noise_sentences": 0,
        "removed_ad_sentences": 0,
        "removed_duplicate_sentences": 0,
        "ad_sentence_ratio": 0.0,
        "noise_sentence_ratio": 0.0,
        "promo_marker_count": 0,
    }

    if not sentences:
        return original, stats

    kept: List[str] = []
    seen: Set[str] = set()
    ad_count = 0
    noise_count = 0

    for sentence in sentences:
        s = clean_text(sentence)
        if not s:
            continue

        label = classify_sentence(s)
        if label == "ad":
            ad_count += 1
        elif label == "noise":
            noise_count += 1

        key = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", s.lower()).strip()[:220]
        if len(key) >= 40 and key in seen:
            stats["removed_duplicate_sentences"] += 1
            continue
        if len(key) >= 40:
            seen.add(key)

        if drop_ads and label == "ad":
            stats["removed_ad_sentences"] += 1
            continue
        if drop_noise and label == "noise":
            stats["removed_noise_sentences"] += 1
            continue

        kept.append(s)

    denominator = max(1, len(sentences))
    lower = original.lower()

    stats["ad_sentence_ratio"] = round(ad_count / denominator, 3)
    stats["noise_sentence_ratio"] = round(noise_count / denominator, 3)
    stats["promo_marker_count"] = sum(1 for marker in PROMO_MARKERS if marker in lower)
    stats["kept_sentences"] = len(kept)

    return clean_text(" ".join(kept)), stats


def english_boundary_pattern(term: str) -> str:
    words = [re.escape(w) for w in term.split() if w]
    if not words:
        return r"$a"
    return r"(?<![a-z0-9])" + r"[\s\-]+".join(words) + r"(?![a-z0-9])"


def term_matches_text(text: str, term: str) -> bool:
    t = normalize_term(term)
    if not t:
        return False

    lower = clean_text(text).lower().replace("’", "'")
    if not lower:
        return False

    if re.search(r"[\u4e00-\u9fff]", t):
        return t in lower

    if re.fullmatch(r"[a-z][a-z0-9'\- ]+", t):
        pattern = english_boundary_pattern(t)
        return re.search(pattern, lower, flags=re.IGNORECASE) is not None

    return t in lower


def axis_token_matches_text(text: str, token: str) -> bool:
    for variant in token_variants(token):
        if term_matches_text(text, variant):
            return True
    return False


def topic_axis_matches(text: str, topic_axes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    matches: List[Dict[str, Any]] = []

    for axis in topic_axes:
        tokens = list(axis.get("tokens", []) or [])
        if not tokens:
            continue

        matched_tokens = [token for token in tokens if axis_token_matches_text(text, token)]
        if not matched_tokens:
            continue

        exact_term_match = term_matches_text(text, str(axis.get("term", "")))
        required = 1
        if len(tokens) >= 3:
            required = 2
        elif len(tokens) == 2:
            # One specific token can activate a two-token axis, but one broad
            # token such as language/human/culture/understanding cannot.
            specific_single_hit = any(t not in AMBIGUOUS_SINGLE_TOPIC_TOKENS for t in matched_tokens)
            required = 1 if specific_single_hit else 2

        if exact_term_match or len(matched_tokens) >= required:
            matches.append(
                {
                    "axis_id": axis.get("axis_id"),
                    "term": axis.get("term"),
                    "matched_tokens": matched_tokens,
                }
            )

    # De-duplicate by axis id/term.
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for match in matches:
        key = str(match.get("axis_id")) + ":" + str(match.get("term"))
        if key in seen:
            continue
        seen.add(key)
        out.append(match)
    return out


def match_terms(
    text: str,
    term_objects: List[Dict[str, Any]],
    topic_axes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    strong: List[str] = []
    support: List[str] = []
    weak: List[str] = []
    score = 0.0

    for obj in term_objects:
        term = str(obj.get("term", ""))
        variants = list(obj.get("variants", []) or [term])
        if not any(term_matches_text(text, variant) for variant in variants):
            continue

        specificity = str(obj.get("specificity", "weak"))
        weight = float(obj.get("weight", 1.0) or 1.0)

        if specificity == "strong":
            strong.append(term)
            score += 1.7 * weight
        elif specificity == "support":
            support.append(term)
            score += 1.0 * weight
        else:
            weak.append(term)
            score += 0.25 * weight

    axis_matches = topic_axis_matches(text, topic_axes or [])
    if axis_matches:
        score += min(len(axis_matches), 3) * 0.8

    return {
        "score": score,
        "strong_matches": unique_preserve_order(strong),
        "support_matches": unique_preserve_order(support),
        "weak_matches": unique_preserve_order(weak),
        "all_matches": unique_preserve_order(strong + support + weak),
        "topic_axis_matches": axis_matches,
        "topic_axis_count": len(axis_matches),
    }


def topic_evidence_ok(match_info: Dict[str, Any], *, require_strong_match: bool) -> bool:
    strong = match_info.get("strong_matches", []) or []
    support = match_info.get("support_matches", []) or []
    axis_count = int(match_info.get("topic_axis_count", 0) or 0)

    if axis_count >= 1 and strong:
        return True

    if axis_count >= 2 and (strong or support):
        return True

    if require_strong_match:
        return axis_count >= 1 and len(support) >= 2

    return axis_count >= 1 and (len(support) >= 1 or len(strong) >= 1)


def source_level_score(source: Dict[str, Any], profile: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score source-level metadata only.

    Do NOT match query terms against query_rewrite or matched_queries; that would
    score the query against itself and inflate source relevance.
    """
    title = clean_text(source.get("title", ""))
    podcast_slug = clean_text(source.get("podcast_slug", ""))
    metadata_blob = clean_text(" ".join([title, podcast_slug]))

    match_info = match_terms(
        metadata_blob,
        profile.get("term_objects", []),
        profile.get("topic_axes", []),
    )
    distance = safe_float(source.get("distance"), 999.0)

    score = 0.0
    score += float(match_info.get("score", 0.0))

    # Retrieval distance is a weak prior, not enough to create an anchor alone.
    if distance <= 0.32:
        score += 1.2
    elif distance <= 0.38:
        score += 0.8
    elif distance <= 0.45:
        score += 0.4

    return {
        "source_score": round(score, 3),
        "metadata_matches": match_info.get("all_matches", []),
        "metadata_strong_matches": match_info.get("strong_matches", []),
        "metadata_support_matches": match_info.get("support_matches", []),
        "metadata_weak_matches": match_info.get("weak_matches", []),
        "metadata_topic_axis_matches": match_info.get("topic_axis_matches", []),
        "metadata_topic_axis_count": match_info.get("topic_axis_count", 0),
        "retrieval_distance": distance,
    }


def score_sentence(sentence: str, profile: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    text = clean_text(sentence)

    score = 0.0
    penalties: List[str] = []

    if not text:
        return {
            "score": -999.0,
            "matched_terms": [],
            "strong_matched_terms": [],
            "support_matched_terms": [],
            "weak_matched_terms": [],
            "topic_axis_matches": [],
            "topic_axis_count": 0,
            "topic_evidence_ok": False,
            "penalties": ["empty"],
        }

    label = classify_sentence(text)
    if label == "ad":
        score -= 20.0
        penalties.append("ad")
    elif label == "noise":
        score -= 10.0
        penalties.append("noise")

    if len(text) < 24:
        score -= 0.4
        penalties.append("short")
    if len(text) > 900:
        score -= 0.5
        penalties.append("long")

    match_info = match_terms(
        text,
        profile.get("term_objects", []),
        profile.get("topic_axes", []),
    )
    score += float(match_info.get("score", 0.0))

    for term in profile.get("avoid_terms", []):
        t = normalize_term(term)
        if t and term_matches_text(text, t):
            score -= 4.0
            penalties.append(f"avoid:{t}")

    if re.search(r"\b(app|subscription|lessons|promo|discount|deal)\b", text.lower()):
        if regex_any(text, AD_LIKE_PATTERNS):
            score -= 6.0
            penalties.append("product_promo_like")

    evidence_ok = topic_evidence_ok(
        match_info,
        require_strong_match=bool(cfg.get("require_strong_match", True)),
    )

    return {
        "score": round(score, 3),
        "matched_terms": match_info.get("all_matches", []),
        "strong_matched_terms": match_info.get("strong_matches", []),
        "support_matched_terms": match_info.get("support_matches", []),
        "weak_matched_terms": match_info.get("weak_matches", []),
        "topic_axis_matches": match_info.get("topic_axis_matches", []),
        "topic_axis_count": match_info.get("topic_axis_count", 0),
        "topic_evidence_ok": evidence_ok,
        "penalties": unique_preserve_order(penalties),
    }


def should_reject_source_before_selection(
    *,
    cleaning_stats: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if not bool(cfg.get("reject_ad_like_sources", True)):
        return False, reasons

    ad_ratio = float(cleaning_stats.get("ad_sentence_ratio", 0.0) or 0.0)
    noise_ratio = float(cleaning_stats.get("noise_sentence_ratio", 0.0) or 0.0)
    promo_marker_count = int(cleaning_stats.get("promo_marker_count", 0) or 0)

    max_ad_ratio = float(cfg.get("max_ad_sentence_ratio", 0.30))
    max_noise_ratio = float(cfg.get("max_noise_sentence_ratio", 0.50))
    max_promo_marker_count = int(cfg.get("max_promo_marker_count", 1))

    if ad_ratio > max_ad_ratio:
        reasons.append(f"ad_sentence_ratio>{max_ad_ratio}")
    if noise_ratio > max_noise_ratio:
        reasons.append(f"noise_sentence_ratio>{max_noise_ratio}")
    if promo_marker_count > max_promo_marker_count:
        reasons.append(f"promo_marker_count>{max_promo_marker_count}")

    return bool(reasons), reasons


def build_top_scored_sentences(scored: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    limit = int(cfg.get("top_scored_sentence_preview_chars", 220))
    return [
        {
            "score": item.get("score"),
            "matched_terms": item.get("matched_terms", []),
            "strong_matched_terms": item.get("strong_matched_terms", []),
            "support_matched_terms": item.get("support_matched_terms", []),
            "weak_matched_terms": item.get("weak_matched_terms", []),
            "topic_axis_matches": item.get("topic_axis_matches", []),
            "topic_axis_count": item.get("topic_axis_count", 0),
            "topic_evidence_ok": item.get("topic_evidence_ok", False),
            "penalties": item.get("penalties", []),
            "text": truncate_text_nicely(item.get("text", ""), limit),
        }
        for item in sorted(scored, key=lambda x: float(x.get("score", 0.0)), reverse=True)[:6]
    ]


def first_clean_content_excerpt(sentences: List[str], *, limit: int) -> str:
    kept: List[str] = []
    total = 0

    for sentence in sentences:
        s = clean_text(sentence)
        if not s:
            continue
        if classify_sentence(s) != "content":
            continue
        kept.append(s)
        total += len(s)
        if total >= limit:
            break

    return truncate_text_nicely(" ".join(kept), limit)


def extract_anchor_excerpt(
    source: Dict[str, Any],
    *,
    profile: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    raw_text = clean_text(source.get("text", ""))

    cleaned_text, cleaning_stats = clean_source_text(
        raw_text,
        drop_noise=bool(cfg.get("drop_noise", True)),
        drop_ads=bool(cfg.get("drop_ads", True)),
    )

    source_score_info = source_level_score(source, profile)

    reject_source, reject_reasons = should_reject_source_before_selection(
        cleaning_stats=cleaning_stats,
        cfg=cfg,
    )

    if reject_source:
        return {
            "selected_excerpt": "",
            "anchor_score": 0.0,
            "matched_terms": [],
            "strong_matched_terms": [],
            "support_matched_terms": [],
            "weak_matched_terms": [],
            "source_score_info": source_score_info,
            "cleaning": cleaning_stats,
            "selection_method": "rejected_source_before_selection",
            "rejected": True,
            "reject_reasons": reject_reasons,
            "top_scored_sentences": [],
        }

    if len(cleaned_text) < int(cfg.get("min_clean_chars", 120)):
        return {
            "selected_excerpt": "",
            "anchor_score": 0.0,
            "matched_terms": [],
            "strong_matched_terms": [],
            "support_matched_terms": [],
            "weak_matched_terms": [],
            "source_score_info": source_score_info,
            "cleaning": cleaning_stats,
            "selection_method": "rejected_too_little_clean_text",
            "rejected": True,
            "reject_reasons": ["too_little_clean_text"],
            "top_scored_sentences": [],
        }

    sentences = split_into_sentences(cleaned_text)
    scored: List[Dict[str, Any]] = []

    for idx, sentence in enumerate(sentences):
        score_obj = score_sentence(sentence, profile, cfg)
        scored.append({"idx": idx, "text": sentence, **score_obj})

    min_score = float(cfg.get("min_sentence_score", 1.6))
    require_topic_sentence_evidence = bool(cfg.get("require_topic_sentence_evidence", True))

    selected_indices = {
        item["idx"]
        for item in scored
        if float(item.get("score", 0.0)) >= min_score
        and (not require_topic_sentence_evidence or bool(item.get("topic_evidence_ok", False)))
        and "ad" not in item.get("penalties", [])
        and "noise" not in item.get("penalties", [])
        and "product_promo_like" not in item.get("penalties", [])
    }

    allow_clean_fallback = bool(cfg.get("allow_clean_source_fallback", False)) or bool(
        cfg.get("allow_trimmed_source_fallback", False)
    )

    # Fallback is intentionally conservative and off by default.
    # It can only fire when there is some sentence-level evidence, not zero.
    if not selected_indices and allow_clean_fallback:
        best_sentence = max(scored, key=lambda x: float(x.get("score", -999.0)), default=None)
        if best_sentence and bool(best_sentence.get("topic_evidence_ok", False)):
            excerpt = first_clean_content_excerpt(
                sentences,
                limit=int(cfg.get("anchor_excerpt_chars", 1800)),
            )
            if excerpt:
                return {
                    "selected_excerpt": excerpt,
                    "anchor_score": round(max(float(best_sentence.get("score", 0.0)), 0.0), 3),
                    "matched_terms": best_sentence.get("matched_terms", []),
                    "strong_matched_terms": best_sentence.get("strong_matched_terms", []),
                    "support_matched_terms": best_sentence.get("support_matched_terms", []),
                    "weak_matched_terms": best_sentence.get("weak_matched_terms", []),
                    "topic_axis_matches": best_sentence.get("topic_axis_matches", []),
                    "topic_axis_count": best_sentence.get("topic_axis_count", 0),
                    "source_score_info": source_score_info,
                    "cleaning": cleaning_stats,
                    "selection_method": "conservative_clean_source_fallback_with_sentence_evidence",
                    "rejected": False,
                    "reject_reasons": [],
                    "top_scored_sentences": build_top_scored_sentences(scored, cfg),
                }

    if not selected_indices:
        return {
            "selected_excerpt": "",
            "anchor_score": 0.0,
            "matched_terms": [],
            "strong_matched_terms": [],
            "support_matched_terms": [],
            "weak_matched_terms": [],
            "source_score_info": source_score_info,
            "cleaning": cleaning_stats,
            "selection_method": "rejected_no_relevant_clean_sentences",
            "rejected": True,
            "reject_reasons": ["no_relevant_clean_sentences"],
            "top_scored_sentences": build_top_scored_sentences(scored, cfg),
        }

    window_before = int(cfg.get("window_before", 0))
    window_after = int(cfg.get("window_after", 1))

    expanded_indices: Set[int] = set()
    for idx in selected_indices:
        for j in range(idx - window_before, idx + window_after + 1):
            if 0 <= j < len(scored):
                candidate = scored[j]
                penalties = candidate.get("penalties", [])
                if "ad" in penalties or "noise" in penalties or "product_promo_like" in penalties:
                    continue
                expanded_indices.add(j)

    selected_sentences = [item for item in scored if item["idx"] in expanded_indices]

    if not selected_sentences:
        return {
            "selected_excerpt": "",
            "anchor_score": 0.0,
            "matched_terms": [],
            "strong_matched_terms": [],
            "support_matched_terms": [],
            "weak_matched_terms": [],
            "source_score_info": source_score_info,
            "cleaning": cleaning_stats,
            "selection_method": "rejected_empty_after_expansion",
            "rejected": True,
            "reject_reasons": ["empty_after_noise_filter"],
            "top_scored_sentences": build_top_scored_sentences(scored, cfg),
        }

    selected_text = clean_text(" ".join(item["text"] for item in selected_sentences))
    selected_text = truncate_text_nicely(selected_text, int(cfg.get("anchor_excerpt_chars", 1800)))

    matched_terms: List[str] = []
    strong_matched_terms: List[str] = []
    support_matched_terms: List[str] = []
    weak_matched_terms: List[str] = []
    topic_axis_matches: List[Dict[str, Any]] = []
    anchor_score = float(source_score_info.get("source_score", 0.0)) * 0.20

    for item in selected_sentences:
        matched_terms.extend(item.get("matched_terms", []))
        strong_matched_terms.extend(item.get("strong_matched_terms", []))
        support_matched_terms.extend(item.get("support_matched_terms", []))
        weak_matched_terms.extend(item.get("weak_matched_terms", []))
        topic_axis_matches.extend(item.get("topic_axis_matches", []) or [])
        anchor_score += float(item.get("score", 0.0))

    unique_axis_matches: List[Dict[str, Any]] = []
    seen_axis_keys: Set[str] = set()
    for axis in topic_axis_matches:
        key = str(axis.get("axis_id")) + ":" + str(axis.get("term"))
        if key in seen_axis_keys:
            continue
        seen_axis_keys.add(key)
        unique_axis_matches.append(axis)

    return {
        "selected_excerpt": selected_text,
        "anchor_score": round(anchor_score, 3),
        "matched_terms": unique_preserve_order(matched_terms),
        "strong_matched_terms": unique_preserve_order(strong_matched_terms),
        "support_matched_terms": unique_preserve_order(support_matched_terms),
        "weak_matched_terms": unique_preserve_order(weak_matched_terms),
        "topic_axis_matches": unique_axis_matches,
        "topic_axis_count": len(unique_axis_matches),
        "source_score_info": source_score_info,
        "cleaning": cleaning_stats,
        "selection_method": "scored_clean_sentence_window",
        "rejected": False,
        "reject_reasons": [],
        "top_scored_sentences": build_top_scored_sentences(scored, cfg),
    }


# ---------------------------------------------------------------------
# Anchor acceptance, diversity, and output
# ---------------------------------------------------------------------


def infer_anchor_role(anchor: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    score = float(anchor.get("anchor_score", 0.0))
    strong_count = len(anchor.get("strong_matched_terms", []) or [])
    support_count = len(anchor.get("support_matched_terms", []) or [])
    axis_count = int(anchor.get("topic_axis_count", 0) or 0)

    source_score_info = anchor.get("source_score_info", {}) or {}
    metadata_axis_count = int(source_score_info.get("metadata_topic_axis_count", 0) or 0)
    metadata_strong = source_score_info.get("metadata_strong_matches", []) or []

    min_core_score = float(cfg.get("min_core_anchor_score", 6.0))
    min_anchor_score = float(cfg.get("min_anchor_score", 3.0))
    min_core_axis_count = int(cfg.get("min_core_topic_axis_count", 2))

    enough_score = score >= min_core_score
    enough_axis_coverage = axis_count >= min_core_axis_count
    single_axis_with_metadata = (
        bool(cfg.get("allow_core_with_single_axis_and_metadata_match", False))
        and axis_count >= 1
        and (metadata_axis_count >= 1 or bool(metadata_strong))
    )

    if enough_score and (enough_axis_coverage or single_axis_with_metadata):
        return "core"

    if score >= min_anchor_score and axis_count >= 1 and (strong_count >= 1 or support_count >= 1):
        return "supporting"

    return "context"


def validate_anchor_acceptance(
    *,
    extracted: Dict[str, Any],
    role: str,
    cfg: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []

    if extracted.get("rejected"):
        reasons.extend(extracted.get("reject_reasons", []) or ["extractor_rejected"])
        return False, reasons

    score = float(extracted.get("anchor_score", 0.0))
    min_anchor_score = float(cfg.get("min_anchor_score", 3.0))
    axis_count = int(extracted.get("topic_axis_count", 0) or 0)

    if score < min_anchor_score:
        reasons.append(f"anchor_score<{min_anchor_score}")

    if axis_count < 1:
        reasons.append("no_topic_axis_coverage")

    if bool(cfg.get("require_strong_match", True)):
        strong = extracted.get("strong_matched_terms", []) or []
        support = extracted.get("support_matched_terms", []) or []
        if not strong and len(support) < 2:
            reasons.append("insufficient_topic_evidence")

    if role == "context" and not bool(cfg.get("allow_context_anchors", False)):
        reasons.append("context_anchor_disabled")

    return not reasons, reasons


def role_order(role: str) -> int:
    order = {"core": 0, "supporting": 1, "context": 2}
    return order.get(role, 9)


def token_set_for_similarity(text: str) -> Set[str]:
    tokens = [normalize_token(t) for t in TOKEN_RE.findall(clean_text(text).lower())]
    return {t for t in tokens if t and t not in EN_STOPWORDS and t not in TASK_OR_CORPUS_TOKENS and len(t) > 1}


def jaccard_similarity(a: str, b: str) -> float:
    set_a = token_set_for_similarity(a)
    set_b = token_set_for_similarity(b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def diversity_rejection_reason(
    candidate: Dict[str, Any],
    selected: List[Dict[str, Any]],
    cfg: Dict[str, Any],
) -> Optional[str]:
    if not bool(cfg.get("enable_diversity_filter", True)):
        return None

    doc_id = clean_text(candidate.get("doc_id", ""))
    podcast_slug = clean_text(candidate.get("podcast_slug", ""))
    max_per_doc = int(cfg.get("max_anchors_per_doc_id", 1))
    max_per_podcast = int(cfg.get("max_anchors_per_podcast_slug", 2))

    if doc_id and max_per_doc > 0:
        count = sum(1 for item in selected if clean_text(item.get("doc_id", "")) == doc_id)
        if count >= max_per_doc:
            return f"diversity:max_anchors_per_doc_id>{max_per_doc}"

    if podcast_slug and max_per_podcast > 0:
        count = sum(1 for item in selected if clean_text(item.get("podcast_slug", "")) == podcast_slug)
        if count >= max_per_podcast:
            return f"diversity:max_anchors_per_podcast_slug>{max_per_podcast}"

    threshold = float(cfg.get("near_duplicate_jaccard_threshold", 0.72))
    candidate_text = clean_text(candidate.get("selected_excerpt", ""))
    for item in selected:
        sim = jaccard_similarity(candidate_text, clean_text(item.get("selected_excerpt", "")))
        if sim >= threshold:
            return f"diversity:near_duplicate_jaccard>={threshold}"

    return None

def anchor_axis_ids(anchor: Dict[str, Any]) -> Set[int]:
    ids: Set[int] = set()

    for item in anchor.get("topic_axis_matches", []) or []:
        if not isinstance(item, dict):
            continue
        axis_id = safe_int(item.get("axis_id"), -1)
        if axis_id > 0:
            ids.add(axis_id)

    return ids


def metadata_axis_count(anchor: Dict[str, Any]) -> int:
    info = (
        anchor.get("selection", {})
        .get("source_score_info", {})
    )
    return safe_int(info.get("metadata_topic_axis_count"), 0)

def build_why_useful(*, role: str, matched_terms: List[str], strong_terms: List[str]) -> str:
    terms = strong_terms[:8] or matched_terms[:8]
    term_text = ", ".join(terms) if terms else "retrieved-source relevance with sentence-level topic evidence"
    return f"{role.capitalize()} cleaned source excerpt selected by sentence-level topic evidence: {term_text}."


def build_suggested_use(*, role: str) -> str:
    if role == "core":
        return (
            "Use as a main content source. Preserve concrete details and unfold them across multiple developed turns. "
            "Do not mechanically translate the excerpt."
        )

    if role == "supporting":
        return "Use to add examples, contrast, or secondary framing. Do not let this source override stronger core sources."

    return "Use only if directly relevant. Omit if it makes the dialogue generic or off-topic."


def build_source_anchors(
    *,
    source_pack: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    cfg = get_source_anchor_config(config)
    sources = get_sources(source_pack)
    profile = build_query_profile(source_pack)

    max_sources = int(cfg.get("max_sources", 10))
    max_anchors = int(cfg.get("max_anchors", 6))

    candidate_anchors: List[Dict[str, Any]] = []
    rejected_candidates: List[Dict[str, Any]] = []

    for source in sources[:max_sources]:
        extracted = extract_anchor_excerpt(source, profile=profile, cfg=cfg)
        excerpt = clean_text(extracted.get("selected_excerpt", ""))

        role = infer_anchor_role(extracted, cfg)
        accepted, rejection_reasons = validate_anchor_acceptance(extracted=extracted, role=role, cfg=cfg)

        base_record = {
            "source_rank": source.get("rank", ""),
            "source_role": role,
            "title": clean_text(source.get("title", "")),
            "podcast_slug": clean_text(source.get("podcast_slug", "")),
            "url": clean_text(source.get("url", "")),
            "doc_id": clean_text(source.get("doc_id", "")),
            "chunk_id": clean_text(source.get("chunk_id", "")),
            "chunk_index": source.get("chunk_index", ""),
            "start_timestamp": source.get("start_timestamp", ""),
            "end_timestamp": source.get("end_timestamp", ""),
            "retrieval_distance": source.get("distance", ""),
            "matched_queries": source.get("matched_queries", []),
            "anchor_score": float(extracted.get("anchor_score", 0.0)),
            "matched_terms": extracted.get("matched_terms", []) or [],
            "strong_matched_terms": extracted.get("strong_matched_terms", []) or [],
            "support_matched_terms": extracted.get("support_matched_terms", []) or [],
            "weak_matched_terms": extracted.get("weak_matched_terms", []) or [],
            "topic_axis_matches": extracted.get("topic_axis_matches", []) or [],
            "topic_axis_count": extracted.get("topic_axis_count", 0) or 0,
            "selected_excerpt": excerpt,
            "selection": {
                "method": extracted.get("selection_method", ""),
                "source_score_info": extracted.get("source_score_info", {}),
                "cleaning": extracted.get("cleaning", {}),
                "top_scored_sentences": extracted.get("top_scored_sentences", []),
                "reject_reasons": extracted.get("reject_reasons", []),
            },
        }

        if not accepted:
            base_record["rejection_reasons"] = rejection_reasons
            rejected_candidates.append(base_record)
            continue

        base_record.update(
            {
                "anchor_id": len(candidate_anchors) + 1,
                "why_useful": build_why_useful(
                    role=role,
                    matched_terms=base_record["matched_terms"],
                    strong_terms=base_record["strong_matched_terms"],
                ),
                "suggested_use": build_suggested_use(role=role),
            }
        )

        candidate_anchors.append(base_record)

    candidate_anchors.sort(
        key=lambda item: (
            role_order(item.get("source_role", "")),
            -float(item.get("anchor_score", 0.0)),
            safe_int(item.get("source_rank"), 999999),
        )
    )

    selected_anchors: List[Dict[str, Any]] = []
    unused_accepted: List[Dict[str, Any]] = []
    covered_axis_ids: Set[int] = set()

    remaining = list(candidate_anchors)

    while remaining and len(selected_anchors) < max_anchors:
        viable: List[Dict[str, Any]] = []

        for candidate in remaining:
            reason = diversity_rejection_reason(candidate, selected_anchors, cfg)
            if reason:
                candidate = dict(candidate)
                candidate["diversity_rejection_reason"] = reason
                unused_accepted.append(candidate)
                continue
            viable.append(candidate)

        if not viable:
            break

        def coverage_key(candidate: Dict[str, Any]) -> Tuple[int, int, int, int, float, int]:
            axes = anchor_axis_ids(candidate)
            new_axis_count = len(axes - covered_axis_ids)
            total_axis_count = len(axes)

            role = str(candidate.get("source_role", ""))
            role_bonus = 2 if role == "core" else 1 if role == "supporting" else 0

            meta_axis_count = metadata_axis_count(candidate)
            score = float(candidate.get("anchor_score", 0.0))
            rank = safe_int(candidate.get("source_rank"), 999999)

            return (
                new_axis_count,
                total_axis_count,
                role_bonus,
                meta_axis_count,
                score,
                -rank,
            )

        best_idx, best_candidate = max(
            enumerate(viable),
            key=lambda pair: coverage_key(pair[1]),
        )

        selected_anchors.append(best_candidate)
        covered_axis_ids.update(anchor_axis_ids(best_candidate))

        remaining = [
            candidate
            for idx, candidate in enumerate(viable)
            if idx != best_idx
        ]

    # Anything still remaining is accepted but not selected because max_anchors
    # was reached or because it added less coverage than selected candidates.
    for candidate in remaining:
        candidate = dict(candidate)
        candidate.setdefault("selection_rejection_reason", "lower_axis_coverage_or_over_max_anchors")
        unused_accepted.append(candidate)

    for idx, anchor in enumerate(selected_anchors, start=1):
        anchor["anchor_id"] = idx

    return {
        "pack_type": "source_anchor_pack",
        "user_query": get_user_query(source_pack),
        "query_profile": {
            "positive_terms": profile.get("positive_terms", []),
            "strong_terms": profile.get("strong_terms", []),
            "support_terms": profile.get("support_terms", []),
            "weak_terms": profile.get("weak_terms", []),
            "term_objects": profile.get("term_objects", []),
            "topic_axes": profile.get("topic_axes", []),
            "core_tokens": profile.get("core_tokens", []),
            "query_rewrite": profile.get("query_rewrite", {}),
        },
        "source_anchors": selected_anchors,
        "unused_accepted_anchor_candidates": unused_accepted,
        "rejected_anchor_candidates": rejected_candidates,
        "_meta": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_count": len(sources),
            "inspected_source_count": min(max_sources, len(sources)),
            "accepted_candidate_count": len(candidate_anchors),
            "rejected_candidate_count": len(rejected_candidates),
            "selected_anchor_count": len(selected_anchors),
            "config": cfg,
        },
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def default_output_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"source_anchor_pack_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/generation.yaml"))
    parser.add_argument("--source_pack", type=Path, default=Path("outputs/source_packs/latest_source_pack.json"))
    parser.add_argument("--output_path", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    source_pack = load_json(args.source_pack)

    cfg = get_source_anchor_config(config)
    output_dir = Path(cfg.get("output_dir", "outputs/source_packs"))
    output_path = args.output_path or default_output_path(output_dir)

    pack = build_source_anchors(source_pack=source_pack, config=config)
    pack["_meta"]["source_pack"] = str(args.source_pack)
    pack["_meta"]["output_path"] = str(output_path)

    write_json(output_path, pack)

    latest_path = Path(cfg.get("latest_path", "outputs/source_packs/latest_source_anchor_pack.json"))
    write_json(latest_path, pack)

    print(f"Wrote source anchor pack: {output_path}")
    print(f"Wrote latest source anchor pack: {latest_path}")
    print(f"Selected anchors: {len(pack.get('source_anchors', []))}")
    print(f"Rejected candidates: {len(pack.get('rejected_anchor_candidates', []))}")

    for anchor in pack.get("source_anchors", []):
        print(
            f"- A{anchor.get('anchor_id')}: "
            f"rank={anchor.get('source_rank')} | "
            f"role={anchor.get('source_role')} | "
            f"score={anchor.get('anchor_score')} | "
            f"title={anchor.get('title')}"
        )

    rejected_preview = pack.get("rejected_anchor_candidates", [])[:8]
    if rejected_preview:
        print("Rejected preview:")
        for item in rejected_preview:
            print(
                f"- rank={item.get('source_rank')} | "
                f"score={item.get('anchor_score')} | "
                f"reasons={item.get('rejection_reasons')} | "
                f"title={item.get('title')}"
            )

    unused_preview = pack.get("unused_accepted_anchor_candidates", [])[:5]
    if unused_preview:
        print("Unused accepted preview:")
        for item in unused_preview:
            extra = item.get("diversity_rejection_reason", "over_max_anchors")
            print(
                f"- rank={item.get('source_rank')} | "
                f"score={item.get('anchor_score')} | "
                f"reason={extra} | "
                f"title={item.get('title')}"
            )


if __name__ == "__main__":
    main()
