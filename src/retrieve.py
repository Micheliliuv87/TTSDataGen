# src/retrieve.py

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import chromadb
import numpy as np
import yaml
from FlagEmbedding import BGEM3FlagModel

from src.query_rewrite import rewrite_query


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def preview_text(text: str, max_chars: int = 500) -> str:
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def normalize_title_key(title: Any) -> str:
    if not isinstance(title, str):
        return ""

    s = title.lower().strip()
    s = re.sub(r"\(\s*20\d{2}\s*\)", "", s)
    s = re.sub(r"\b20\d{2}\b", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def flatten_chroma_results(
    results: Dict[str, Any],
    query_variants: List[str],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    ids_by_query = results.get("ids", [])
    docs_by_query = results.get("documents", [])
    metas_by_query = results.get("metadatas", [])
    distances_by_query = results.get("distances", [])

    for q_idx, query_text in enumerate(query_variants):
        ids = ids_by_query[q_idx] if q_idx < len(ids_by_query) else []
        docs = docs_by_query[q_idx] if q_idx < len(docs_by_query) else []
        metas = metas_by_query[q_idx] if q_idx < len(metas_by_query) else []
        distances = distances_by_query[q_idx] if q_idx < len(distances_by_query) else []

        for rank, chunk_id in enumerate(ids):
            metadata = metas[rank] or {}
            document = docs[rank] or ""
            distance = float(distances[rank])

            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "distance": distance,
                    "matched_query": query_text,
                    "matched_query_index": q_idx,
                    "document": document,
                    "metadata": metadata,
                }
            )

    return candidates


def merge_duplicate_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}

    for cand in candidates:
        chunk_id = cand["chunk_id"]

        if chunk_id not in by_id:
            new_cand = dict(cand)
            new_cand["matched_queries"] = [cand["matched_query"]]
            by_id[chunk_id] = new_cand
            continue

        old = by_id[chunk_id]

        if cand["distance"] < old["distance"]:
            old.update(cand)

        if cand["matched_query"] not in old["matched_queries"]:
            old["matched_queries"].append(cand["matched_query"])

    return list(by_id.values())


def select_final_sources(
    candidates: List[Dict[str, Any]],
    final_top_k: int,
    max_chunks_per_doc: int,
    max_chunks_per_podcast: int,
    max_chunks_per_title: int,
) -> List[Dict[str, Any]]:
    ranked = sorted(candidates, key=lambda x: x["distance"])

    selected: List[Dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    seen_content_hashes: set[str] = set()

    doc_counts: Counter[str] = Counter()
    podcast_counts: Counter[str] = Counter()
    title_counts: Counter[str] = Counter()

    for cand in ranked:
        if len(selected) >= final_top_k:
            break

        chunk_id = cand["chunk_id"]
        metadata = cand.get("metadata") or {}

        if chunk_id in seen_chunk_ids:
            continue

        content_hash = str(metadata.get("chunk_content_hash") or "")
        if content_hash and content_hash in seen_content_hashes:
            continue

        doc_id = str(metadata.get("doc_id") or "")
        podcast_slug = str(metadata.get("podcast_slug") or "")
        title_key = normalize_title_key(metadata.get("title"))

        if doc_id and doc_counts[doc_id] >= max_chunks_per_doc:
            continue

        if podcast_slug and podcast_counts[podcast_slug] >= max_chunks_per_podcast:
            continue

        if title_key and title_counts[title_key] >= max_chunks_per_title:
            continue

        selected.append(cand)

        seen_chunk_ids.add(chunk_id)
        if content_hash:
            seen_content_hashes.add(content_hash)

        if doc_id:
            doc_counts[doc_id] += 1
        if podcast_slug:
            podcast_counts[podcast_slug] += 1
        if title_key:
            title_counts[title_key] += 1

    return selected

def assess_coverage(
    selected: List[Dict[str, Any]],
    weak_relevance_distance: float,
    min_sources_required: int,
) -> Dict[str, Any]:
    usable_source_count = len(selected)
    strong_source_count = sum(
        1 for cand in selected
        if cand["distance"] <= weak_relevance_distance
    )

    if usable_source_count == 0:
        coverage_status = "none"
    elif usable_source_count < min_sources_required:
        coverage_status = "weak"
    elif strong_source_count >= min_sources_required:
        coverage_status = "strong"
    else:
        coverage_status = "medium"

    return {
        "coverage_status": coverage_status,
        "usable_source_count": usable_source_count,
        "strong_source_count": strong_source_count,
    }
    
def make_source_pack(
    user_query: str,
    rewrite_result: Dict[str, Any],
    selected: List[Dict[str, Any]],
    params: Dict[str, Any],
    coverage: Dict[str, Any],
) -> Dict[str, Any]:
    sources: List[Dict[str, Any]] = []

    for rank, cand in enumerate(selected, start=1):
        metadata = cand.get("metadata") or {}

        sources.append(
            {
                "rank": rank,
                "distance": cand["distance"],
                "chunk_id": cand["chunk_id"],
                "doc_id": metadata.get("doc_id", ""),
                "title": metadata.get("title", ""),
                "podcast_slug": metadata.get("podcast_slug", ""),
                "url": metadata.get("url", ""),
                "start_timestamp": metadata.get("start_timestamp", ""),
                "end_timestamp": metadata.get("end_timestamp", ""),
                "chunk_index": metadata.get("chunk_index", ""),
                "char_count": metadata.get("char_count", ""),
                "matched_queries": cand.get(
                    "matched_queries",
                    [cand.get("matched_query", "")],
                ),
                "text": cand.get("document", ""),
            }
        )

    return {
        "user_query": user_query,
        "query_rewrite": rewrite_result,
        "retrieval_params": params,
        "coverage": coverage,
        "sources": sources,
    }


def print_results(selected: List[Dict[str, Any]], preview_chars: int) -> None:
    print("\nFinal selected sources:")
    print("=" * 80)

    for rank, cand in enumerate(selected, start=1):
        metadata = cand.get("metadata") or {}
        document = cand.get("document") or ""
        matched_queries = cand.get("matched_queries", [cand.get("matched_query", "")])

        print(f"\n[{rank}] distance={cand['distance']:.4f}")
        print(f"chunk_id: {cand['chunk_id']}")
        print(f"title: {metadata.get('title', '')}")
        print(f"podcast_slug: {metadata.get('podcast_slug', '')}")
        print(f"url: {metadata.get('url', '')}")
        print(
            f"timestamp: {metadata.get('start_timestamp', '')} - {metadata.get('end_timestamp', '')}"
        )
        print(f"matched_queries: {len(matched_queries)}")
        print(f"preview: {preview_text(document, max_chars=preview_chars)}")
        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    parser.add_argument("--query", type=str, required=True)

    parser.add_argument("--raw_top_k", type=int, default=None)
    parser.add_argument("--final_top_k", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--max_chunks_per_doc", type=int, default=None)
    parser.add_argument("--max_chunks_per_podcast", type=int, default=None)
    parser.add_argument("--max_chunks_per_title", type=int, default=None)

    parser.add_argument("--collection_name", type=str, default=None)
    parser.add_argument("--persist_dir", type=Path, default=None)
    parser.add_argument("--embedding_model", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=None)

    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--disable_rewrite", action="store_true")

    parser.add_argument("--output_path", type=Path, default=None)
    parser.add_argument("--preview_chars", type=int, default=None)

    args = parser.parse_args()

    config = load_yaml(args.config)

    embedding_cfg = config.get("embedding", {})
    chroma_cfg = config.get("chroma", {})
    retrieval_cfg = config.get("retrieval", {})

    persist_dir = args.persist_dir or Path(
        chroma_cfg.get("persist_dir", "vector_db/chroma_content")
    )
    collection_name = args.collection_name or chroma_cfg.get(
        "collection_name",
        "happyscribe_bge_m3",
    )
    embedding_model = args.embedding_model or embedding_cfg.get(
        "model_name",
        "BAAI/bge-m3",
    )
    max_length = args.max_length or int(embedding_cfg.get("max_length", 1024))

    raw_top_k = args.raw_top_k or int(retrieval_cfg.get("raw_top_k_per_query", 30))

    final_top_k = (
        args.top_k
        or args.final_top_k
        or int(retrieval_cfg.get("final_top_k", 8))
    )

    max_chunks_per_doc = args.max_chunks_per_doc or int(
        retrieval_cfg.get("max_chunks_per_doc", 2)
    )
    max_chunks_per_podcast = args.max_chunks_per_podcast or int(
        retrieval_cfg.get("max_chunks_per_podcast", 3)
    )
    max_chunks_per_title = args.max_chunks_per_title or int(
        retrieval_cfg.get("max_chunks_per_title", 2)
    )

    preview_chars = args.preview_chars or int(retrieval_cfg.get("preview_chars", 500))
    
    min_relevance_distance = float(
        retrieval_cfg.get("min_relevance_distance", 0.42)
    )

    weak_relevance_distance = float(
        retrieval_cfg.get("weak_relevance_distance", 0.40)
    )

    min_sources_required = int(
        retrieval_cfg.get("min_sources_required", 3)
    )

    if args.use_fp16 and args.no_fp16:
        raise ValueError("Use only one of --use_fp16 or --no_fp16")

    if args.use_fp16:
        use_fp16 = True
    elif args.no_fp16:
        use_fp16 = False
    else:
        use_fp16 = bool(embedding_cfg.get("use_fp16", False))

    if args.disable_rewrite:
        rewrite_result = {
            "original_query": args.query,
            "retrieval_queries": [args.query],
            "rewrite_used": False,
            "fallback_reason": "disabled_by_cli",
        }
    else:
        rewrite_result = rewrite_query(args.query, config)

    query_variants = rewrite_result.get("retrieval_queries", [args.query])
    query_variants = [str(q).strip() for q in query_variants if str(q).strip()]

    if not query_variants:
        query_variants = [args.query]

    print("Retrieve config:")
    print(
        json.dumps(
            {
                "persist_dir": str(persist_dir),
                "collection_name": collection_name,
                "embedding_model": embedding_model,
                "max_length": max_length,
                "use_fp16": use_fp16,
                "raw_top_k_per_query": raw_top_k,
                "final_top_k": final_top_k,
                "max_chunks_per_doc": max_chunks_per_doc,
                "max_chunks_per_podcast": max_chunks_per_podcast,
                "max_chunks_per_title": max_chunks_per_title,
                "query": args.query,
                "query_rewrite": rewrite_result,
                "min_relevance_distance": min_relevance_distance,
                "weak_relevance_distance": weak_relevance_distance,
                "min_sources_required": min_sources_required,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)

    print(f"Collection count: {collection.count()}")

    model = BGEM3FlagModel(
        embedding_model,
        use_fp16=use_fp16,
    )

    outputs = model.encode(
        query_variants,
        batch_size=len(query_variants),
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )

    query_embeddings = outputs["dense_vecs"]

    if isinstance(query_embeddings, np.ndarray):
        query_embeddings = query_embeddings.tolist()

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=raw_top_k,
        include=["documents", "metadatas", "distances"],
    )

    candidates = flatten_chroma_results(results, query_variants)
    merged_candidates = merge_duplicate_candidates(candidates)

    strong_candidates = [
        c for c in merged_candidates
        if c["distance"] <= min_relevance_distance
    ]

    selected = select_final_sources(
        candidates=strong_candidates,
        final_top_k=final_top_k,
        max_chunks_per_doc=max_chunks_per_doc,
        max_chunks_per_podcast=max_chunks_per_podcast,
        max_chunks_per_title=max_chunks_per_title,
    )
    
    params = {
        "raw_top_k_per_query": raw_top_k,
        "final_top_k": final_top_k,
        "max_chunks_per_doc": max_chunks_per_doc,
        "max_chunks_per_podcast": max_chunks_per_podcast,
        "max_chunks_per_title": max_chunks_per_title,
        "min_relevance_distance": min_relevance_distance,
        "weak_relevance_distance": weak_relevance_distance,
        "min_sources_required": min_sources_required,
    }

    coverage = assess_coverage(
        selected=selected,
        weak_relevance_distance=weak_relevance_distance,
        min_sources_required=min_sources_required,
    )

    source_pack = make_source_pack(
        user_query=args.query,
        rewrite_result=rewrite_result,
        selected=selected,
        params=params,
        coverage=coverage,
    )
    
    print("\nCoverage:")
    print(json.dumps(coverage, ensure_ascii=False, indent=2))

    print_results(selected, preview_chars=preview_chars)

    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(
            json.dumps(source_pack, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote source pack: {args.output_path}")


if __name__ == "__main__":
    main()