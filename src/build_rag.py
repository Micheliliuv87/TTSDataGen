# src/build_rag.py

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import chromadb
import numpy as np
import yaml
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {e}") from e


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def batched(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def sanitize_metadata_value(value: Any) -> str | int | float | bool:
    """
    Chroma metadata should be scalar values.
    Convert None / lists / dicts into safe strings.
    """
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def make_metadata(chunk: Dict[str, Any]) -> Dict[str, str | int | float | bool]:
    exclude = {"text"}
    metadata: Dict[str, str | int | float | bool] = {}

    for key, value in chunk.items():
        if key in exclude:
            continue
        metadata[key] = sanitize_metadata_value(value)

    return metadata


def get_existing_ids(collection: Any, ids: List[str]) -> set[str]:
    if not ids:
        return set()

    try:
        result = collection.get(ids=ids, include=[])
        return set(result.get("ids", []))
    except Exception:
        # If collection is empty or Chroma behaves differently across versions,
        # fail open and let add() handle true duplicates.
        return set()


def load_chunks(
    chunks_path: Path,
    max_chunks: Optional[int],
    start_offset: int,
) -> Iterable[Dict[str, Any]]:
    seen = 0
    yielded = 0

    for chunk in read_jsonl(chunks_path):
        if seen < start_offset:
            seen += 1
            continue

        seen += 1

        if max_chunks is not None and yielded >= max_chunks:
            break

        text = chunk.get("text")
        chunk_id = chunk.get("chunk_id")

        if not isinstance(text, str) or not text.strip():
            continue

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            continue

        yielded += 1
        yield chunk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    parser.add_argument("--chunks_path", type=Path, default=None)
    parser.add_argument("--persist_dir", type=Path, default=None)
    parser.add_argument("--collection_name", type=str, default=None)
    parser.add_argument("--embedding_model", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--no_fp16", action="store_true")
    parser.add_argument("--max_chunks", type=int, default=None)
    parser.add_argument("--start_offset", type=int, default=0)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no_skip_existing", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)

    embedding_cfg = config.get("embedding", {})
    chroma_cfg = config.get("chroma", {})
    input_cfg = config.get("input", {})
    build_cfg = config.get("build", {})

    chunks_path = args.chunks_path or Path(input_cfg.get("chunks_path", "data/processed/rag/podcasts/happyscribe/chunks.jsonl"))
    persist_dir = args.persist_dir or Path(chroma_cfg.get("persist_dir", "vector_db/chroma_content"))
    collection_name = args.collection_name or chroma_cfg.get("collection_name", "happyscribe_bge_m3")
    embedding_model = args.embedding_model or embedding_cfg.get("model_name", "BAAI/bge-m3")
    batch_size = args.batch_size or int(embedding_cfg.get("batch_size", 16))
    max_length = args.max_length or int(embedding_cfg.get("max_length", 1024))

    if args.use_fp16 and args.no_fp16:
        raise ValueError("Use only one of --use_fp16 or --no_fp16")

    if args.use_fp16:
        use_fp16 = True
    elif args.no_fp16:
        use_fp16 = False
    else:
        use_fp16 = bool(embedding_cfg.get("use_fp16", False))

    skip_existing = bool(build_cfg.get("skip_existing", True))
    if args.no_skip_existing:
        skip_existing = False

    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks_path not found: {chunks_path}")

    persist_dir.mkdir(parents=True, exist_ok=True)

    print("Build RAG config:")
    print(
        json.dumps(
            {
                "chunks_path": str(chunks_path),
                "persist_dir": str(persist_dir),
                "collection_name": collection_name,
                "embedding_model": embedding_model,
                "batch_size": batch_size,
                "max_length": max_length,
                "use_fp16": use_fp16,
                "max_chunks": args.max_chunks,
                "start_offset": args.start_offset,
                "skip_existing": skip_existing,
                "reset": args.reset,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    client = chromadb.PersistentClient(path=str(persist_dir))

    if args.reset:
        try:
            client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            print(f"No existing collection to delete: {collection_name}")

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": chroma_cfg.get("distance_metric", "cosine")},
    )

    print(f"Loading embedding model: {embedding_model}")
    model = BGEM3FlagModel(
        embedding_model,
        use_fp16=use_fp16,
    )

    buffer: List[Dict[str, Any]] = []
    total_seen = 0
    total_added = 0
    total_skipped_existing = 0
    total_skipped_empty = 0
    start_time = time.time()

    chunk_iter = load_chunks(
        chunks_path=chunks_path,
        max_chunks=args.max_chunks,
        start_offset=args.start_offset,
    )

    for chunk in tqdm(chunk_iter, desc="Reading chunks"):
        buffer.append(chunk)

        if len(buffer) < batch_size:
            continue

        ids = [c["chunk_id"] for c in buffer]
        total_seen += len(buffer)

        if skip_existing:
            existing_ids = get_existing_ids(collection, ids)
            if existing_ids:
                new_buffer = [c for c in buffer if c["chunk_id"] not in existing_ids]
                total_skipped_existing += len(buffer) - len(new_buffer)
                buffer = new_buffer

        if buffer:
            texts = [c["text"] for c in buffer]
            ids = [c["chunk_id"] for c in buffer]
            metadatas = [make_metadata(c) for c in buffer]

            outputs = model.encode(
                texts,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )

            embeddings = outputs["dense_vecs"]

            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()

            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            total_added += len(buffer)

        buffer = []

    # final partial batch
    if buffer:
        ids = [c["chunk_id"] for c in buffer]
        total_seen += len(buffer)

        if skip_existing:
            existing_ids = get_existing_ids(collection, ids)
            if existing_ids:
                new_buffer = [c for c in buffer if c["chunk_id"] not in existing_ids]
                total_skipped_existing += len(buffer) - len(new_buffer)
                buffer = new_buffer

        if buffer:
            texts = [c["text"] for c in buffer]
            ids = [c["chunk_id"] for c in buffer]
            metadatas = [make_metadata(c) for c in buffer]

            outputs = model.encode(
                texts,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )

            embeddings = outputs["dense_vecs"]

            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()

            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

            total_added += len(buffer)

    elapsed = round(time.time() - start_time, 2)

    try:
        collection_count = collection.count()
    except Exception:
        collection_count = None

    summary = {
        "chunks_path": str(chunks_path),
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "batch_size": batch_size,
        "max_length": max_length,
        "use_fp16": use_fp16,
        "max_chunks": args.max_chunks,
        "start_offset": args.start_offset,
        "total_seen": total_seen,
        "total_added": total_added,
        "total_skipped_existing": total_skipped_existing,
        "total_skipped_empty": total_skipped_empty,
        "collection_count": collection_count,
        "elapsed_seconds": elapsed,
    }

    print("Build complete:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()