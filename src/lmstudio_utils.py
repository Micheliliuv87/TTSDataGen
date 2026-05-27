# src/lmstudio_utils.py

from __future__ import annotations

from typing import List

from openai import OpenAI


def make_lmstudio_client(
    base_url: str = "http://localhost:1234/v1",
    api_key: str = "lm-studio",
    timeout_seconds: int = 60,
) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout_seconds,
    )


def list_lmstudio_model_ids(
    base_url: str = "http://localhost:1234/v1",
    api_key: str = "lm-studio",
    timeout_seconds: int = 30,
) -> List[str]:
    client = make_lmstudio_client(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )
    models = client.models.list()
    return [model.id for model in models.data]


def assert_lmstudio_model_available(
    model: str,
    base_url: str = "http://localhost:1234/v1",
    api_key: str = "lm-studio",
    timeout_seconds: int = 30,
) -> None:
    model_ids = list_lmstudio_model_ids(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )

    if model not in model_ids:
        available = "\n".join(f"  - {model_id}" for model_id in model_ids)
        raise RuntimeError(
            f"LM Studio model not available: {model}\n\n"
            f"Available models:\n{available}\n\n"
            "Check configs/rag.yaml or configs/generation.yaml. "
            "The model value must match one of the IDs returned by "
            "curl http://localhost:1234/v1/models"
        )