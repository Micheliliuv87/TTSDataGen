# src/run_pipeline.py

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_command(cmd: list[str]) -> None:
    print("\nRunning:")
    print(" ".join(cmd))
    print("=" * 80)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--rag_config", type=Path, default=Path("configs/rag.yaml"))
    parser.add_argument(
        "--generation_config",
        type=Path,
        default=Path("configs/generation.yaml"),
    )
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--language", type=str, default=None)
    parser.add_argument("--extra_instructions", type=str, default="")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    source_pack_path = Path("outputs/source_packs") / f"source_pack_pipeline_{timestamp}.json"
    dialogue_path = Path("outputs/dialogues") / f"dialogue_pipeline_{timestamp}.md"

    source_pack_path.parent.mkdir(parents=True, exist_ok=True)
    dialogue_path.parent.mkdir(parents=True, exist_ok=True)

    retrieve_cmd = [
        sys.executable,
        "-m",
        "src.retrieve",
        "--config",
        str(args.rag_config),
        "--query",
        args.query,
        "--output_path",
        str(source_pack_path),
    ]

    run_command(retrieve_cmd)

    generate_cmd = [
        sys.executable,
        "-m",
        "src.generate_dialogue",
        "--config",
        str(args.generation_config),
        "--source_pack",
        str(source_pack_path),
        "--output_path",
        str(dialogue_path),
        "--save_prompt",
    ]

    if args.turns is not None:
        generate_cmd.extend(["--turns", str(args.turns)])

    if args.language is not None:
        generate_cmd.extend(["--language", args.language])

    if args.extra_instructions:
        generate_cmd.extend(["--extra_instructions", args.extra_instructions])

    run_command(generate_cmd)

    latest_source_pack = Path("outputs/source_packs/latest_source_pack.json")
    latest_dialogue = Path("outputs/dialogues/latest_dialogue.md")

    shutil.copyfile(source_pack_path, latest_source_pack)
    shutil.copyfile(dialogue_path, latest_dialogue)

    print("\nPipeline complete.")
    print(f"Source pack: {source_pack_path}")
    print(f"Dialogue: {dialogue_path}")
    print(f"Latest source pack: {latest_source_pack}")
    print(f"Latest dialogue: {latest_dialogue}")


if __name__ == "__main__":
    main()