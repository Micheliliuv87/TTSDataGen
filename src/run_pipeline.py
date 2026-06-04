from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------
# Basic IO
# ---------------------------------------------------------------------


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


def run_command(cmd: List[str]) -> None:
    print("\nRunning:")
    print(shlex.join(cmd))
    print("=" * 80)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {shlex.join(cmd)}")


def ensure_dirs() -> None:
    Path("outputs/source_packs").mkdir(parents=True, exist_ok=True)
    Path("outputs/dialogues").mkdir(parents=True, exist_ok=True)
    Path("outputs/expansions").mkdir(parents=True, exist_ok=True)
    Path("outputs/polishes").mkdir(parents=True, exist_ok=True)
    Path("outputs/pipeline_runs").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Summaries / assertions
# ---------------------------------------------------------------------


def summarize_source_pack(path: Path) -> Dict[str, Any]:
    pack = load_json(path)

    sources = pack.get("sources", []) or []
    if not isinstance(sources, list):
        sources = []

    coverage = pack.get("coverage", {}) or {}
    if not isinstance(coverage, dict):
        coverage = {}

    return {
        "path": str(path),
        "query": pack.get("user_query") or pack.get("query") or pack.get("original_query") or "",
        "source_count": len(sources),
        "coverage": coverage,
        "coverage_status": coverage.get("coverage_status", ""),
        "usable_source_count": coverage.get("usable_source_count", ""),
        "strong_source_count": coverage.get("strong_source_count", ""),
    }


def summarize_anchor_pack(path: Path) -> Dict[str, Any]:
    pack = load_json(path)

    anchors = pack.get("source_anchors", []) or []
    if not isinstance(anchors, list):
        anchors = []

    rejected = pack.get("rejected_anchor_candidates", []) or []
    if not isinstance(rejected, list):
        rejected = []

    meta = pack.get("_meta", {}) or {}
    if not isinstance(meta, dict):
        meta = {}

    return {
        "path": str(path),
        "anchor_count": len(anchors),
        "rejected_anchor_count": len(rejected),
        "meta": {
            "source_count": meta.get("source_count", ""),
            "inspected_source_count": meta.get("inspected_source_count", ""),
            "accepted_candidate_count": meta.get("accepted_candidate_count", ""),
            "rejected_candidate_count": meta.get("rejected_candidate_count", ""),
            "selected_anchor_count": meta.get("selected_anchor_count", ""),
        },
        "anchors": [
            {
                "anchor_id": anchor.get("anchor_id", ""),
                "source_rank": anchor.get("source_rank", ""),
                "source_role": anchor.get("source_role", ""),
                "anchor_score": anchor.get("anchor_score", ""),
                "title": anchor.get("title", ""),
                "matched_terms": anchor.get("matched_terms", []),
                "strong_matched_terms": anchor.get("strong_matched_terms", []),
            }
            for anchor in anchors
            if isinstance(anchor, dict)
        ],
    }


def summarize_dialogue_meta(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
        }

    meta = load_json(path)
    return {
        "path": str(path),
        "exists": True,
        "model": meta.get("model", ""),
        "rounds": meta.get("rounds", ""),
        "total_dialogue_lines": meta.get("total_dialogue_lines", ""),
        "source_anchor_count": meta.get("source_anchor_count", ""),
        "raw_response_chars": meta.get("raw_response_chars", ""),
        "revised_body_chars": meta.get("revised_body_chars", ""),
        "expanded_body_chars": meta.get("expanded_body_chars", ""),
        "final_body_chars": meta.get("final_body_chars", ""),
        "final_output_chars": meta.get("final_output_chars", ""),
        "finish_reasons": meta.get("finish_reasons", ""),
        "parse_error_count": meta.get("parse_error_count", ""),
    }


def summarize_validation(path: Path) -> Dict[str, Any]:
    report = load_json(path)

    stats = report.get("stats", {}) or {}
    length = stats.get("length", {}) or {}

    return {
        "path": str(path),
        "passed": report.get("passed"),
        "mechanical_passed": report.get("mechanical_passed", report.get("passed")),
        "quality_passed": report.get("quality_passed"),
        "needs_rewrite": report.get("needs_rewrite"),
        "verdict": report.get("verdict"),
        "error_count": len(report.get("errors", []) or []),
        "warning_count": len(report.get("warnings", []) or []),
        "quality_blocking_warning_count": len(report.get("quality_blocking_warnings", []) or []),
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
            "short_line_count": length.get("short_line_count"),
            "short_line_ratio": length.get("short_line_ratio"),
            "developed_line_count": length.get("developed_line_count"),
            "developed_line_ratio": length.get("developed_line_ratio"),
        },
    }


def summarize_critique(path: Path) -> Dict[str, Any]:
    report = load_json(path)

    source_usage = report.get("source_usage", {}) or {}
    dialogue_quality = report.get("dialogue_quality", {}) or {}

    return {
        "path": str(path),
        "critic_version": report.get("critic_version"),
        "ready_for_rewrite": report.get("ready_for_rewrite"),
        "overall_score": report.get("overall_score"),
        "summary": report.get("summary", ""),
        "source_usage": {
            "status": source_usage.get("status", ""),
            "underused_anchor_ids": source_usage.get("underused_anchor_ids", []),
            "awkward_or_forced_anchor_ids": source_usage.get("awkward_or_forced_anchor_ids", []),
        },
        "dialogue_quality": {
            "density": dialogue_quality.get("density", ""),
            "progression": dialogue_quality.get("progression", ""),
            "repetition": dialogue_quality.get("repetition", ""),
            "naturalness": dialogue_quality.get("naturalness", ""),
            "speaker_balance": dialogue_quality.get("speaker_balance", ""),
        },
        "major_issue_count": len(report.get("major_issues", []) or []),
        "rewrite_priority_count": len(report.get("rewrite_priorities", []) or []),
    }


def assert_retrieval_has_sources(source_summary: Dict[str, Any], *, allow_empty_sources: bool) -> None:
    if allow_empty_sources:
        return

    source_count = int(source_summary.get("source_count", 0) or 0)

    if source_count > 0:
        return

    raise RuntimeError(
        "Retrieval returned zero selected sources, so generation was stopped.\n\n"
        f"Source pack: {source_summary.get('path')}\n"
        f"Coverage: {json.dumps(source_summary.get('coverage', {}), ensure_ascii=False, indent=2)}\n\n"
        "This usually means the current podcast vector DB does not contain enough relevant material "
        "for this query, or the retrieval/query rewrite stage failed. Try a broader query, improve "
        "query rewrite, lower retrieval thresholds, or add relevant source data."
    )


def assert_anchor_pack_has_anchors(anchor_summary: Dict[str, Any], *, allow_empty_anchors: bool) -> None:
    if allow_empty_anchors:
        return

    anchor_count = int(anchor_summary.get("anchor_count", 0) or 0)

    if anchor_count > 0:
        return

    raise RuntimeError(
        "Source anchor pack contains zero selected anchors, so generation was stopped.\n\n"
        f"Anchor pack: {anchor_summary.get('path')}\n"
        f"Summary: {json.dumps(anchor_summary.get('meta', {}), ensure_ascii=False, indent=2)}\n\n"
        "This means build_source_anchor_pack.py filtered all retrieved sources. Inspect "
        "`rejected_anchor_candidates` in the anchor pack. You may need to loosen thresholds in "
        "configs/generation.yaml or improve retrieval."
    )


def assert_validation_mechanical_passed(validation_summary: Dict[str, Any], *, stage_name: str) -> None:
    if validation_summary.get("mechanical_passed") is True:
        return

    raise RuntimeError(
        f"{stage_name} validation failed mechanically. Stopping pipeline.\n\n"
        f"Validation summary:\n{json.dumps(validation_summary, ensure_ascii=False, indent=2)}"
    )


def assert_validation_quality_passed(
    validation_summary: Dict[str, Any],
    *,
    stage_name: str,
    allow_quality_fail: bool,
) -> None:
    if allow_quality_fail:
        return

    if validation_summary.get("quality_passed") is True:
        return

    raise RuntimeError(
        f"{stage_name} validation did not pass quality checks. Stopping pipeline.\n\n"
        f"Validation summary:\n{json.dumps(validation_summary, ensure_ascii=False, indent=2)}"
    )


# ---------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------


def append_common_generation_options(
    cmd: List[str],
    *,
    rounds: Optional[int],
    turns: Optional[int],
    language: Optional[str],
    save_prompt: bool,
    dry_run_prompt: bool,
) -> None:
    if rounds is not None:
        cmd.extend(["--rounds", str(rounds)])

    if turns is not None:
        cmd.extend(["--turns", str(turns)])

    if language is not None:
        cmd.extend(["--language", language])

    if save_prompt:
        cmd.append("--save_prompt")

    if dry_run_prompt:
        cmd.append("--dry_run_prompt")


def run_validate_dialogue(
    *,
    dialogue_path: Path,
    meta_path: Path,
    require_source_appendix: bool,
) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "src.validate_dialogue",
        "--dialogue",
        str(dialogue_path),
        "--meta",
        str(meta_path),
    ]

    if require_source_appendix:
        cmd.append("--require_source_appendix")

    run_command(cmd)

    validation_path = dialogue_path.with_suffix(".validation.json")
    if not validation_path.exists():
        raise FileNotFoundError(f"Expected validation report was not written: {validation_path}")

    return validation_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--extra_instructions", type=str, default="")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["draft", "full"],
        default="full",
        help="draft: retrieve→anchor→generate→validate. full: draft + critique→expand→validate→polish→validate.",
    )

    parser.add_argument("--rag_config", type=Path, default=Path("configs/rag.yaml"))
    parser.add_argument("--generation_config", type=Path, default=Path("configs/generation.yaml"))
    parser.add_argument("--critic_config", type=Path, default=Path("configs/critic.yaml"))
    parser.add_argument("--expander_config", type=Path, default=Path("configs/expander.yaml"))
    parser.add_argument("--polisher_config", type=Path, default=Path("configs/polisher.yaml"))
    parser.add_argument("--prompt_templates", type=Path, default=None)

    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--turns", type=int, default=None)
    parser.add_argument("--language", type=str, default=None)

    parser.add_argument(
        "--skip_critique",
        action="store_true",
        help="Skip critique stage. Only useful for draft-like debugging.",
    )
    parser.add_argument(
        "--skip_expand",
        action="store_true",
        help="Skip expansion stage.",
    )
    parser.add_argument(
        "--skip_polish",
        action="store_true",
        help="Skip polish stage.",
    )

    parser.add_argument(
        "--allow_empty_sources",
        action="store_true",
        help="Continue even if retrieval selects zero sources. Not recommended for normal runs.",
    )
    parser.add_argument(
        "--allow_empty_anchors",
        action="store_true",
        help="Continue even if build_source_anchor_pack selects zero anchors. Not recommended.",
    )
    parser.add_argument(
        "--allow_raw_source_fallback",
        action="store_true",
        help="Pass through to generate_dialogue.py. Not recommended for normal runs.",
    )
    parser.add_argument(
        "--allow_expand_quality_fail",
        action="store_true",
        help="Continue to polish even if expanded dialogue still has quality warnings.",
    )
    parser.add_argument(
        "--allow_polish_quality_fail",
        action="store_true",
        help="Do not fail pipeline if polished dialogue has quality warnings.",
    )

    parser.add_argument("--save_prompt", action="store_true", default=True)
    parser.add_argument("--dry_run_prompt", action="store_true")

    args = parser.parse_args()

    ensure_dirs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    source_pack_path = Path(f"outputs/source_packs/source_pack_pipeline_{timestamp}.json")
    source_anchor_pack_path = Path(f"outputs/source_packs/source_anchor_pack_pipeline_{timestamp}.json")

    dialogue_path = Path(f"outputs/dialogues/dialogue_pipeline_{timestamp}.md")
    dialogue_meta_path = dialogue_path.with_suffix(".meta.json")
    generated_validation_path = dialogue_path.with_suffix(".validation.json")
    critique_report_path = dialogue_path.with_suffix(".critique.json")

    expanded_dialogue_path = Path(f"outputs/expansions/expanded_dialogue_pipeline_{timestamp}.md")
    expanded_meta_path = expanded_dialogue_path.with_suffix(".meta.json")
    expanded_validation_path = expanded_dialogue_path.with_suffix(".validation.json")

    polished_dialogue_path = Path(f"outputs/polishes/polished_expanded_dialogue_pipeline_{timestamp}.md")
    polished_meta_path = polished_dialogue_path.with_suffix(".meta.json")
    polished_validation_path = polished_dialogue_path.with_suffix(".validation.json")

    pipeline_meta_path = Path(f"outputs/pipeline_runs/pipeline_{timestamp}.json")

    pipeline_meta: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "timestamp": timestamp,
        "mode": args.mode,
        "query": args.query,
        "extra_instructions": args.extra_instructions,
        "paths": {
            "source_pack": str(source_pack_path),
            "source_anchor_pack": str(source_anchor_pack_path),
            "generated_dialogue": str(dialogue_path),
            "generated_validation": str(generated_validation_path),
            "critique_report": str(critique_report_path),
            "expanded_dialogue": str(expanded_dialogue_path),
            "expanded_validation": str(expanded_validation_path),
            "polished_dialogue": str(polished_dialogue_path),
            "polished_validation": str(polished_validation_path),
            "pipeline_meta": str(pipeline_meta_path),
        },
        "configs": {
            "rag_config": str(args.rag_config),
            "generation_config": str(args.generation_config),
            "critic_config": str(args.critic_config),
            "expander_config": str(args.expander_config),
            "polisher_config": str(args.polisher_config),
            "prompt_templates": str(args.prompt_templates) if args.prompt_templates else "",
        },
        "flags": {
            "skip_critique": args.skip_critique,
            "skip_expand": args.skip_expand,
            "skip_polish": args.skip_polish,
            "allow_empty_sources": args.allow_empty_sources,
            "allow_empty_anchors": args.allow_empty_anchors,
            "allow_raw_source_fallback": args.allow_raw_source_fallback,
            "allow_expand_quality_fail": args.allow_expand_quality_fail,
            "allow_polish_quality_fail": args.allow_polish_quality_fail,
            "save_prompt": args.save_prompt,
            "dry_run_prompt": args.dry_run_prompt,
        },
        "stages": {},
        "final": {},
    }

    try:
        # -------------------------------------------------------------
        # Stage 1: Retrieve
        # -------------------------------------------------------------

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

        source_summary = summarize_source_pack(source_pack_path)
        pipeline_meta["stages"]["retrieve"] = source_summary
        write_json(pipeline_meta_path, pipeline_meta)

        print("\nRetrieve summary:")
        print(json.dumps(source_summary, ensure_ascii=False, indent=2))

        assert_retrieval_has_sources(
            source_summary,
            allow_empty_sources=args.allow_empty_sources,
        )

        # -------------------------------------------------------------
        # Stage 2: Build source anchor pack
        # -------------------------------------------------------------

        build_anchor_cmd = [
            sys.executable,
            "-m",
            "src.build_source_anchor_pack",
            "--config",
            str(args.generation_config),
            "--source_pack",
            str(source_pack_path),
            "--output_path",
            str(source_anchor_pack_path),
        ]

        run_command(build_anchor_cmd)

        anchor_summary = summarize_anchor_pack(source_anchor_pack_path)
        pipeline_meta["stages"]["build_source_anchor_pack"] = anchor_summary
        write_json(pipeline_meta_path, pipeline_meta)

        print("\nSource anchor summary:")
        print(json.dumps(anchor_summary, ensure_ascii=False, indent=2))

        assert_anchor_pack_has_anchors(
            anchor_summary,
            allow_empty_anchors=args.allow_empty_anchors,
        )

        # -------------------------------------------------------------
        # Stage 3: Generate dialogue
        # -------------------------------------------------------------

        generate_cmd = [
            sys.executable,
            "-m",
            "src.generate_dialogue",
            "--config",
            str(args.generation_config),
            "--source_pack",
            str(source_pack_path),
            "--source_anchor_pack",
            str(source_anchor_pack_path),
            "--output_path",
            str(dialogue_path),
            "--extra_instructions",
            args.extra_instructions,
        ]

        if args.prompt_templates is not None:
            generate_cmd.extend(["--prompt_templates", str(args.prompt_templates)])

        append_common_generation_options(
            generate_cmd,
            rounds=args.rounds,
            turns=args.turns,
            language=args.language,
            save_prompt=args.save_prompt,
            dry_run_prompt=args.dry_run_prompt,
        )

        if args.allow_raw_source_fallback:
            generate_cmd.append("--allow_raw_source_fallback")

        if args.allow_empty_anchors:
            generate_cmd.append("--allow_empty_source_anchor_pack")

        run_command(generate_cmd)

        pipeline_meta["stages"]["generate_dialogue"] = {
            "output_path": str(dialogue_path),
            "prompt_path": str(dialogue_path.with_suffix(".prompt.json")),
            "meta_path": str(dialogue_meta_path),
            "meta_summary": summarize_dialogue_meta(dialogue_meta_path),
        }
        write_json(pipeline_meta_path, pipeline_meta)

        if args.dry_run_prompt:
            pipeline_meta["status"] = "dry_run_prompt"
            pipeline_meta["final"] = {
                "stage": "generate_dialogue_prompt",
                "prompt_path": str(dialogue_path.with_suffix(".prompt.json")),
                "pipeline_meta": str(pipeline_meta_path),
            }
            write_json(pipeline_meta_path, pipeline_meta)
            print("\nDry-run prompt completed.")
            print(f"Prompt:        {dialogue_path.with_suffix('.prompt.json')}")
            print(f"Pipeline meta: {pipeline_meta_path}")
            return

        # -------------------------------------------------------------
        # Stage 4: Validate generated dialogue
        # -------------------------------------------------------------

        generated_validation_path = run_validate_dialogue(
            dialogue_path=dialogue_path,
            meta_path=dialogue_meta_path,
            require_source_appendix=True,
        )

        generated_validation_summary = summarize_validation(generated_validation_path)
        pipeline_meta["stages"]["validate_generated"] = generated_validation_summary
        write_json(pipeline_meta_path, pipeline_meta)

        print("\nGenerated validation summary:")
        print(json.dumps(generated_validation_summary, ensure_ascii=False, indent=2))

        assert_validation_mechanical_passed(
            generated_validation_summary,
            stage_name="Generated dialogue",
        )

        if args.mode == "draft":
            pipeline_meta["status"] = "success"
            pipeline_meta["final"] = {
                "stage": "generated_dialogue",
                "dialogue": str(dialogue_path),
                "validation": str(generated_validation_path),
                "quality_passed": generated_validation_summary.get("quality_passed"),
                "pipeline_meta": str(pipeline_meta_path),
            }
            write_json(pipeline_meta_path, pipeline_meta)

            print("\nDraft pipeline completed successfully.")
            print(f"Source pack:        {source_pack_path}")
            print(f"Source anchor pack: {source_anchor_pack_path}")
            print(f"Dialogue:           {dialogue_path}")
            print(f"Validation:         {generated_validation_path}")
            print(f"Pipeline meta:      {pipeline_meta_path}")
            return

        # -------------------------------------------------------------
        # Stage 5: Critique generated dialogue
        # -------------------------------------------------------------

        current_dialogue_path = dialogue_path
        current_meta_path = dialogue_meta_path
        current_validation_path = generated_validation_path
        current_stage = "generated_dialogue"

        if not args.skip_critique and not args.skip_expand:
            critique_cmd = [
                sys.executable,
                "-m",
                "src.critique_dialogue",
                "--config",
                str(args.critic_config),
                "--dialogue",
                str(dialogue_path),
                "--source_anchor_pack",
                str(source_anchor_pack_path),
                "--validation_report",
                str(generated_validation_path),
            ]

            if args.save_prompt:
                critique_cmd.append("--save_prompt")

            run_command(critique_cmd)

            if not critique_report_path.exists():
                raise FileNotFoundError(f"Expected critique report was not written: {critique_report_path}")

            critique_summary = summarize_critique(critique_report_path)
            pipeline_meta["stages"]["critique_generated"] = critique_summary
            write_json(pipeline_meta_path, pipeline_meta)

            print("\nCritique summary:")
            print(json.dumps(critique_summary, ensure_ascii=False, indent=2))

        elif args.skip_critique and not args.skip_expand:
            raise RuntimeError(
                "--skip_critique cannot be used with expansion, because expand_dialogue.py requires a critique report."
            )

        # -------------------------------------------------------------
        # Stage 6: Expand dialogue
        # -------------------------------------------------------------

        if not args.skip_expand:
            expand_cmd = [
                sys.executable,
                "-m",
                "src.expand_dialogue",
                "--config",
                str(args.expander_config),
                "--dialogue",
                str(dialogue_path),
                "--source_anchor_pack",
                str(source_anchor_pack_path),
                "--validation_report",
                str(generated_validation_path),
                "--critique_report",
                str(critique_report_path),
                "--output_path",
                str(expanded_dialogue_path),
            ]

            if args.save_prompt:
                expand_cmd.append("--save_prompt")

            run_command(expand_cmd)

            pipeline_meta["stages"]["expand_dialogue"] = {
                "output_path": str(expanded_dialogue_path),
                "prompt_path": str(expanded_dialogue_path.with_suffix(".prompt.json")),
                "meta_path": str(expanded_meta_path),
                "meta_summary": summarize_dialogue_meta(expanded_meta_path),
            }
            write_json(pipeline_meta_path, pipeline_meta)

            # ---------------------------------------------------------
            # Stage 7: Validate expanded dialogue
            # ---------------------------------------------------------

            expanded_validation_path = run_validate_dialogue(
                dialogue_path=expanded_dialogue_path,
                meta_path=expanded_meta_path,
                require_source_appendix=True,
            )

            expanded_validation_summary = summarize_validation(expanded_validation_path)
            pipeline_meta["stages"]["validate_expanded"] = expanded_validation_summary
            write_json(pipeline_meta_path, pipeline_meta)

            print("\nExpanded validation summary:")
            print(json.dumps(expanded_validation_summary, ensure_ascii=False, indent=2))

            assert_validation_mechanical_passed(
                expanded_validation_summary,
                stage_name="Expanded dialogue",
            )
            assert_validation_quality_passed(
                expanded_validation_summary,
                stage_name="Expanded dialogue",
                allow_quality_fail=args.allow_expand_quality_fail,
            )

            current_dialogue_path = expanded_dialogue_path
            current_meta_path = expanded_meta_path
            current_validation_path = expanded_validation_path
            current_stage = "expanded_dialogue"

        # -------------------------------------------------------------
        # Stage 8: Polish dialogue
        # -------------------------------------------------------------

        if not args.skip_polish:
            if current_stage == "generated_dialogue":
                polished_dialogue_path = Path(f"outputs/polishes/polished_dialogue_pipeline_{timestamp}.md")
                polished_meta_path = polished_dialogue_path.with_suffix(".meta.json")
                polished_validation_path = polished_dialogue_path.with_suffix(".validation.json")

            polish_cmd = [
                sys.executable,
                "-m",
                "src.polish_dialogue",
                "--config",
                str(args.polisher_config),
                "--dialogue",
                str(current_dialogue_path),
                "--validation_report",
                str(current_validation_path),
                "--output_path",
                str(polished_dialogue_path),
            ]

            if args.save_prompt:
                polish_cmd.append("--save_prompt")

            run_command(polish_cmd)

            pipeline_meta["stages"]["polish_dialogue"] = {
                "input_stage": current_stage,
                "input_dialogue": str(current_dialogue_path),
                "output_path": str(polished_dialogue_path),
                "prompt_path": str(polished_dialogue_path.with_suffix(".prompt.json")),
                "meta_path": str(polished_meta_path),
                "meta_summary": summarize_dialogue_meta(polished_meta_path),
            }
            write_json(pipeline_meta_path, pipeline_meta)

            # ---------------------------------------------------------
            # Stage 9: Validate polished dialogue
            # ---------------------------------------------------------

            polished_validation_path = run_validate_dialogue(
                dialogue_path=polished_dialogue_path,
                meta_path=polished_meta_path,
                require_source_appendix=True,
            )

            polished_validation_summary = summarize_validation(polished_validation_path)
            pipeline_meta["stages"]["validate_polished"] = polished_validation_summary
            write_json(pipeline_meta_path, pipeline_meta)

            print("\nPolished validation summary:")
            print(json.dumps(polished_validation_summary, ensure_ascii=False, indent=2))

            assert_validation_mechanical_passed(
                polished_validation_summary,
                stage_name="Polished dialogue",
            )
            assert_validation_quality_passed(
                polished_validation_summary,
                stage_name="Polished dialogue",
                allow_quality_fail=args.allow_polish_quality_fail,
            )

            current_dialogue_path = polished_dialogue_path
            current_meta_path = polished_meta_path
            current_validation_path = polished_validation_path
            current_stage = "polished_dialogue"

        pipeline_meta["status"] = "success"
        pipeline_meta["final"] = {
            "stage": current_stage,
            "dialogue": str(current_dialogue_path),
            "meta": str(current_meta_path),
            "validation": str(current_validation_path),
            "pipeline_meta": str(pipeline_meta_path),
        }
        write_json(pipeline_meta_path, pipeline_meta)

        print("\nPipeline completed successfully.")
        print(f"Final stage:        {current_stage}")
        print(f"Final dialogue:     {current_dialogue_path}")
        print(f"Final validation:   {current_validation_path}")
        print(f"Source pack:        {source_pack_path}")
        print(f"Source anchor pack: {source_anchor_pack_path}")
        print(f"Pipeline meta:      {pipeline_meta_path}")

    except Exception as exc:
        pipeline_meta["status"] = "failed"
        pipeline_meta["error"] = str(exc)
        write_json(pipeline_meta_path, pipeline_meta)
        raise


if __name__ == "__main__":
    main()