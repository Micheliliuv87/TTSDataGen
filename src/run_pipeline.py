from __future__ import annotations

import argparse
import json
import re
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


# ---------------------------------------------------------------------
# Pipeline result contract for UIs / wrappers
# ---------------------------------------------------------------------


def path_exists(path_value: Any) -> bool:
    if not path_value:
        return False

    try:
        return Path(str(path_value)).exists()
    except OSError:
        return False


def validation_status_from_summary(validation_summary: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(validation_summary, dict) or not validation_summary:
        return {
            "passed": None,
            "mechanical_passed": None,
            "quality_passed": None,
            "needs_rewrite": None,
            "verdict": "",
        }

    return {
        "passed": validation_summary.get("passed"),
        "mechanical_passed": validation_summary.get("mechanical_passed"),
        "quality_passed": validation_summary.get("quality_passed"),
        "needs_rewrite": validation_summary.get("needs_rewrite"),
        "verdict": validation_summary.get("verdict", ""),
    }


def make_display_artifact(
    *,
    stage: str,
    dialogue_path: Any,
    meta_path: Any = "",
    validation_path: Any = "",
    validation_summary: Optional[Dict[str, Any]] = None,
    is_official_final: bool = False,
    status_label: str,
    warning: str = "",
    reason: str = "",
) -> Dict[str, Any]:
    validation_summary = validation_summary or {}

    return {
        "stage": stage,
        "dialogue": str(dialogue_path) if dialogue_path else "",
        "meta": str(meta_path) if meta_path else "",
        "validation": str(validation_path) if validation_path else "",
        "is_official_final": is_official_final,
        "status_label": status_label,
        "warning": warning,
        "reason": reason,
        **validation_status_from_summary(validation_summary),
    }


def latest_expanded_candidate_stage(stages: Dict[str, Any]) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
    retry_indices: List[int] = []
    for key in stages:
        match = re.fullmatch(r"expand_dialogue_retry_(\d+)", key)
        if match:
            retry_indices.append(int(match.group(1)))

    for idx in sorted(retry_indices, reverse=True):
        stage_key = f"expand_dialogue_retry_{idx}"
        validation_key = f"validate_expanded_retry_{idx}"
        stage = stages.get(stage_key, {}) or {}
        validation = stages.get(validation_key, {}) or {}
        if stage.get("output_path") or path_exists(stage.get("output_path")):
            return stage_key, stage, validation

    stage = stages.get("expand_dialogue", {}) or {}
    validation = stages.get("validate_expanded", {}) or {}
    return "expand_dialogue", stage, validation


def choose_display_artifact(pipeline_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Select the best artifact that downstream UIs should display.

    This keeps UI code simple: the pipeline owns artifact precedence and
    status semantics. UI layers should not have to guess whether polished,
    expanded, or generated output exists after a failed run.
    """
    final = pipeline_meta.get("final", {}) or {}
    stages = pipeline_meta.get("stages", {}) or {}
    paths = pipeline_meta.get("paths", {}) or {}

    final_dialogue = final.get("dialogue")
    if final_dialogue:
        final_stage = str(final.get("stage") or "final")
        validation_path = final.get("validation", "")
        validation_summary = {}
        if final_stage == "generated_dialogue":
            validation_summary = stages.get("validate_generated", {}) or {}
        elif final_stage == "expanded_dialogue":
            _, _, validation_summary = latest_expanded_candidate_stage(stages)
        elif final_stage == "polished_dialogue":
            validation_summary = stages.get("validate_polished", {}) or {}

        return make_display_artifact(
            stage=final_stage,
            dialogue_path=final_dialogue,
            meta_path=final.get("meta", ""),
            validation_path=validation_path,
            validation_summary=validation_summary,
            is_official_final=True,
            status_label="最终通过版本",
            reason="pipeline_success_final",
        )

    polish_stage = stages.get("polish_dialogue", {}) or {}
    polished_dialogue = polish_stage.get("output_path")
    if not polished_dialogue and path_exists(paths.get("polished_dialogue")):
        polished_dialogue = paths.get("polished_dialogue")

    if polished_dialogue:
        polished_validation = (
            (stages.get("validate_polished", {}) or {}).get("path")
            or paths.get("polished_validation")
        )
        return make_display_artifact(
            stage="polished_dialogue",
            dialogue_path=polished_dialogue,
            meta_path=polish_stage.get("meta_path", ""),
            validation_path=polished_validation or "",
            validation_summary=stages.get("validate_polished", {}) or {},
            is_official_final=False,
            status_label="已生成但未通过最终质量检查",
            warning=(
                "Pipeline 最终质量检查未通过，但 polished markdown 已经生成。"
                "下面展示的是可检查的候选输出，不是正式通过版本。"
            ),
            reason="polished_candidate_available",
        )

    expanded_stage_key, expanded_stage, expanded_validation_summary = latest_expanded_candidate_stage(stages)
    expanded_dialogue = expanded_stage.get("output_path")
    if not expanded_dialogue and path_exists(paths.get("expanded_dialogue")):
        expanded_dialogue = paths.get("expanded_dialogue")

    if expanded_dialogue:
        expanded_validation = (
            expanded_validation_summary.get("path")
            or paths.get("expanded_validation")
        )
        return make_display_artifact(
            stage="expanded_dialogue",
            dialogue_path=expanded_dialogue,
            meta_path=expanded_stage.get("meta_path", ""),
            validation_path=expanded_validation or "",
            validation_summary=expanded_validation_summary,
            is_official_final=False,
            status_label="中间扩写版本",
            warning=(
                "Pipeline 没有产出正式 final 版本。下面展示的是 expanded 中间版本，"
                "需要人工检查。"
            ),
            reason=f"{expanded_stage_key}_candidate_available",
        )

    generated_stage = stages.get("generate_dialogue", {}) or {}
    generated_dialogue = generated_stage.get("output_path")
    if not generated_dialogue and path_exists(paths.get("generated_dialogue")):
        generated_dialogue = paths.get("generated_dialogue")

    if generated_dialogue:
        generated_validation = (
            (stages.get("validate_generated", {}) or {}).get("path")
            or paths.get("generated_validation")
        )
        return make_display_artifact(
            stage="generated_dialogue",
            dialogue_path=generated_dialogue,
            meta_path=generated_stage.get("meta_path", ""),
            validation_path=generated_validation or "",
            validation_summary=stages.get("validate_generated", {}) or {},
            is_official_final=False,
            status_label="初稿版本",
            warning=(
                "Pipeline 没有产出正式 final 版本。下面展示的是 generated 初稿，"
                "通常还需要扩写或润色。"
            ),
            reason="generated_candidate_available",
        )

    return make_display_artifact(
        stage="none",
        dialogue_path="",
        status_label="没有可展示输出",
        warning="没有找到 generated / expanded / polished markdown 输出。",
        reason="no_dialogue_artifact_available",
    )


def build_source_quality_summary(pipeline_meta: Dict[str, Any]) -> Dict[str, Any]:
    stages = pipeline_meta.get("stages", {}) or {}
    retrieve = stages.get("retrieve", {}) or {}
    anchor = stages.get("build_source_anchor_pack", {}) or {}

    coverage_status = str(retrieve.get("coverage_status") or "").lower()
    usable_source_count = int(retrieve.get("usable_source_count") or 0)
    strong_source_count = int(retrieve.get("strong_source_count") or 0)
    anchor_count = int(anchor.get("anchor_count") or 0)
    rejected_anchor_count = int(anchor.get("rejected_anchor_count") or 0)

    if coverage_status in {"high", "strong"} or strong_source_count > 0:
        level = "较高"
        message = "系统找到了较强相关素材，适合直接生成。"
    elif coverage_status == "medium" or usable_source_count >= 3 or anchor_count > 0:
        level = "中等"
        message = "结果可以生成，但素材可能不完全贴合主题，生成内容可能需要扩写或存在轻微噪声。"
    else:
        level = "较低"
        message = "当前数据库中相关素材不足。建议输入更具体的主题，或补充更多 source 数据。"

    return {
        "level": level,
        "message": message,
        "coverage_status": coverage_status or "unknown",
        "usable_source_count": usable_source_count,
        "strong_source_count": strong_source_count,
        "anchor_count": anchor_count,
        "rejected_anchor_count": rejected_anchor_count,
    }


def build_ui_summary(pipeline_meta: Dict[str, Any]) -> Dict[str, Any]:
    status = str(pipeline_meta.get("status") or "running")
    artifact = pipeline_meta.get("display_artifact", {}) or {}
    source_quality = pipeline_meta.get("source_quality", {}) or {}

    has_artifact = bool(artifact.get("dialogue"))
    quality_passed = artifact.get("quality_passed")
    mechanical_passed = artifact.get("mechanical_passed")

    if status == "running":
        severity = "info"
        message = "Pipeline 正在运行。页面可以刷新或重新打开，当前任务会通过 pipeline_meta 和 UI job 文件恢复。"
    elif status == "cancelled":
        severity = "warning"
        message = "Pipeline 已被用户取消。"
    elif status == "success":
        severity = "success"
        message = "Pipeline 已成功完成，当前展示的是最终通过版本。"
    elif has_artifact:
        severity = "warning"
        message = "Pipeline 最终检查未通过，但已生成候选输出，可以人工检查或下载。"
    elif source_quality.get("anchor_count", 0) == 0:
        severity = "error"
        message = "素材匹配度较低：当前数据库没有找到足够可靠的 source anchors，因此已停止生成。"
    else:
        severity = "error"
        message = "Pipeline 运行失败，且没有找到可展示的 dialogue artifact。请查看日志。"

    return {
        "status": status,
        "severity": severity,
        "message": message,
        "has_display_artifact": has_artifact,
        "display_stage": artifact.get("stage", "none"),
        "display_status_label": artifact.get("status_label", ""),
        "mechanical_passed": mechanical_passed,
        "quality_passed": quality_passed,
    }


def refresh_pipeline_meta_derived_fields(pipeline_meta: Dict[str, Any]) -> None:
    source_quality = build_source_quality_summary(pipeline_meta)
    display_artifact = choose_display_artifact(pipeline_meta)

    pipeline_meta["source_quality"] = source_quality
    pipeline_meta["display_artifact"] = display_artifact
    # Backward-compatible alias for older UI experiments.
    pipeline_meta["best_available_artifact"] = display_artifact
    pipeline_meta["ui_summary"] = build_ui_summary(pipeline_meta)


def write_pipeline_meta(path: Path, pipeline_meta: Dict[str, Any]) -> None:
    refresh_pipeline_meta_derived_fields(pipeline_meta)
    write_json(path, pipeline_meta)


def set_progress(
    pipeline_meta: Dict[str, Any],
    *,
    stage: str,
    percent: int,
    label: str,
    detail: str = "",
) -> None:
    bounded_percent = max(0, min(100, int(percent)))
    pipeline_meta["progress"] = {
        "stage": stage,
        "percent": bounded_percent,
        "label": label,
        "detail": detail,
        "is_estimated": True,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def update_progress(
    path: Path,
    pipeline_meta: Dict[str, Any],
    *,
    stage: str,
    percent: int,
    label: str,
    detail: str = "",
) -> None:
    set_progress(
        pipeline_meta,
        stage=stage,
        percent=percent,
        label=label,
        detail=detail,
    )
    write_pipeline_meta(path, pipeline_meta)
    print(f"\nProgress: {max(0, min(100, int(percent)))}% - {label}", flush=True)
    if detail:
        print(f"Progress detail: {detail}", flush=True)


def print_pipeline_footer(*, status: str, pipeline_meta_path: Path, pipeline_meta: Dict[str, Any]) -> None:
    refresh_pipeline_meta_derived_fields(pipeline_meta)
    artifact = pipeline_meta.get("display_artifact", {}) or {}
    ui_summary = pipeline_meta.get("ui_summary", {}) or {}

    print(f"\nPipeline result:    {status}")

    print(f"Pipeline meta:      {pipeline_meta_path}")
    print(f"Display stage:      {artifact.get('stage', 'none')}")
    print(f"Display dialogue:   {artifact.get('dialogue', '')}")
    print(f"Display validation: {artifact.get('validation', '')}")
    print(f"UI message:         {ui_summary.get('message', '')}")


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

def run_expand_and_validate(
    *,
    expander_config: Path,
    dialogue_path: Path,
    source_anchor_pack_path: Path,
    validation_report_path: Path,
    critique_report_path: Path,
    output_path: Path,
    save_prompt: bool,
) -> tuple[Path, Path, Path, Dict[str, Any]]:
    expand_cmd = [
        sys.executable,
        "-m",
        "src.expand_dialogue",
        "--config",
        str(expander_config),
        "--dialogue",
        str(dialogue_path),
        "--source_anchor_pack",
        str(source_anchor_pack_path),
        "--validation_report",
        str(validation_report_path),
        "--critique_report",
        str(critique_report_path),
        "--output_path",
        str(output_path),
    ]

    if save_prompt:
        expand_cmd.append("--save_prompt")

    run_command(expand_cmd)

    meta_path = output_path.with_suffix(".meta.json")

    validation_path = run_validate_dialogue(
        dialogue_path=output_path,
        meta_path=meta_path,
        require_source_appendix=True,
    )

    validation_summary = summarize_validation(validation_path)

    return output_path, meta_path, validation_path, validation_summary

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
        "--run_id",
        type=str,
        default=None,
        help=(
            "Optional stable run id used for output filenames, usually YYYYMMDD_HHMMSS. "
            "This is mainly for UI wrappers that need to know paths before the pipeline finishes."
        ),
    )

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
        "--max_expand_retries",
        type=int,
        default=1,
        help=(
            "Retry expansion this many times if expanded dialogue passes mechanically "
            "but fails quality checks. Default: 1."
        ),
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

    if args.run_id:
        run_id = args.run_id.strip()
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError(
                "--run_id may only contain letters, numbers, underscores, and hyphens."
            )
        timestamp = run_id
    else:
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
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "timestamp": timestamp,
        "run_id": timestamp,
        "status": "running",
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
            "max_expand_retries": args.max_expand_retries,
            "allow_polish_quality_fail": args.allow_polish_quality_fail,
            "save_prompt": args.save_prompt,
            "dry_run_prompt": args.dry_run_prompt,
        },
        "stages": {},
        "final": {},
    }

    # Write an initial running meta file before any expensive stage starts.
    # This lets Streamlit or future UIs reconnect after a refresh or mobile tab switch.
    update_progress(
        pipeline_meta_path,
        pipeline_meta,
        stage="queued",
        percent=0,
        label="任务已创建",
        detail="等待开始检索素材。",
    )
    print("\nPipeline started.")
    print(f"Run ID:             {timestamp}")
    print(f"Pipeline meta:      {pipeline_meta_path}")

    try:
        # -------------------------------------------------------------
        # Stage 1: Retrieve
        # -------------------------------------------------------------

        update_progress(
            pipeline_meta_path,
            pipeline_meta,
            stage="retrieve",
            percent=5,
            label="正在检索相关素材",
            detail="根据用户输入检索 podcast source pack。",
        )

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
        update_progress(
            pipeline_meta_path,
            pipeline_meta,
            stage="retrieve_done",
            percent=12,
            label="素材检索完成",
            detail="已生成 source pack，准备筛选 source anchors。",
        )

        print("\nRetrieve summary:")
        print(json.dumps(source_summary, ensure_ascii=False, indent=2))

        assert_retrieval_has_sources(
            source_summary,
            allow_empty_sources=args.allow_empty_sources,
        )

        # -------------------------------------------------------------
        # Stage 2: Build source anchor pack
        # -------------------------------------------------------------

        update_progress(
            pipeline_meta_path,
            pipeline_meta,
            stage="build_source_anchor_pack",
            percent=15,
            label="正在筛选核心素材片段",
            detail="从检索结果中选择可用于生成的 source anchors。",
        )

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
        update_progress(
            pipeline_meta_path,
            pipeline_meta,
            stage="build_source_anchor_pack_done",
            percent=22,
            label="核心素材筛选完成",
            detail="已生成 source anchor pack，准备生成初稿。",
        )

        print("\nSource anchor summary:")
        print(json.dumps(anchor_summary, ensure_ascii=False, indent=2))

        assert_anchor_pack_has_anchors(
            anchor_summary,
            allow_empty_anchors=args.allow_empty_anchors,
        )

        # -------------------------------------------------------------
        # Stage 3: Generate dialogue
        # -------------------------------------------------------------

        update_progress(
            pipeline_meta_path,
            pipeline_meta,
            stage="generate_dialogue",
            percent=28,
            label="正在生成初稿",
            detail="本阶段会调用本地大模型，可能需要较长时间。",
        )

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
        update_progress(
            pipeline_meta_path,
            pipeline_meta,
            stage="generate_dialogue_done",
            percent=40,
            label="初稿生成完成",
            detail="准备检查初稿格式和内容密度。",
        )

        if args.dry_run_prompt:
            pipeline_meta["status"] = "dry_run_prompt"
            pipeline_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
            pipeline_meta["final"] = {
                "stage": "generate_dialogue_prompt",
                "prompt_path": str(dialogue_path.with_suffix(".prompt.json")),
                "pipeline_meta": str(pipeline_meta_path),
            }
            write_pipeline_meta(pipeline_meta_path, pipeline_meta)
            print("\nDry-run prompt completed.")
            print(f"Prompt:        {dialogue_path.with_suffix('.prompt.json')}")
            print_pipeline_footer(status="dry_run_prompt", pipeline_meta_path=pipeline_meta_path, pipeline_meta=pipeline_meta)
            return

        # -------------------------------------------------------------
        # Stage 4: Validate generated dialogue
        # -------------------------------------------------------------

        update_progress(
            pipeline_meta_path,
            pipeline_meta,
            stage="validate_generated",
            percent=43,
            label="正在校验初稿",
            detail="检查轮数、A/B 格式、Source Appendix 和基础质量指标。",
        )

        generated_validation_path = run_validate_dialogue(
            dialogue_path=dialogue_path,
            meta_path=dialogue_meta_path,
            require_source_appendix=True,
        )

        generated_validation_summary = summarize_validation(generated_validation_path)
        pipeline_meta["stages"]["validate_generated"] = generated_validation_summary
        update_progress(
            pipeline_meta_path,
            pipeline_meta,
            stage="validate_generated_done",
            percent=46,
            label="初稿校验完成",
            detail="准备进入 critique / expand 阶段。",
        )

        print("\nGenerated validation summary:")
        print(json.dumps(generated_validation_summary, ensure_ascii=False, indent=2))

        assert_validation_mechanical_passed(
            generated_validation_summary,
            stage_name="Generated dialogue",
        )

        if args.mode == "draft":
            pipeline_meta["status"] = "success"
            pipeline_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
            pipeline_meta["final"] = {
                "stage": "generated_dialogue",
                "dialogue": str(dialogue_path),
                "validation": str(generated_validation_path),
                "quality_passed": generated_validation_summary.get("quality_passed"),
                "pipeline_meta": str(pipeline_meta_path),
            }
            update_progress(
                pipeline_meta_path,
                pipeline_meta,
                stage="completed",
                percent=100,
                label="快速草稿生成完成",
                detail="Draft pipeline 已完成。",
            )

            print("\nDraft pipeline completed successfully.")
            print(f"Source pack:        {source_pack_path}")
            print(f"Source anchor pack: {source_anchor_pack_path}")
            print(f"Dialogue:           {dialogue_path}")
            print(f"Validation:         {generated_validation_path}")
            print_pipeline_footer(status="success", pipeline_meta_path=pipeline_meta_path, pipeline_meta=pipeline_meta)
            return

        # -------------------------------------------------------------
        # Stage 5: Critique generated dialogue
        # -------------------------------------------------------------

        current_dialogue_path = dialogue_path
        current_meta_path = dialogue_meta_path
        current_validation_path = generated_validation_path
        current_stage = "generated_dialogue"

        if not args.skip_critique and not args.skip_expand:
            update_progress(
                pipeline_meta_path,
                pipeline_meta,
                stage="critique_generated",
                percent=52,
                label="正在分析初稿问题",
                detail="Critique agent 正在检查内容密度、自然度和素材使用。",
            )

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
            update_progress(
                pipeline_meta_path,
                pipeline_meta,
                stage="critique_generated_done",
                percent=58,
                label="初稿分析完成",
                detail="准备根据 critique 进行扩写。",
            )

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
            update_progress(
                pipeline_meta_path,
                pipeline_meta,
                stage="expand_dialogue",
                percent=62,
                label="正在扩写对话",
                detail="Expander 会根据校验报告和 critique 增加内容密度。",
            )

            expanded_dialogue_path, expanded_meta_path, expanded_validation_path, expanded_validation_summary = (
                run_expand_and_validate(
                    expander_config=args.expander_config,
                    dialogue_path=dialogue_path,
                    source_anchor_pack_path=source_anchor_pack_path,
                    validation_report_path=generated_validation_path,
                    critique_report_path=critique_report_path,
                    output_path=expanded_dialogue_path,
                    save_prompt=args.save_prompt,
                )
            )

            pipeline_meta["stages"]["expand_dialogue"] = {
                "output_path": str(expanded_dialogue_path),
                "prompt_path": str(expanded_dialogue_path.with_suffix(".prompt.json")),
                "meta_path": str(expanded_meta_path),
                "meta_summary": summarize_dialogue_meta(expanded_meta_path),
            }
            pipeline_meta["stages"]["validate_expanded"] = expanded_validation_summary
            update_progress(
                pipeline_meta_path,
                pipeline_meta,
                stage="expand_dialogue_done",
                percent=72,
                label="扩写与校验完成",
                detail="正在判断是否需要扩写重试。",
            )

            print("\nExpanded validation summary:")
            print(json.dumps(expanded_validation_summary, ensure_ascii=False, indent=2))

            assert_validation_mechanical_passed(
                expanded_validation_summary,
                stage_name="Expanded dialogue",
            )

            expand_retry_count = 0

            while (
                expanded_validation_summary.get("quality_passed") is not True
                and expand_retry_count < args.max_expand_retries
            ):
                expand_retry_count += 1

                print(
                    f"\nExpanded dialogue did not pass quality checks. "
                    f"Retrying expansion {expand_retry_count}/{args.max_expand_retries}..."
                )
                update_progress(
                    pipeline_meta_path,
                    pipeline_meta,
                    stage=f"expand_dialogue_retry_{expand_retry_count}",
                    percent=min(82, 72 + expand_retry_count * 4),
                    label=f"正在进行扩写重试 {expand_retry_count}/{args.max_expand_retries}",
                    detail="上一版扩写未完全通过质量检查，正在尝试补足内容密度。",
                )

                retry_dialogue_path = Path(
                    f"outputs/expansions/expanded_dialogue_pipeline_{timestamp}_retry{expand_retry_count}.md"
                )

                retry_dialogue_path, retry_meta_path, retry_validation_path, retry_validation_summary = (
                    run_expand_and_validate(
                        expander_config=args.expander_config,
                        dialogue_path=expanded_dialogue_path,
                        source_anchor_pack_path=source_anchor_pack_path,
                        validation_report_path=expanded_validation_path,
                        critique_report_path=critique_report_path,
                        output_path=retry_dialogue_path,
                        save_prompt=args.save_prompt,
                    )
                )

                pipeline_meta["stages"][f"expand_dialogue_retry_{expand_retry_count}"] = {
                    "input_dialogue": str(expanded_dialogue_path),
                    "input_validation": str(expanded_validation_path),
                    "output_path": str(retry_dialogue_path),
                    "prompt_path": str(retry_dialogue_path.with_suffix(".prompt.json")),
                    "meta_path": str(retry_meta_path),
                    "meta_summary": summarize_dialogue_meta(retry_meta_path),
                }
                pipeline_meta["stages"][f"validate_expanded_retry_{expand_retry_count}"] = retry_validation_summary
                update_progress(
                    pipeline_meta_path,
                    pipeline_meta,
                    stage=f"expand_dialogue_retry_{expand_retry_count}_done",
                    percent=min(84, 76 + expand_retry_count * 4),
                    label=f"扩写重试 {expand_retry_count} 完成",
                    detail="准备继续判断是否进入润色阶段。",
                )

                print(f"\nExpanded retry {expand_retry_count} validation summary:")
                print(json.dumps(retry_validation_summary, ensure_ascii=False, indent=2))

                assert_validation_mechanical_passed(
                    retry_validation_summary,
                    stage_name=f"Expanded dialogue retry {expand_retry_count}",
                )

                expanded_dialogue_path = retry_dialogue_path
                expanded_meta_path = retry_meta_path
                expanded_validation_path = retry_validation_path
                expanded_validation_summary = retry_validation_summary

            if expanded_validation_summary.get("quality_passed") is not True:
                if args.allow_expand_quality_fail:
                    print(
                        "\nWarning: Expanded dialogue still did not pass quality checks after retry, "
                        "but --allow_expand_quality_fail is set. Continuing to polish."
                    )
                else:
                    print(
                        "\nWarning: Expanded dialogue still did not pass quality checks after retry, "
                        "but mechanical validation passed. Continuing to polish and letting final "
                        "polished validation decide the full pipeline result."
                    )

                pipeline_meta["stages"]["validate_expanded_final"] = {
                    **expanded_validation_summary,
                    "expand_retry_count": expand_retry_count,
                    "continued_to_polish_after_expand_quality_fail": True,
                }
                update_progress(
                    pipeline_meta_path,
                    pipeline_meta,
                    stage="validate_expanded_final",
                    percent=84,
                    label="扩写阶段结束",
                    detail="扩写仍有质量提醒，但格式通过，继续交给润色和最终校验。",
                )

            current_dialogue_path = expanded_dialogue_path
            current_meta_path = expanded_meta_path
            current_validation_path = expanded_validation_path
            current_stage = "expanded_dialogue"

        # -------------------------------------------------------------
        # Stage 8: Polish dialogue
        # -------------------------------------------------------------

        if not args.skip_polish:
            update_progress(
                pipeline_meta_path,
                pipeline_meta,
                stage="polish_dialogue",
                percent=88,
                label="正在润色对话",
                detail="Polisher 正在优化语言自然度和重复表达。",
            )

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
            update_progress(
                pipeline_meta_path,
                pipeline_meta,
                stage="polish_dialogue_done",
                percent=94,
                label="润色完成",
                detail="准备进行最终质量校验。",
            )

            # ---------------------------------------------------------
            # Stage 9: Validate polished dialogue
            # ---------------------------------------------------------

            update_progress(
                pipeline_meta_path,
                pipeline_meta,
                stage="validate_polished",
                percent=96,
                label="正在最终校验",
                detail="检查 polished dialogue 是否满足最终质量门槛。",
            )

            polished_validation_path = run_validate_dialogue(
                dialogue_path=polished_dialogue_path,
                meta_path=polished_meta_path,
                require_source_appendix=True,
            )

            polished_validation_summary = summarize_validation(polished_validation_path)
            pipeline_meta["stages"]["validate_polished"] = polished_validation_summary
            update_progress(
                pipeline_meta_path,
                pipeline_meta,
                stage="validate_polished_done",
                percent=98,
                label="最终校验完成",
                detail="正在写入最终结果。",
            )

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
        pipeline_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        pipeline_meta["final"] = {
            "stage": current_stage,
            "dialogue": str(current_dialogue_path),
            "meta": str(current_meta_path),
            "validation": str(current_validation_path),
            "pipeline_meta": str(pipeline_meta_path),
        }
        update_progress(
            pipeline_meta_path,
            pipeline_meta,
            stage="completed",
            percent=100,
            label="Pipeline 已完成",
            detail="最终结果已写入。",
        )

        print("\nPipeline completed successfully.")
        print(f"Final stage:        {current_stage}")
        print(f"Final dialogue:     {current_dialogue_path}")
        print(f"Final validation:   {current_validation_path}")
        print(f"Source pack:        {source_pack_path}")
        print(f"Source anchor pack: {source_anchor_pack_path}")
        print_pipeline_footer(status="success", pipeline_meta_path=pipeline_meta_path, pipeline_meta=pipeline_meta)

    except Exception as exc:
        current_progress = pipeline_meta.get("progress", {}) or {}
        current_percent = int(current_progress.get("percent") or 0)
        set_progress(
            pipeline_meta,
            stage="failed",
            percent=current_percent,
            label="Pipeline 最终检查未通过或运行失败",
            detail=str(exc),
        )
        pipeline_meta["status"] = "failed"
        pipeline_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
        pipeline_meta["error"] = str(exc)
        write_pipeline_meta(pipeline_meta_path, pipeline_meta)
        print_pipeline_footer(status="failed", pipeline_meta_path=pipeline_meta_path, pipeline_meta=pipeline_meta)
        print(f"Error:              {exc}")
        raise


if __name__ == "__main__":
    main()