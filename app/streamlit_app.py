from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# ---------------------------------------------------------------------
# Paths / project metadata
# ---------------------------------------------------------------------

APP_PATH = Path(__file__).resolve()
PROJECT_ROOT = APP_PATH.parents[1]

APP_NAME = "TTSDataGen V0.2"
REPO_URL = "https://github.com/Micheliliuv87/TTSDataGen"
CONTACT_URL = "https://micheliliuv87.github.io/"
LICENSE_NAME = "TTSDataGen License - Non-Commercial, All Rights Reserved"
COPYRIGHT_TEXT = "© 2026 Micheli V. All rights reserved."

UI_JOBS_DIR = PROJECT_ROOT / "outputs" / "ui_jobs"
CURRENT_JOB_PATH = UI_JOBS_DIR / "current_job.json"
PIPELINE_RUNS_DIR = PROJECT_ROOT / "outputs" / "pipeline_runs"
PIPELINE_LOGS_DIR = PROJECT_ROOT / "logs" / "pipeline"

AUTO_REFRESH_SECONDS = 2.0
UNRECOVERABLE_GRACE_SECONDS = 90

TERMINAL_PIPELINE_STATUSES = {
    "success",
    "failed",
    "dry_run_prompt",
    "cancelled",
    "cancelled_unrecoverable",
    "finished_unknown",
}

# ---------------------------------------------------------------------
# UI-only helpers
# ---------------------------------------------------------------------


def render_developer_info() -> None:
    st.markdown(f"**{APP_NAME}**")
    st.markdown(f"[GitHub Repository]({REPO_URL})")
    st.markdown(f"**License:** {LICENSE_NAME}")
    st.markdown(f"**Copyright:** {COPYRIGHT_TEXT}")
    st.caption(
        "This project is licensed for non-commercial personal, academic, "
        "and research use. Commercial use requires prior written permission."
    )
    st.caption("Third-party models and components are governed by their own licenses.")
    st.caption(
        "Generated outputs depend on local models, retrieved sources, and user prompts. "
        "Please review outputs before use."
    )
    st.markdown(f"[Commercial licensing / contact]({CONTACT_URL})")


def inject_fixed_footer_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --tts-footer-height: 3.6rem;
            --tts-footer-safe-space: 10rem;
        }

        .block-container,
        [data-testid="stMainBlockContainer"],
        [data-testid="stAppViewContainer"] .main .block-container {
            padding-bottom: var(--tts-footer-safe-space) !important;
        }

        .tts-fixed-footer {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 999999;
            min-height: var(--tts-footer-height);
            padding: 0.42rem 1rem;
            border-top: 1px solid rgba(128, 128, 128, 0.25);
            background: var(--background-color);
            color: var(--text-color);
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.14);
            font-size: 0.74rem;
            line-height: 1.35;
        }

        .tts-fixed-footer-inner {
            max-width: 1100px;
            margin: 0 auto;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            gap: 0.42rem;
            opacity: 0.82;
        }

        .tts-fixed-footer a {
            color: #4da3ff;
            text-decoration: none;
        }

        .tts-fixed-footer a:hover {
            text-decoration: underline;
        }

        .tts-footer-separator {
            opacity: 0.55;
        }

        .tts-bottom-safe-area {
            height: var(--tts-footer-safe-space);
        }

        @media (max-width: 1200px) {
            :root {
                --tts-footer-height: 4.8rem;
                --tts-footer-safe-space: 14rem;
            }

            .tts-fixed-footer {
                font-size: 0.70rem;
                padding: 0.38rem 0.75rem;
            }

            .tts-fixed-footer-inner {
                max-width: 95vw;
                justify-content: center;
                gap: 0.35rem;
            }
        }

        @media (max-width: 700px) {
            :root {
                --tts-footer-height: 5.6rem;
                --tts-footer-safe-space: 16rem;
            }

            .tts-fixed-footer {
                font-size: 0.68rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_fixed_footer() -> None:
    inject_fixed_footer_css()
    st.markdown(
        f"""
        <div class="tts-fixed-footer">
            <div class="tts-fixed-footer-inner">
                <span>{APP_NAME}</span>
                <span class="tts-footer-separator">·</span>
                <span>Non-commercial use only</span>
                <span class="tts-footer-separator">·</span>
                <span>{COPYRIGHT_TEXT}</span>
                <span class="tts-footer-separator">·</span>
                <a href="{REPO_URL}" target="_blank">GitHub</a>
                <span class="tts-footer-separator">·</span>
                <a href="{CONTACT_URL}" target="_blank">Contact / Commercial Licensing</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bottom_safe_area() -> None:
    st.markdown('<div class="tts-bottom-safe-area"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Basic IO helpers
# ---------------------------------------------------------------------


def ensure_ui_dirs() -> None:
    UI_JOBS_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    PIPELINE_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_text(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return ""

    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Failed to read file: {exc}"


def read_tail(path: Optional[Path], *, line_count: int = 180) -> str:
    text = read_text(path)
    if not text:
        return ""

    lines = text.splitlines()
    return "\n".join(lines[-line_count:])


NOISY_LOG_PATTERNS = [
    r"you are sending unauthenticated",
    r"unauthenticated request",
    r"^warning:\s*you are sending",
    r"^warning:\s*.*token",
    r"futurewarning",
    r"userwarning",
    r"deprecationwarning",
    r"urllib3",
    r"huggingface_hub",
]


def is_noisy_log_line(line: str) -> bool:
    text = str(line or "").strip().lower()
    if not text:
        return False

    return any(re.search(pattern, text) for pattern in NOISY_LOG_PATTERNS)


def split_log_for_display(log_text: str) -> tuple[str, str]:
    lines = str(log_text or "").splitlines()
    clean_lines: list[str] = []
    noisy_lines: list[str] = []

    for line in lines:
        if is_noisy_log_line(line):
            noisy_lines.append(line)
        else:
            clean_lines.append(line)

    return "\n".join(clean_lines).strip(), "\n".join(noisy_lines).strip()


def render_task_logs(log_path: Optional[Path]) -> None:
    log_tail = read_tail(log_path, line_count=240)
    if not log_tail:
        st.caption("日志文件尚未写入。")
        return

    clean_log, noisy_log = split_log_for_display(log_tail)

    with st.expander("任务日志（关键输出）", expanded=False):
        if clean_log:
            st.code(clean_log, language="text")
        else:
            st.caption("暂无关键日志。")

    if noisy_log:
        noisy_count = len(noisy_log.splitlines())
        with st.expander(f"折叠的 warning / debug 日志（{noisy_count} 行）", expanded=False):
            st.code(noisy_log, language="text")

    with st.expander("完整原始日志", expanded=False):
        st.code(log_tail, language="text")


def resolve_project_path(path_value: Any) -> Optional[Path]:
    if not path_value:
        return None

    path = Path(str(path_value))
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def rel_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None

    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


# ---------------------------------------------------------------------
# Markdown preview helpers
# ---------------------------------------------------------------------


def split_source_appendix(markdown_text: str) -> tuple[str, str]:
    text = str(markdown_text or "").replace("\r\n", "\n")

    match = re.search(
        r"\n+##\s*(?:Source Appendix|Source Notes|Sources)\b",
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return text.strip(), ""

    body = text[: match.start()].strip()
    appendix = text[match.start() :].strip()
    return body, appendix


def prepare_dialogue_preview(markdown_text: str) -> str:
    body, _ = split_source_appendix(markdown_text)

    # Avoid Streamlit markdown merging numbered A/B lines after wrapping.
    body = re.sub(r"(?<!\n)\s+(\d+\.\s*[AB]\s*:)", r"\n\n\1", body)
    body = re.sub(r"(?m)^\s*Dialogue\s*$", r"# Dialogue", body)
    body = re.sub(r"(?m)^\s*(Round\s+\d+)\s*$", r"### \1", body)
    body = re.sub(r"(?m)^(\d+\.\s*[AB]\s*:)", r"\n\1", body)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    return body


# ---------------------------------------------------------------------
# Pipeline contract consumers
# ---------------------------------------------------------------------


def get_display_artifact(pipeline_meta: Dict[str, Any]) -> Dict[str, Any]:
    artifact = (
        pipeline_meta.get("display_artifact")
        or pipeline_meta.get("best_available_artifact")
        or {}
    )

    if isinstance(artifact, dict) and artifact.get("stage"):
        return artifact

    final = pipeline_meta.get("final", {}) or {}
    if final.get("dialogue"):
        return {
            "stage": final.get("stage", "final"),
            "dialogue": final.get("dialogue", ""),
            "meta": final.get("meta", ""),
            "validation": final.get("validation", ""),
            "is_official_final": True,
            "status_label": "最终通过版本",
            "warning": "",
        }

    return {
        "stage": "none",
        "dialogue": "",
        "meta": "",
        "validation": "",
        "is_official_final": False,
        "status_label": "没有可展示输出",
        "warning": "没有找到 generated / expanded / polished markdown 输出。",
    }


def get_source_quality(pipeline_meta: Dict[str, Any]) -> Dict[str, Any]:
    source_quality = pipeline_meta.get("source_quality")
    if isinstance(source_quality, dict) and source_quality:
        return source_quality

    stages = pipeline_meta.get("stages", {}) or {}
    retrieve = stages.get("retrieve", {}) or {}
    anchor = stages.get("build_source_anchor_pack", {}) or {}

    return {
        "level": "未知",
        "message": "当前 pipeline_meta 没有 source_quality 字段。",
        "coverage_status": retrieve.get("coverage_status", "unknown"),
        "usable_source_count": retrieve.get("usable_source_count", 0),
        "strong_source_count": retrieve.get("strong_source_count", 0),
        "anchor_count": anchor.get("anchor_count", 0),
        "rejected_anchor_count": anchor.get("rejected_anchor_count", 0),
    }


def validation_badge(validation: Dict[str, Any], artifact: Dict[str, Any]) -> str:
    quality_passed = validation.get("quality_passed", artifact.get("quality_passed"))
    mechanical_passed = validation.get("mechanical_passed", artifact.get("mechanical_passed"))

    if quality_passed is True:
        return "通过"
    if mechanical_passed is True:
        return "格式通过，质量需检查"
    if validation or artifact.get("validation"):
        return "未通过"
    return "未知"


def infer_progress_from_meta(pipeline_meta: Dict[str, Any], job: Dict[str, Any]) -> Dict[str, Any]:
    progress = pipeline_meta.get("progress")
    if isinstance(progress, dict) and progress.get("percent") is not None:
        try:
            percent = int(progress.get("percent") or 0)
        except (TypeError, ValueError):
            percent = 0
        return {
            "percent": max(0, min(100, percent)),
            "label": str(progress.get("label") or "Pipeline 正在运行"),
            "detail": str(progress.get("detail") or ""),
            "stage": str(progress.get("stage") or ""),
            "updated_at": str(progress.get("updated_at") or ""),
            "is_estimated": bool(progress.get("is_estimated", True)),
        }

    status = str(pipeline_meta.get("status") or job.get("status") or "running")
    stages = pipeline_meta.get("stages", {}) or {}

    if status == "success":
        return {"percent": 100, "label": "Pipeline 已完成", "detail": "", "stage": "completed", "updated_at": "", "is_estimated": True}
    if status in {"failed", "cancelled"}:
        return {"percent": 100, "label": "Pipeline 已结束", "detail": "请查看结果或错误信息。", "stage": status, "updated_at": "", "is_estimated": True}

    checkpoints = [
        ("validate_polished", 96, "正在最终校验"),
        ("polish_dialogue", 90, "正在润色对话"),
        ("validate_expanded_retry_1", 80, "正在处理扩写重试结果"),
        ("expand_dialogue_retry_1", 76, "正在扩写重试"),
        ("validate_expanded", 70, "扩写已完成，正在校验"),
        ("expand_dialogue", 62, "正在扩写对话"),
        ("critique_generated", 55, "正在分析初稿问题"),
        ("validate_generated", 45, "正在校验初稿"),
        ("generate_dialogue", 32, "正在生成初稿"),
        ("build_source_anchor_pack", 20, "正在筛选核心素材"),
        ("retrieve", 10, "正在检索相关素材"),
    ]

    for key, percent, label in checkpoints:
        if key in stages:
            return {
                "percent": percent,
                "label": label,
                "detail": "当前进度为阶段估算值。",
                "stage": key,
                "updated_at": "",
                "is_estimated": True,
            }

    return {
        "percent": 3,
        "label": "任务已启动",
        "detail": "正在等待 pipeline 写入进度。",
        "stage": "starting",
        "updated_at": "",
        "is_estimated": True,
    }


def render_task_progress(job: Dict[str, Any], pipeline_meta: Dict[str, Any]) -> None:
    progress = infer_progress_from_meta(pipeline_meta, job)
    percent = int(progress.get("percent") or 0)
    label = str(progress.get("label") or "Pipeline 正在运行")
    detail = str(progress.get("detail") or "")
    updated_at = str(progress.get("updated_at") or "")

    left, right = st.columns([5, 1])
    with left:
        st.progress(percent / 100, text=label)
    with right:
        st.metric("进度", f"{percent}%")

    caption_parts = []
    if detail:
        caption_parts.append(detail)
    if updated_at:
        caption_parts.append(f"更新时间：{updated_at}")

    age = job_age_seconds(job) if job else 0
    if age > 0:
        minutes = int(age // 60)
        seconds = int(age % 60)
        caption_parts.append(f"已运行约 {minutes}分{seconds:02d}秒")

    if progress.get("is_estimated"):
        caption_parts.append("进度为阶段估算值；模型生成阶段可能停留较久。")

    if caption_parts:
        st.caption(" · ".join(caption_parts))


# ---------------------------------------------------------------------
# Persistent UI job manager
# ---------------------------------------------------------------------


def pid_is_alive(pid_value: Any) -> bool:
    try:
        pid = int(pid_value)
    except (TypeError, ValueError):
        return False

    if pid <= 0:
        return False

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False

    # macOS / Linux: zombie process still exists in process table,
    # but it should not be treated as a running pipeline.
    if sys.platform != "win32":
        try:
            result = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)],
                capture_output=True,
                text=True,
                timeout=1,
            )
            proc_stat = result.stdout.strip().upper()
            if "Z" in proc_stat:
                return False
        except Exception:
            pass

    return True


def load_current_job() -> Dict[str, Any]:
    return load_json(CURRENT_JOB_PATH)


def save_current_job(job: Dict[str, Any]) -> None:
    write_json(CURRENT_JOB_PATH, job)


def job_pipeline_meta_path(job: Dict[str, Any]) -> Optional[Path]:
    return resolve_project_path(job.get("pipeline_meta_path"))


def job_log_path(job: Dict[str, Any]) -> Optional[Path]:
    return resolve_project_path(job.get("log_path"))


def cancel_process_group(pid_value: Any, *, reason: str, timeout_seconds: float = 3.0) -> bool:
    try:
        pid = int(pid_value)
    except (TypeError, ValueError):
        return False

    if pid <= 0:
        return False

    if not pid_is_alive(pid):
        return True

    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            return False

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if not pid_is_alive(pid):
            return True
        time.sleep(0.2)

    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except Exception:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            return False

    return True


def mark_pipeline_meta_cancelled(job: Dict[str, Any], *, status: str, message: str) -> None:
    meta_path = job_pipeline_meta_path(job)
    if meta_path is None:
        return

    meta = load_json(meta_path)
    if not meta:
        meta = {
            "timestamp": job.get("run_id", ""),
            "run_id": job.get("run_id", ""),
            "query": job.get("query", ""),
            "mode": job.get("mode", ""),
            "paths": {
                "pipeline_meta": rel_path(meta_path),
            },
        }

    meta["status"] = "cancelled"
    meta["error"] = message
    meta["ui_cancel_status"] = status
    meta["ui_cancelled_at"] = datetime.now().isoformat(timespec="seconds")

    write_json(meta_path, meta)


def cancel_job(job: Dict[str, Any], *, status: str = "cancelled", reason: str = "用户取消任务。") -> Dict[str, Any]:
    killed = cancel_process_group(job.get("pid"), reason=reason)
    now = datetime.now().isoformat(timespec="seconds")

    job["status"] = status
    job["cancel_reason"] = reason
    job["cancelled_at"] = now
    job["process_killed"] = killed

    mark_pipeline_meta_cancelled(job, status=status, message=reason)
    save_current_job(job)
    return job


def job_age_seconds(job: Dict[str, Any]) -> float:
    started_at = parse_datetime(job.get("started_at"))
    if started_at is None:
        return 0.0

    return max(0.0, (datetime.now() - started_at).total_seconds())


def is_unrecoverable_running_job(job: Dict[str, Any]) -> bool:
    if not pid_is_alive(job.get("pid")):
        return False

    if job_age_seconds(job) < UNRECOVERABLE_GRACE_SECONDS:
        return False

    log_path = job_log_path(job)
    meta_path = job_pipeline_meta_path(job)

    has_log = bool(log_path and log_path.exists())
    has_meta = bool(meta_path and meta_path.exists())

    # If either log or meta exists, the page can still recover useful progress.
    return not has_log and not has_meta


def refresh_job(job: Dict[str, Any]) -> Dict[str, Any]:
    if not job:
        return {}

    status = str(job.get("status") or "")

    if status in {"cancelled", "cancelled_unrecoverable"}:
        return job

    meta = load_json(job_pipeline_meta_path(job))
    meta_status = str(meta.get("status") or "")
    now = datetime.now().isoformat(timespec="seconds")

    # Most important rule:
    # If pipeline_meta says the pipeline is done, trust it first.
    # Do not keep showing "running" just because pid_is_alive() still returns true.
    if meta_status in TERMINAL_PIPELINE_STATUSES:
        job["status"] = meta_status
        job["pipeline_status"] = meta_status
        job["finished_at"] = job.get("finished_at") or now
        job["last_seen_at"] = now
        save_current_job(job)
        return job

    if is_unrecoverable_running_job(job):
        return cancel_job(
            job,
            status="cancelled_unrecoverable",
            reason=(
                "UI 无法恢复该任务：运行超过恢复宽限时间后仍找不到日志文件或 pipeline_meta。"
                "已自动停止后台 pipeline，避免本地模型继续无提示运行。"
            ),
        )

    alive = pid_is_alive(job.get("pid"))

    if alive:
        job["status"] = "running"
        job["last_seen_at"] = now
        if meta_status:
            job["pipeline_status"] = meta_status
        save_current_job(job)
        return job

    if meta_status in TERMINAL_PIPELINE_STATUSES:
        job["status"] = meta_status
    elif status:
        job["status"] = status if status != "running" else "finished_unknown"
    else:
        job["status"] = "finished_unknown"

    job["finished_at"] = job.get("finished_at") or now
    save_current_job(job)
    return job


def start_pipeline_job(
    *,
    query: str,
    mode: str,
    extra_instructions: str,
    rounds: Optional[int],
) -> Dict[str, Any]:
    ensure_ui_dirs()

    run_id = make_run_id()
    log_path = PIPELINE_LOGS_DIR / f"streamlit_pipeline_{run_id}.log"
    pipeline_meta_path = PIPELINE_RUNS_DIR / f"pipeline_{run_id}.json"

    cmd = [
        sys.executable,
        "run_pipeline.py",
        "--run_id",
        run_id,
        "--query",
        query,
        "--mode",
        mode,
        "--save_prompt",
    ]

    if extra_instructions.strip():
        cmd.extend(["--extra_instructions", extra_instructions.strip()])

    if rounds is not None:
        cmd.extend(["--rounds", str(rounds)])

    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    log_file.close()

    job = {
        "job_id": run_id,
        "run_id": run_id,
        "status": "running",
        "pid": process.pid,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "mode": mode,
        "rounds": rounds,
        "extra_instructions": extra_instructions,
        "cmd": cmd,
        "log_path": rel_path(log_path),
        "pipeline_meta_path": rel_path(pipeline_meta_path),
    }

    save_current_job(job)
    return job


def clear_current_job() -> None:
    if CURRENT_JOB_PATH.exists():
        CURRENT_JOB_PATH.unlink()


# ---------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------


def render_job_status(job: Dict[str, Any]) -> None:
    if not job:
        return

    status = job.get("status", "unknown")
    log_path = job_log_path(job)
    meta_path = job_pipeline_meta_path(job)
    pipeline_meta = load_json(meta_path)

    st.divider()
    st.subheader("当前任务")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Job ID", job.get("job_id", "-"))
    with col2:
        st.metric("状态", status)
    with col3:
        st.metric("模式", job.get("mode", "-"))

    with st.expander("本次请求", expanded=False):
        st.write(job.get("query", ""))
        if job.get("extra_instructions"):
            st.caption("补充要求")
            st.write(job.get("extra_instructions"))
        st.json(
            {
                "pid": job.get("pid"),
                "started_at": job.get("started_at"),
                "rounds": job.get("rounds"),
                "log_path": str(log_path) if log_path else "",
                "pipeline_meta_path": str(meta_path) if meta_path else "",
            }
        )

    if status == "running":
        st.info("任务正在后台运行。刷新页面、切换标签页或手机临时退出后，页面会通过 current_job.json 恢复当前任务。")

        render_task_progress(job, pipeline_meta)

        if st.button("取消当前任务", type="secondary", use_container_width=True):
            cancel_job(job, reason="用户在 Streamlit UI 中取消当前任务。")
            st.rerun()

        render_task_logs(log_path)

    elif status == "cancelled":
        st.warning(job.get("cancel_reason") or "任务已取消。")
        if st.button("清除当前任务记录", use_container_width=True):
            clear_current_job()
            st.rerun()

    elif status == "cancelled_unrecoverable":
        st.error(job.get("cancel_reason") or "任务因无法恢复页面状态而被自动停止。")
        if st.button("清除当前任务记录", use_container_width=True):
            clear_current_job()
            st.rerun()

    elif status == "finished_unknown":
        st.warning("后台进程已经结束，但没有读取到明确的 pipeline 终态。请查看日志和 pipeline_meta。")
        render_task_logs(log_path)
        if st.button("清除当前任务记录", use_container_width=True):
            clear_current_job()
            st.rerun()


def render_pipeline_result(pipeline_meta_path: Optional[Path]) -> None:
    pipeline_meta = load_json(pipeline_meta_path)
    if not pipeline_meta:
        return

    display_artifact = get_display_artifact(pipeline_meta)
    source_quality = get_source_quality(pipeline_meta)
    ui_summary = pipeline_meta.get("ui_summary", {}) or {}

    dialogue_path = resolve_project_path(display_artifact.get("dialogue"))
    validation_path = resolve_project_path(display_artifact.get("validation"))
    meta_path = resolve_project_path(display_artifact.get("meta"))

    dialogue_text = read_text(dialogue_path)
    validation = load_json(validation_path)
    artifact_meta = load_json(meta_path)

    st.divider()
    st.subheader("生成结果")

    severity = ui_summary.get("severity")
    message = ui_summary.get("message")
    if severity == "success":
        st.success(message or "Pipeline 已成功完成，当前展示的是最终通过版本。")
    elif severity == "warning":
        st.warning(message or "Pipeline 最终检查未通过，但已生成候选输出。")
    elif severity == "error":
        st.error(message or "Pipeline 运行失败。")
    elif severity == "info":
        st.info(message or "Pipeline 正在运行。")
    else:
        status_text = pipeline_meta.get("status", "unknown")
        st.info(f"Pipeline status: {status_text}")

    if display_artifact.get("warning"):
        st.warning(display_artifact["warning"])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("展示版本", display_artifact.get("status_label", "unknown"))

    with col2:
        st.metric("质量", validation_badge(validation, display_artifact))

    with col3:
        stats = validation.get("stats", {}) or {}
        st.metric("轮数", stats.get("inferred_rounds", artifact_meta.get("rounds", "-")))

    with col4:
        length = stats.get("length", {}) or {}
        st.metric("Developed ratio", length.get("developed_line_ratio", "-"))

    if source_quality.get("level") == "较高":
        st.success(f"素材匹配度：{source_quality['level']}。{source_quality.get('message', '')}")
    elif source_quality.get("level") == "中等":
        st.warning(f"素材匹配度：{source_quality['level']}。{source_quality.get('message', '')}")
    elif source_quality.get("level") == "较低":
        st.error(f"素材匹配度：{source_quality['level']}。{source_quality.get('message', '')}")
    else:
        st.info(f"素材匹配度：{source_quality.get('level', '未知')}。{source_quality.get('message', '')}")

    with st.expander("素材与验证详情", expanded=False):
        st.json(
            {
                "ui_summary": ui_summary,
                "source_quality": source_quality,
                "display_artifact": display_artifact,
                "validation": {
                    "passed": validation.get("passed"),
                    "mechanical_passed": validation.get("mechanical_passed"),
                    "quality_passed": validation.get("quality_passed"),
                    "needs_rewrite": validation.get("needs_rewrite"),
                    "verdict": validation.get("verdict"),
                    "stats": validation.get("stats", {}),
                },
                "paths": {
                    "dialogue": str(dialogue_path) if dialogue_path else "",
                    "validation": str(validation_path) if validation_path else "",
                    "meta": str(meta_path) if meta_path else "",
                    "pipeline_meta": str(pipeline_meta_path) if pipeline_meta_path else "",
                    "job_file": str(CURRENT_JOB_PATH),
                },
            }
        )

    st.markdown("### Dialogue 输出")

    if dialogue_text:
        preview_tab, raw_tab, source_tab = st.tabs(["排版预览", "原始 Markdown", "Source Appendix"])
        _, appendix_text = split_source_appendix(dialogue_text)

        with preview_tab:
            st.markdown(prepare_dialogue_preview(dialogue_text))

        with raw_tab:
            st.code(dialogue_text, language="markdown")

        with source_tab:
            if appendix_text:
                st.markdown(appendix_text)
            else:
                st.info("这个文件没有 Source Appendix。")

        st.download_button(
            "下载 Markdown",
            data=dialogue_text,
            file_name=(
                dialogue_path.name
                if display_artifact.get("is_official_final")
                else f"candidate_{dialogue_path.name}"
            ) if dialogue_path else "dialogue.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.warning("没有读取到可展示的 markdown 文件。")


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------


ensure_ui_dirs()

st.set_page_config(
    page_title="TTSDataGen V0.2",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ TTSDataGen V0.2")
st.caption(
    "输入中文或英文需求，生成中文 A/B 对话。"
    "Full 模式会进行多轮校验、扩写和润色。"
)

render_fixed_footer()

current_job = refresh_job(load_current_job())
job_running = bool(current_job and current_job.get("status") == "running" and pid_is_alive(current_job.get("pid")))

with st.sidebar:
    st.header("生成模式")

    mode_label = st.radio(
        "选择模式",
        options=["快速草稿", "高质量完整版"],
        index=1,
        help="快速草稿只生成初稿；高质量完整版会自动检查、扩写和润色。",
        disabled=job_running,
    )

    mode = {"快速草稿": "draft", "高质量完整版": "full"}[mode_label]

    if mode == "full":
        st.markdown(
            """
            <div style="
                padding: 0.9rem 1rem;
                border-radius: 0.6rem;
                background: rgba(49, 130, 206, 0.16);
                border: 1px solid rgba(49, 130, 206, 0.35);
                line-height: 1.65;
                font-size: 0.92rem;
            ">
            <strong>Full 模式说明</strong><br><br>
            会经过检索、生成、校验、扩写、重试和润色，并多次调用本地模型。<br><br>
            30轮可能需要 20–30 分钟或更久；轮数越多越慢。V0.2 阶段建议先用 12–20 轮试跑，确认方向后再生成 30 轮。<br><br>
            建议输入更具体的主题；输入内容越具体，生成质量通常越好。
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "**快速草稿说明**\n\n"
            "- 只生成初稿，速度较快。\n"
            "- 内容密度和语言质量可能不如 Full 模式。"
        )

    st.divider()

    rounds_enabled = st.checkbox("手动指定轮数", value=False, disabled=job_running)
    rounds = None
    if rounds_enabled:
        rounds = st.number_input(
            "轮数",
            min_value=1,
            max_value=80,
            value=12,
            step=1,
            disabled=job_running,
        )

    st.divider()

    st.markdown("**运行前检查**")
    st.write("项目状态：本地运行中")
    st.write("请确认 LM Studio 已启动，并加载对应模型。")

    if job_running:
        st.warning("已有任务正在运行。请等待完成，或在主页面取消当前任务。")

    with st.expander("开发者信息"):
        render_developer_info()
        st.divider()
        st.caption("本地调试信息")
        st.write(f"项目根目录：`{PROJECT_ROOT}`")
        st.write(f"Job 文件：`{CURRENT_JOB_PATH}`")

st.info("你可以用中文或英文描述需求；当前 V0.2 默认生成中文 A/B 对话。")

with st.form("generation_form"):
    query = st.text_area(
        "你想生成什么对话？",
        value=(
            "生成12轮A与B对话，主题是早餐文化、家庭习惯和代际差异。"
            "请围绕一个普通家庭的早晨展开，讨论早餐选择如何反映工作节奏、健康观念、"
            "代际差异和童年记忆。语气自然，适合中文口语训练；"
            "A和B都要有实质内容，可以互相提问、补充例子、提出不同看法；"
            "不要写成百科总结。"
        ),
        height=160,
        placeholder=(
            "例如：生成12轮A与B对话，主题是法国大革命前夕的王室形象、"
            "财政危机和民众不满。语气自然，有故事感，适合中文口语训练。"
        ),
        disabled=job_running,
    )

    with st.expander("本次补充要求（可选）"):
        st.caption("只影响本次初稿生成，不会修改系统 rules。适合填写语气、受众、风格等临时要求。")
        extra_instructions = st.text_area(
            "临时补充说明",
            value="",
            height=100,
            placeholder="例如：语气自然一点，适合口语训练；不要太学术；尽量多用生活化例子。",
            disabled=job_running,
        )

    submitted = st.form_submit_button(
        "开始生成",
        type="primary",
        use_container_width=True,
        disabled=job_running,
    )

if submitted:
    if not query.strip():
        st.error("请先输入生成需求。")
        st.stop()

    job = start_pipeline_job(
        query=query.strip(),
        mode=mode,
        extra_instructions=extra_instructions,
        rounds=int(rounds) if rounds is not None else None,
    )
    st.success(f"已启动后台任务：{job['job_id']}")
    st.rerun()

current_job = refresh_job(load_current_job())

if current_job:
    render_job_status(current_job)

    pipeline_meta_path = job_pipeline_meta_path(current_job)
    pipeline_meta = load_json(pipeline_meta_path)
    meta_status = str(pipeline_meta.get("status") or "")

    if pipeline_meta_path and pipeline_meta_path.exists():
        if current_job.get("status") != "running" or meta_status in TERMINAL_PIPELINE_STATUSES:
            render_pipeline_result(pipeline_meta_path)

    if current_job.get("status") == "running" and meta_status not in TERMINAL_PIPELINE_STATUSES:
        time.sleep(AUTO_REFRESH_SECONDS)
        st.rerun()

    if current_job.get("status") in {
        "success",
        "failed",
        "dry_run_prompt",
        "finished_unknown",
    }:
        if st.button("清除当前任务记录", use_container_width=True):
            clear_current_job()
            st.rerun()
