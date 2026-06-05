from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

APP_PATH = Path(__file__).resolve()
PROJECT_ROOT = APP_PATH.parents[1]

APP_NAME = "TTSDataGen V0.2"
REPO_URL = "https://github.com/Micheliliuv87/TTSDataGen"
CONTACT_URL = "https://micheliliuv87.github.io/"
LICENSE_NAME = "TTSDataGen License - Non-Commercial, All Rights Reserved"
COPYRIGHT_TEXT = "© 2026 Micheli V. All rights reserved."

# ---------------------------------------------------------------------
# Helpers
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
    st.caption(
        "Third-party models and components are governed by their own licenses."
    )
    st.caption(
        "Generated outputs depend on local models, retrieved sources, and user prompts. "
        "Please review outputs before use."
    )
    st.markdown(f"[Commercial licensing / contact]({CONTACT_URL})")
    

def inject_fixed_footer_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-bottom: 7rem;
        }

        .tts-fixed-footer {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 999999;
            padding: 0.55rem 1rem;
            border-top: 1px solid rgba(128, 128, 128, 0.25);
            background: var(--background-color);
            color: var(--text-color);
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.12);
            font-size: 0.78rem;
            line-height: 1.35;
        }

        .tts-fixed-footer-inner {
            max-width: 980px;
            margin: 0 auto;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: center;
            gap: 0.45rem;
            opacity: 0.86;
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

        @media (max-width: 700px) {
            .tts-fixed-footer {
                font-size: 0.70rem;
                padding: 0.45rem 0.75rem;
            }

            .tts-fixed-footer-inner {
                justify-content: flex-start;
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
    
    
def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return data if isinstance(data, dict) else {}


def read_text(path: Optional[Path]) -> str:
    if path is None or not path.exists():
        return ""

    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"Failed to read file: {exc}"

def save_streamlit_log(log_text: str, pipeline_meta_path: Optional[Path] = None) -> Path:
    logs_dir = PROJECT_ROOT / "logs" / "pipeline"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if pipeline_meta_path is not None:
        match = re.search(r"pipeline_(\d{8}_\d{6})\.json", pipeline_meta_path.name)
        if match:
            timestamp = match.group(1)

    log_path = logs_dir / f"streamlit_pipeline_{timestamp}.log"
    log_path.write_text(log_text, encoding="utf-8")
    return log_path


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

    # 有些 markdown renderer 会把长 numbered-list 行显示得很挤。
    body = re.sub(r"(?<!\n)\s+(\d+\.\s*[AB]\s*:)", r"\n\n\1", body)

    # Round heading 改成更清楚的网页小标题。
    body = re.sub(r"(?m)^\s*Dialogue\s*$", r"# Dialogue", body)
    body = re.sub(r"(?m)^\s*(Round\s+\d+)\s*$", r"### \1", body)

    # 每条 A/B speaker line 前补一个空行，避免 markdown 有序列表粘连。
    body = re.sub(r"(?m)^(\d+\.\s*[AB]\s*:)", r"\n\1", body)

    # 清理过多空行。
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    return body

def path_from_meta(meta: Dict[str, Any], *keys: str) -> Optional[Path]:
    current: Any = meta
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    if not current:
        return None

    path = Path(str(current))
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def resolve_display_artifact(pipeline_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pick the best available artifact to display in the UI.

    Important:
    - pipeline_meta["final"] is only populated when the full pipeline succeeds.
    - If final validation fails, polished/expanded/generated files may still exist.
    - The UI should show the best available artifact with a clear warning.
    """
    final = pipeline_meta.get("final", {}) or {}
    stages = pipeline_meta.get("stages", {}) or {}
    paths = pipeline_meta.get("paths", {}) or {}

    # 1. Success path: official final artifact.
    final_dialogue = final.get("dialogue")
    if final_dialogue:
        return {
            "stage": final.get("stage", "final"),
            "dialogue_path": final_dialogue,
            "validation_path": final.get("validation", ""),
            "meta_path": final.get("meta", ""),
            "is_official_final": True,
            "status_label": "最终通过版本",
            "warning": "",
        }

    # 2. Failed path: polished output exists but final quality check failed.
    polish_stage = stages.get("polish_dialogue", {}) or {}
    polished_dialogue = (
        polish_stage.get("output_path")
        or paths.get("polished_dialogue")
    )
    polished_validation = (
        stages.get("validate_polished", {}) or {}
    ).get("path") or paths.get("polished_validation")

    if polished_dialogue:
        return {
            "stage": "polished_dialogue",
            "dialogue_path": polished_dialogue,
            "validation_path": polished_validation or "",
            "meta_path": polish_stage.get("meta_path", ""),
            "is_official_final": False,
            "status_label": "已生成但未通过最终质量检查",
            "warning": (
                "Pipeline 最终质量检查未通过，但 polished markdown 已经生成。"
                "下面展示的是可检查的候选输出，不是正式通过版本。"
            ),
        }

    # 3. Fallback: expanded output.
    expanded_stage = stages.get("expand_dialogue_retry_1") or stages.get("expand_dialogue") or {}
    expanded_dialogue = (
        expanded_stage.get("output_path")
        or paths.get("expanded_dialogue")
    )
    expanded_validation = (
        stages.get("validate_expanded_retry_1", {}) or stages.get("validate_expanded", {}) or {}
    ).get("path") or paths.get("expanded_validation")

    if expanded_dialogue:
        return {
            "stage": "expanded_dialogue",
            "dialogue_path": expanded_dialogue,
            "validation_path": expanded_validation or "",
            "meta_path": expanded_stage.get("meta_path", ""),
            "is_official_final": False,
            "status_label": "中间扩写版本",
            "warning": (
                "Pipeline 没有产出正式 final 版本。下面展示的是 expanded 中间版本，"
                "需要人工检查。"
            ),
        }

    # 4. Fallback: generated draft.
    generated_stage = stages.get("generate_dialogue", {}) or {}
    generated_dialogue = (
        generated_stage.get("output_path")
        or paths.get("generated_dialogue")
    )
    generated_validation = (
        stages.get("validate_generated", {}) or {}
    ).get("path") or paths.get("generated_validation")

    if generated_dialogue:
        return {
            "stage": "generated_dialogue",
            "dialogue_path": generated_dialogue,
            "validation_path": generated_validation or "",
            "meta_path": generated_stage.get("meta_path", ""),
            "is_official_final": False,
            "status_label": "初稿版本",
            "warning": (
                "Pipeline 没有产出正式 final 版本。下面展示的是 generated 初稿，"
                "通常还需要扩写或润色。"
            ),
        }

    return {
        "stage": "none",
        "dialogue_path": "",
        "validation_path": "",
        "meta_path": "",
        "is_official_final": False,
        "status_label": "没有可展示输出",
        "warning": "没有找到 generated / expanded / polished markdown 输出。",
    }


def resolve_project_path(path_value: Any) -> Optional[Path]:
    if not path_value:
        return None

    path = Path(str(path_value))
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path

def extract_pipeline_meta_path(log_text: str) -> Optional[Path]:
    patterns = [
        r"Pipeline meta:\s+(.+?\.json)",
        r'"pipeline_meta"\s*:\s*"([^"]+?\.json)"',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, log_text)
        if matches:
            path = Path(matches[-1].strip())
            if path.is_absolute():
                return path
            return PROJECT_ROOT / path

    return None


def summarize_source_quality(pipeline_meta: Dict[str, Any]) -> Dict[str, Any]:
    retrieve = pipeline_meta.get("stages", {}).get("retrieve", {}) or {}
    anchor = pipeline_meta.get("stages", {}).get("build_source_anchor_pack", {}) or {}

    coverage_status = str(retrieve.get("coverage_status") or "").lower()
    usable_source_count = int(retrieve.get("usable_source_count") or 0)
    strong_source_count = int(retrieve.get("strong_source_count") or 0)
    anchor_count = int(anchor.get("anchor_count") or 0)
    rejected_anchor_count = int(anchor.get("rejected_anchor_count") or 0)

    if coverage_status == "high" or strong_source_count > 0:
        level = "较高"
        message = "系统找到了较强相关素材，适合直接生成。"
    elif coverage_status == "medium" or usable_source_count >= 3 or anchor_count > 0:
        level = "中等"
        message = "结果可以生成，但素材可能不完全贴合主题，生成内容可能需要扩写或存在轻微噪声。"
    else:
        level = "较低"
        message = "当前数据库中相关素材不足。建议换一个更具体或更宽泛的主题，或补充更多 source 数据。"

    return {
        "level": level,
        "message": message,
        "coverage_status": coverage_status or "unknown",
        "usable_source_count": usable_source_count,
        "strong_source_count": strong_source_count,
        "anchor_count": anchor_count,
        "rejected_anchor_count": rejected_anchor_count,
    }


def validation_badge(validation: Dict[str, Any]) -> str:
    if not validation:
        return "未知"

    if validation.get("quality_passed") is True:
        return "通过"

    if validation.get("mechanical_passed") is True:
        return "格式通过，质量需检查"

    return "未通过"


def run_pipeline_command(
    *,
    query: str,
    mode: str,
    extra_instructions: str,
    rounds: Optional[int],
) -> tuple[int, str]:
    cmd = [
        sys.executable,
        "run_pipeline.py",
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

    log_lines: list[str] = []

    process = subprocess.Popen(
        cmd,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_placeholder = st.empty()

    assert process.stdout is not None
    for line in process.stdout:
        log_lines.append(line)
        tail = "".join(log_lines[-160:])
        log_placeholder.code(tail, language="text")

    return_code = process.wait()
    full_log = "".join(log_lines)
    return return_code, full_log


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------

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

with st.sidebar:
    st.header("生成模式")

    mode_label = st.radio(
        "选择模式",
        options=[
            "快速草稿",
            "高质量完整版",
        ],
        index=1,
        help="快速草稿只生成初稿；高质量完整版会自动检查、扩写和润色。",
    )

    mode = {
        "快速草稿": "draft",
        "高质量完整版": "full",
    }[mode_label]

    if mode == "full":
        st.info(
            "Full 模式会经过检索、生成、校验、扩写、重试和润色，"
            "会多次调用本地模型。30轮可能需要 20–30 分钟或更久；"
            "轮数越多越慢。V0.2 阶段建议先用 12–20 轮试跑，确认方向后再生成 30 轮。\n"
            "建议输入更具体的主题，输入内容🈷越具体，生成质量越好"
        )
    else:
        st.info(
            "快速草稿只生成初稿，速度较快，但内容密度和语言质量可能不如 Full 模式。"
        )

    st.divider()

    rounds_enabled = st.checkbox("手动指定轮数", value=False)
    rounds = None
    if rounds_enabled:
        rounds = st.number_input(
            "轮数",
            min_value=1,
            max_value=80,
            value=12,
            step=1,
        )

    st.divider()

    st.markdown("**运行前检查**")
    st.write("项目状态：本地运行中")
    st.write("请确认 LM Studio 已启动，并加载对应模型。")

    with st.expander("开发者信息"):
        render_developer_info()
        st.divider()
        st.caption("本地调试信息")
        st.write(f"项目根目录：`{PROJECT_ROOT}`")

st.info("你可以用中文或英文描述需求；当前 V0.2 默认生成中文 A/B 对话。")

with st.form("generation_form"):
    query = st.text_area(
        "你想生成什么对话？",
        value="生成12轮A与B对话，主题是早餐文化、家庭习惯和代际差异",
        height=120,
        placeholder="例如：生成30轮A与B对话，主题是厨房气味、童年记忆和家庭聚会",
    )

    with st.expander("本次补充要求（可选）"):
        st.caption("只影响本次初稿生成，不会修改系统 rules。适合填写语气、受众、风格等临时要求。")
        extra_instructions = st.text_area(
            "临时补充说明",
            value="",
            height=100,
            placeholder="例如：语气自然一点，适合口语训练；不要太学术；尽量多用生活化例子。",
        )

    submitted = st.form_submit_button(
        "开始生成",
        type="primary",
        use_container_width=True,
    )


if submitted:
    if not query.strip():
        st.error("请先输入生成需求。")
        st.stop()

    st.session_state.pop("last_pipeline_meta_path", None)
    st.session_state.pop("last_log", None)

    with st.status("正在运行 pipeline...", expanded=True) as status:
        return_code, log_text = run_pipeline_command(
            query=query.strip(),
            mode=mode,
            extra_instructions=extra_instructions,
            rounds=int(rounds) if rounds is not None else None,
        )

        st.session_state["last_log"] = log_text

        pipeline_meta_path = extract_pipeline_meta_path(log_text)
        if pipeline_meta_path is not None:
            st.session_state["last_pipeline_meta_path"] = str(pipeline_meta_path)

        log_path = save_streamlit_log(log_text, pipeline_meta_path)
        st.session_state["last_log_path"] = str(log_path)

        if return_code == 0:
            status.update(label="Pipeline 运行完成", state="complete", expanded=False)
        else:
            status.update(label="Pipeline 运行失败", state="error", expanded=True)
            st.error("生成失败。请查看上方日志。")

pipeline_meta_path_raw = st.session_state.get("last_pipeline_meta_path")
pipeline_meta_path = Path(pipeline_meta_path_raw) if pipeline_meta_path_raw else None

if pipeline_meta_path and pipeline_meta_path.exists():
    pipeline_meta = load_json(pipeline_meta_path)
    display_artifact = resolve_display_artifact(pipeline_meta)

    final_dialogue_path = resolve_project_path(display_artifact.get("dialogue_path"))
    final_validation_path = resolve_project_path(display_artifact.get("validation_path"))
    final_meta_path = resolve_project_path(display_artifact.get("meta_path"))

    final_dialogue = read_text(final_dialogue_path)
    final_validation = load_json(final_validation_path) if final_validation_path else {}
    final_meta = load_json(final_meta_path) if final_meta_path else {}

    st.divider()
    st.subheader("生成结果")

    if display_artifact.get("warning"):
        st.warning(display_artifact["warning"])

    pipeline_status = pipeline_meta.get("status", "unknown")
    if pipeline_status == "success":
        st.success("Pipeline 已成功完成，当前展示的是最终通过版本。")
    elif pipeline_status == "failed":
        st.error("Pipeline 最终状态为 failed。当前展示的是已生成的最佳候选输出。")
        st.info(
                "建议：如果候选输出已经可用，可以直接下载；如果希望通过质量门槛，"
                "可以减少轮数、换更具体的主题，或重新运行 Full 模式。"
            )
    else:
        st.info(f"Pipeline status: {pipeline_status}")
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("展示版本", display_artifact.get("status_label", "unknown"))

    with col2:
        st.metric("质量", validation_badge(final_validation))

    with col3:
        stats = final_validation.get("stats", {}) or {}
        st.metric("轮数", stats.get("inferred_rounds", final_meta.get("rounds", "-")))

    with col4:
        length = stats.get("length", {}) or {}
        st.metric("Developed ratio", length.get("developed_line_ratio", "-"))

    source_quality = summarize_source_quality(pipeline_meta)

    if source_quality["level"] == "较高":
        st.success(f"素材匹配度：{source_quality['level']}。{source_quality['message']}")
    elif source_quality["level"] == "中等":
        st.warning(f"素材匹配度：{source_quality['level']}。{source_quality['message']}")
    else:
        st.error(f"素材匹配度：{source_quality['level']}。{source_quality['message']}")

    with st.expander("素材与验证详情", expanded=False):
        st.json(
            {
                "source_quality": source_quality,
                "final_validation": {
                    "passed": final_validation.get("passed"),
                    "mechanical_passed": final_validation.get("mechanical_passed"),
                    "quality_passed": final_validation.get("quality_passed"),
                    "needs_rewrite": final_validation.get("needs_rewrite"),
                    "verdict": final_validation.get("verdict"),
                    "stats": final_validation.get("stats", {}),
                },
                "paths": {
                    "display_stage": display_artifact.get("stage", ""),
                    "display_status_label": display_artifact.get("status_label", ""),
                    "is_official_final": display_artifact.get("is_official_final", False),
                    "dialogue": str(final_dialogue_path) if final_dialogue_path else "",
                    "validation": str(final_validation_path) if final_validation_path else "",
                    "meta": str(final_meta_path) if final_meta_path else "",
                    "pipeline_meta": str(pipeline_meta_path),
                    "streamlit_log": st.session_state.get("last_log_path", ""),
                },
            }
        )

    st.markdown("### 最终 Dialogue")

    if final_dialogue:
        preview_tab, raw_tab, source_tab = st.tabs(
            ["排版预览", "原始 Markdown", "Source Appendix"]
        )

        body_text, appendix_text = split_source_appendix(final_dialogue)

        with preview_tab:
            st.markdown(prepare_dialogue_preview(final_dialogue))

        with raw_tab:
            st.code(final_dialogue, language="markdown")

        with source_tab:
            if appendix_text:
                st.markdown(appendix_text)
            else:
                st.info("这个文件没有 Source Appendix。")

        st.download_button(
            "下载 Markdown",
            data=final_dialogue,
            file_name=(
                final_dialogue_path.name
                if display_artifact.get("is_official_final")
                else f"candidate_{final_dialogue_path.name}"
            ) if final_dialogue_path else "dialogue.md",
            mime="text/markdown",
            use_container_width=True,
        )
    else:
        st.warning("没有读取到最终 markdown 文件。")

elif st.session_state.get("last_log"):
    st.divider()
    st.subheader("运行日志")
    st.code(st.session_state["last_log"], language="text")
