from __future__ import annotations

import asyncio
import hashlib
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import chainlit as cl
from chainlit.input_widget import Select

from state import AgentState, create_initial_state
from utils.experiment_logger import ExperimentLogger
from workflow import generate_app, plan_only_app

MODE_KEY = "\u8fd0\u884c\u6a21\u5f0f"  # 运行模式
PLANNING_MODE = "Planning \u6a21\u5f0f"  # Planning 模式
GENERATION_MODE = "Generation \u6a21\u5f0f"  # Generation 模式


def _normalize_storyboard_text(value: Any) -> str | None:
    """Normalize planner output to JSON text for session persistence."""

    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        return text or None

    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        return str(value)


def _merge_state(runtime_state: AgentState, patch: dict[str, Any]) -> None:
    """Safely merge partial state updates."""

    for key, value in patch.items():
        runtime_state[key] = value


def _safe_get_session_state() -> AgentState:
    """Get state from session or initialize it when absent/invalid."""

    try:
        stored = cl.user_session.get("agent_state")
        if isinstance(stored, dict):
            merged = create_initial_state(stored.get("task", ""))
            for key, value in stored.items():
                merged[key] = value
            merged["storyboard"] = _normalize_storyboard_text(merged.get("storyboard"))
            return merged
    except Exception:
        pass
    return create_initial_state("")


def _safe_set_session_state(state: AgentState) -> None:
    """Persist state into Chainlit session safely."""

    try:
        state["storyboard"] = _normalize_storyboard_text(state.get("storyboard"))
        cl.user_session.set("agent_state", state)
    except Exception:
        # Keep runtime robust even if session persistence fails.
        pass


def _get_chat_settings() -> dict[str, Any]:
    """Read chat settings from session with defaults."""

    try:
        settings = cl.user_session.get("chat_settings")
        if isinstance(settings, dict):
            return settings
    except Exception:
        pass
    return {MODE_KEY: GENERATION_MODE}


def _get_current_mode() -> str:
    """Resolve active mode from session settings."""

    settings = _get_chat_settings()
    mode = settings.get(MODE_KEY, GENERATION_MODE)
    if mode not in {PLANNING_MODE, GENERATION_MODE}:
        return GENERATION_MODE
    return mode


def _safe_stat_mtime(path: Path) -> float:
    """Get file mtime with safe fallback."""

    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _find_latest_video(since_ts: float | None = None) -> Path | None:
    """Find latest non-partial mp4 after a given timestamp."""

    roots: list[Path] = [Path("media/runs")]
    candidates: list[Path] = []
    try:
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*.mp4"):
                if path.is_file() and "partial_movie_files" not in path.parts:
                    candidates.append(path)
    except Exception:
        return None

    if since_ts is not None:
        candidates = [p for p in candidates if _safe_stat_mtime(p) >= since_ts]
    if not candidates:
        return None

    candidates.sort(key=_safe_stat_mtime, reverse=True)
    return candidates[0]


def _find_latest_image(since_ts: float | None = None) -> Path | None:
    """Find latest png after a given timestamp."""

    roots: list[Path] = [Path("media/runs")]
    candidates: list[Path] = []
    try:
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*.png"):
                if path.is_file():
                    candidates.append(path)
    except Exception:
        return None

    if since_ts is not None:
        candidates = [p for p in candidates if _safe_stat_mtime(p) >= since_ts]
    if not candidates:
        return None

    candidates.sort(key=_safe_stat_mtime, reverse=True)
    return candidates[0]


def _short_text(text: str, limit: int = 1600) -> str:
    """Trim very long content for UI readability."""

    clean = text.strip()
    if len(clean) <= limit:
        return clean
    return f"{clean[:limit]} ...<truncated>"


def _task_log_metadata(text: str) -> dict[str, Any]:
    """Create non-content task metadata for privacy-preserving logs."""

    clean = text.strip()
    digest = hashlib.sha256(clean.encode("utf-8")).hexdigest() if clean else None
    return {"task_length": len(clean), "task_sha256": digest}


def _is_generation_control_command(text: str) -> bool:
    """Detect command-like messages that ask to use existing planning result."""

    raw = text.strip()
    if not raw:
        return False

    lowered = raw.lower()
    tokens = [
        "planning",
        "storyboard",
        "use plan",
        "generate now",
        "start generation",
        "根据分镜",
        "按分镜",
        "按planning",
        "上次规划",
        "上述规划",
        "开始生成",
        "生成视频",
    ]
    return any(token in lowered for token in tokens)


def _storyboard_to_readable_text(storyboard_text: str | None) -> str:
    """Convert storyboard JSON text into human-readable plain text."""

    if storyboard_text is None or not storyboard_text.strip():
        return "未生成有效分镜。"

    raw = storyboard_text.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    except Exception:
        return raw

    if not isinstance(parsed, list):
        return raw

    lines: list[str] = ["分镜规划结果："]
    for i, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            lines.append(f"{i}. {str(item)}")
            continue

        scene_no = item.get("scene_number", i)
        scene_slice = str(item.get("scene_slice", "")).strip()
        action = str(item.get("action", "")).strip()
        description = str(item.get("description", "")).strip()

        lines.append(f"{i}. 第 {scene_no} 幕")
        if scene_slice:
            lines.append(f"   时间段：{scene_slice}")
        if action:
            lines.append(f"   动作：{action}")
        if description:
            lines.append(f"   讲解内容：{description}")

    return "\n".join(lines)


async def _iterate_graph_events(graph_app: Any, state: AgentState):
    """Iterate graph events without blocking the Chainlit event loop."""

    if hasattr(graph_app, "astream"):
        async for event in graph_app.astream(state):
            yield event
        return

    event_queue: queue.Queue[Any] = queue.Queue()

    def _worker() -> None:
        try:
            for event in graph_app.stream(state):
                event_queue.put(event)
        except Exception as exc:
            event_queue.put(exc)
        finally:
            event_queue.put(None)

    threading.Thread(target=_worker, daemon=True).start()

    while True:
        item = await asyncio.to_thread(event_queue.get)
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def _emit_step(node_name: str, node_update: dict[str, Any], state: AgentState) -> None:
    """Render node-level output as Chainlit step."""

    retry = state.get("retry_count", 0)
    title_map: dict[str, str] = {
        "planner": "Planner",
        "coder": "Coder",
        "ast_reviewer": "AST Reviewer",
        "execution": "Execution",
        "vision_critic": "Vision Critic",
    }
    title = title_map.get(node_name, node_name)

    async with cl.Step(name=f"{title} (retry={retry})") as step:
        if node_name == "planner":
            story_text = _normalize_storyboard_text(node_update.get("storyboard")) or "No storyboard."
            step.output = _storyboard_to_readable_text(story_text)
            return

        if node_name == "coder":
            code = str(node_update.get("code", "")).strip()
            if not code:
                step.output = "Coder output is empty."
                step.is_error = True
                return
            step.output = f"```python\n{_short_text(code)}\n```"
            return

        if node_name == "ast_reviewer":
            ast_error = node_update.get("ast_error")
            if ast_error:
                step.output = f"AST failed:\n{_short_text(str(ast_error), 1200)}"
                step.is_error = True
            else:
                step.output = "AST passed."
            return

        if node_name == "execution":
            render_error = node_update.get("render_error")
            if render_error:
                step.output = f"Render failed:\n{_short_text(str(render_error), 1200)}"
                step.is_error = True
            else:
                step.output = "Render passed."
            return

        if node_name == "vision_critic":
            vision_error = node_update.get("vision_error")
            if vision_error:
                step.output = f"Vision not passed:\n{_short_text(str(vision_error), 1200)}"
                step.is_error = True
            else:
                step.output = "Vision passed."
            return

        step.output = _short_text(str(node_update), 1200)


async def _run_plan_only(state: AgentState) -> AgentState:
    """Run planner-only workflow and return updated state."""

    async for event in _iterate_graph_events(plan_only_app, state):
        for node_name, node_update in event.items():
            if isinstance(node_update, dict):
                _merge_state(state, node_update)
                await _emit_step(node_name, node_update, state)

    state["storyboard"] = _normalize_storyboard_text(state.get("storyboard"))
    return state


async def _run_generation(state: AgentState) -> AgentState:
    """Run full generation workflow and return updated state."""

    async for event in _iterate_graph_events(generate_app, state):
        for node_name, node_update in event.items():
            if isinstance(node_update, dict):
                _merge_state(state, node_update)
                await _emit_step(node_name, node_update, state)
    return state


def _node_summary(node_name: str, node_update: dict[str, Any]) -> dict[str, Any]:
    """Create compact node summary payload for experiment logs."""

    summary: dict[str, Any] = {"node": node_name}
    if node_name == "planner":
        text = _normalize_storyboard_text(node_update.get("storyboard")) or ""
        summary["storyboard_length"] = len(text)
        return summary
    if node_name == "coder":
        summary["code_length"] = len(str(node_update.get("code", "")))
        summary["coder_input_mode"] = node_update.get("coder_input_mode")
        summary["coder_storyboard_used"] = node_update.get("coder_storyboard_used")
        return summary
    if node_name == "ast_reviewer":
        ast_error = node_update.get("ast_error")
        summary["ast_error"] = _short_text(str(ast_error), 500) if ast_error else None
        return summary
    if node_name == "execution":
        render_error = node_update.get("render_error")
        summary["render_error"] = _short_text(str(render_error), 500) if render_error else None
        summary["failure_type"] = node_update.get("failure_type")
        summary["execution_environment"] = node_update.get("execution_environment")
        return summary
    if node_name == "vision_critic":
        vision_error = node_update.get("vision_error")
        summary["vision_error"] = _short_text(str(vision_error), 500) if vision_error else None
        summary["vision_verdict"] = node_update.get("vision_verdict")
        summary["vision_severity"] = node_update.get("vision_severity")
        return summary
    summary["raw"] = str(node_update)[:500]
    return summary


def _reset_run_fields(state: AgentState) -> None:
    """Reset run-specific mutable fields to avoid cross-run contamination."""

    state["code"] = ""
    state["ast_error"] = None
    state["render_error"] = None
    state["vision_error"] = None
    state["retry_count"] = 0
    state["render_media_dir"] = None
    state["render_image_path"] = None
    state["render_video_path"] = None
    state["failure_stage"] = None
    state["failure_type"] = None
    state["final_verdict"] = "unknown"
    state["success_reason"] = None
    state["failure_reason"] = None
    state["vision_skipped"] = False
    state["vision_verdict"] = None
    state["vision_severity"] = None
    state["vision_issue_count"] = 0
    state["storyboard_present"] = bool(state.get("storyboard"))
    state["coder_input_mode"] = "unknown"
    state["coder_storyboard_used"] = False
    state["retry_count_reason"] = "coder attempt count"


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize session state and chat settings."""

    initial_state = create_initial_state("")
    _safe_set_session_state(initial_state)

    settings_value: dict[str, Any] = {MODE_KEY: GENERATION_MODE}
    try:
        settings = await cl.ChatSettings(
            [
                Select(
                    id=MODE_KEY,
                    label=MODE_KEY,
                    values=[PLANNING_MODE, GENERATION_MODE],
                    initial_index=1,
                )
            ]
        ).send()
        if isinstance(settings, dict):
            settings_value.update(settings)
    except Exception:
        # Keep default mode if settings UI fails.
        pass

    cl.user_session.set("chat_settings", settings_value)

    await cl.Message(
        content=(
            "Dual-mode workflow is ready.\n"
            "Select the running mode from chat settings."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, Any]) -> None:
    """Persist settings whenever user updates mode selector."""

    try:
        if isinstance(settings, dict):
            cl.user_session.set("chat_settings", settings)
    except Exception:
        pass


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle chat message under current running mode."""

    state = _safe_get_session_state()
    user_text = message.content.strip()
    mode = _get_current_mode()
    run_started_at = time.time()
    logger = ExperimentLogger()
    logger.log(
        "request_received",
        {"mode": mode, **_task_log_metadata(user_text), "has_storyboard": bool(state.get("storyboard"))},
    )

    existing_storyboard = bool(state.get("storyboard"))

    if mode == PLANNING_MODE:
        if user_text:
            state["task"] = user_text
    else:
        if user_text:
            if existing_storyboard and _is_generation_control_command(user_text):
                # Keep original planning task + storyboard.
                pass
            elif existing_storyboard:
                # Treat as a new request and clear old storyboard to avoid mismatch.
                state["task"] = user_text
                state["storyboard"] = None
                await cl.Message(
                    content="Detected a new request. Previous storyboard is cleared and will be re-planned."
                ).send()
            else:
                state["task"] = user_text
    _reset_run_fields(state)

    if mode == PLANNING_MODE:
        progress_msg = cl.Message(content="Planning mode: planner-only graph.")
        await progress_msg.send()
        try:
            async for event in _iterate_graph_events(plan_only_app, state):
                for node_name, node_update in event.items():
                    if isinstance(node_update, dict):
                        _merge_state(state, node_update)
                        logger.log("node_update", _node_summary(node_name, node_update))
                        await _emit_step(node_name, node_update, state)
            state["storyboard"] = _normalize_storyboard_text(state.get("storyboard"))
            _safe_set_session_state(state)
            story_text = state.get("storyboard") or "Planner returned no storyboard."
            human_story = _storyboard_to_readable_text(story_text)
            logger.log("planning_finished", {"storyboard_length": len(story_text)})
            progress_msg.content = "Planning 完成，以下是可读分镜文本。"
            progress_msg.elements = [
                cl.Text(name="Storyboard", content=human_story, language="text")
            ]
            await progress_msg.update()
        except Exception as exc:
            logger.log("planning_failed", {"error": str(exc)})
            progress_msg.content = f"Planning failed: {exc}"
            await progress_msg.update()
        return

    progress_msg = cl.Message(content="Generation mode: run the full feedback loop.")
    await progress_msg.send()

    try:
        # Scenario B: direct generation without prior storyboard.
        if not state.get("storyboard"):
            await cl.Message(content="No previous storyboard. Run planner first.").send()
            async for event in _iterate_graph_events(plan_only_app, state):
                for node_name, node_update in event.items():
                    if isinstance(node_update, dict):
                        _merge_state(state, node_update)
                        logger.log("node_update", _node_summary(node_name, node_update))
                        await _emit_step(node_name, node_update, state)
            state["storyboard"] = _normalize_storyboard_text(state.get("storyboard"))
            logger.log(
                "autoplanning_finished",
                {"storyboard_length": len(state.get("storyboard") or "")},
            )

        async for event in _iterate_graph_events(generate_app, state):
            for node_name, node_update in event.items():
                if isinstance(node_update, dict):
                    _merge_state(state, node_update)
                    logger.log("node_update", _node_summary(node_name, node_update))
                    await _emit_step(node_name, node_update, state)
        _safe_set_session_state(state)
    except Exception as exc:
        logger.log("generation_failed", {"error": str(exc)})
        progress_msg.content = f"Generation failed: {exc}"
        await progress_msg.update()
        return

    state_video = state.get("render_video_path")
    latest_video: Path | None = None
    if isinstance(state_video, str) and state_video.strip():
        candidate_video = Path(state_video)
        if candidate_video.exists() and candidate_video.is_file():
            latest_video = candidate_video
    if latest_video is None:
        latest_video = _find_latest_video(since_ts=run_started_at)

    if latest_video is not None:
        logger.log(
            "generation_finished",
            {
                "artifact_type": "video",
                "artifact_path": str(latest_video),
                "retry_count": state.get("retry_count"),
                "final_verdict": state.get("final_verdict"),
                "failure_stage": state.get("failure_stage"),
                "failure_type": state.get("failure_type"),
                "success_reason": state.get("success_reason"),
                "failure_reason": state.get("failure_reason"),
                "execution_environment": state.get("execution_environment"),
            },
        )
        progress_msg.content = "Generation finished. Video has been created."
        progress_msg.elements = [
            cl.Video(name="Generated Video", path=str(latest_video), display="inline")
        ]
        await progress_msg.update()
        return

    state_image = state.get("render_image_path")
    latest_image: Path | None = None
    if isinstance(state_image, str) and state_image.strip():
        candidate_image = Path(state_image)
        if candidate_image.exists() and candidate_image.is_file():
            latest_image = candidate_image
    if latest_image is None:
        latest_image = _find_latest_image(since_ts=run_started_at)

    if latest_image is not None:
        logger.log(
            "generation_finished",
            {
                "artifact_type": "image",
                "artifact_path": str(latest_image),
                "retry_count": state.get("retry_count"),
                "final_verdict": state.get("final_verdict"),
                "failure_stage": state.get("failure_stage"),
                "failure_type": state.get("failure_type"),
                "success_reason": state.get("success_reason"),
                "failure_reason": state.get("failure_reason"),
                "execution_environment": state.get("execution_environment"),
            },
        )
        progress_msg.content = "Generation finished. No new mp4 found, showing latest frame."
        progress_msg.elements = [
            cl.Image(name="Rendered Frame", path=str(latest_image), display="inline")
        ]
        await progress_msg.update()
        return

    logger.log(
        "generation_finished",
        {
            "artifact_type": "none",
            "retry_count": state.get("retry_count"),
            "final_verdict": state.get("final_verdict"),
            "failure_stage": state.get("failure_stage"),
            "failure_type": state.get("failure_type"),
            "success_reason": state.get("success_reason"),
            "failure_reason": state.get("failure_reason"),
            "execution_environment": state.get("execution_environment"),
        },
    )
    progress_msg.content = "Generation finished, but no new artifact was found."
    await progress_msg.update()
