from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from state import AgentState
from utils.model_provider import (
    build_qwen_vision_model,
    get_vision_timeout_seconds,
    invoke_with_hard_timeout,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_MAX_VISION_KEYFRAMES = 4
DEFAULT_VISION_FRAME_WIDTH = 960
DEFAULT_FRAME_EXTRACT_IMAGE = "manimcommunity/manim:stable"

VISION_PROMPT_TEMPLATE = """
You are a careful Manim visual critic for rendered keyframes from one animation.

Task:
{task}

Review all provided images in chronological order and output STRICT JSON only with this schema:
{{
  "verdict": "OK" | "REVISE",
  "severity": "low" | "medium" | "high",
  "issues": [
    {{
      "target": "object name or area",
      "problem": "what is wrong",
      "fix": "specific manim fix using shift/next_to/buff or object replacement"
    }}
  ],
  "global_fix": "short global instruction"
}}

Rules:
1) Return verdict=OK when the animation is usable for teaching and the remaining issues are none or low severity.
2) Use verdict=REVISE when there is a medium/high severity issue, missing core semantics, severe overlap, or clearly off-screen content.
3) If minor and acceptable issues exist, set severity=low and keep issues concise and actionable.
4) For animation tasks, do not require every object to appear in every single frame; judge whether it appears in the appropriate keyframe.
5) Prefer concrete layout fixes over abstract semantic criticism.
6) Recommend only standard Manim Community APIs. Never mention nonstandard names like TrailEffect, PathTracer, FadingTrail, or TransformWithRotation. For fading trails, recommend TracedPath with dissipating_time parameter. For rotation, recommend Rotate.
7) Do not output markdown.
""".strip()


def _build_vision_llm() -> ChatOpenAI:
    """Create multimodal vision model client routed to Qwen."""

    return build_qwen_vision_model(
        model_env_name="MANIM_VISION_MODEL",
        default_model="qwen-vl-max",
        temperature_env_name="MANIM_VISION_TEMPERATURE",
        default_temperature=0.1,
    )


def _safe_int_env(name: str, default: int) -> int:
    """Read positive integer env value with fallback."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _configured_max_keyframes() -> int:
    """Return bounded keyframe count for vision review."""

    return min(6, max(1, _safe_int_env("MANIM_VISION_KEYFRAMES", DEFAULT_MAX_VISION_KEYFRAMES)))


def _configured_frame_width() -> int:
    """Return bounded keyframe width for vision review."""

    return min(1280, max(320, _safe_int_env("MANIM_VISION_FRAME_WIDTH", DEFAULT_VISION_FRAME_WIDTH)))


def _extract_text_content(content: Any) -> str:
    """Extract plain text from multimodal response payload."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)
        return "\n".join(chunk for chunk in chunks if chunk.strip())
    return str(content)


def _find_latest_render_image(search_roots: list[Path]) -> Path | None:
    """Find latest image file from provided roots."""

    candidates: list[Path] = []
    try:
        for root in search_roots:
            if not root.exists():
                continue
            candidates.extend(path for path in root.rglob("*.png") if path.is_file())
    except Exception as exc:
        LOGGER.exception("Failed while scanning render images: %s", exc)
        return None

    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _video_from_state(state: AgentState) -> Path | None:
    """Resolve rendered video path directly from state when available."""

    try:
        video_path = state.get("render_video_path")
        if isinstance(video_path, str) and video_path.strip():
            candidate = Path(video_path)
            if candidate.exists() and candidate.is_file():
                return candidate
    except Exception:
        return None
    return None


def _image_from_state(state: AgentState) -> Path | None:
    """Resolve image path directly from state when available."""

    try:
        image_path = state.get("render_image_path")
        if isinstance(image_path, str) and image_path.strip():
            candidate = Path(image_path)
            if candidate.exists() and candidate.is_file():
                return candidate
    except Exception:
        return None
    return None


def _frame_dir_from_state(state: AgentState) -> Path:
    """Choose a per-run directory for extracted vision keyframes."""

    render_dir = state.get("render_media_dir")
    if isinstance(render_dir, str) and render_dir.strip():
        return Path(render_dir) / "vision_keyframes"
    return Path("media/runs/vision_keyframes")


def _extract_video_keyframes(video_path: Path, output_dir: Path) -> list[Path]:
    """Extract a small chronological keyframe set from a rendered video."""

    max_keyframes = _configured_max_keyframes()
    frame_width = _configured_frame_width()

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in output_dir.glob("frame_*.png"):
            if old_frame.is_file():
                old_frame.unlink()
    except Exception as exc:
        LOGGER.warning("Failed to prepare keyframe directory: %s", exc)
        return []

    output_pattern = output_dir / "frame_%02d.png"
    host_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps=1,scale={frame_width}:-1",
        "-frames:v",
        str(max_keyframes),
        str(output_pattern),
    ]

    def _run_extract(command: list[str]) -> bool:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
                check=False,
            )
        except FileNotFoundError:
            return False
        except subprocess.TimeoutExpired:
            return False
        except Exception as exc:
            LOGGER.warning("Keyframe extraction command failed: %s", exc)
            return False
        return result.returncode == 0

    if not _run_extract(host_command):
        docker_command = _docker_ffmpeg_command(
            video_path,
            output_dir,
            output_pattern,
            max_keyframes=max_keyframes,
            frame_width=frame_width,
        )
        if docker_command is None or not _run_extract(docker_command):
            LOGGER.warning("Keyframe extraction failed; falling back to still image review.")
            return []

    frames = sorted(path for path in output_dir.glob("frame_*.png") if path.is_file())
    return frames[:max_keyframes]


def _docker_ffmpeg_command(
    video_path: Path,
    output_dir: Path,
    output_pattern: Path,
    *,
    max_keyframes: int,
    frame_width: int,
) -> list[str] | None:
    """Build Docker ffmpeg fallback command when host ffmpeg is unavailable."""

    try:
        run_root = output_dir.parent.resolve()
        rel_video = video_path.resolve().relative_to(run_root).as_posix()
        rel_output_pattern = output_pattern.resolve().relative_to(run_root).as_posix()
    except Exception as exc:
        LOGGER.warning("Failed to build Docker ffmpeg paths: %s", exc)
        return None

    image = os.getenv("MANIM_SANDBOX_IMAGE", DEFAULT_FRAME_EXTRACT_IMAGE).strip() or DEFAULT_FRAME_EXTRACT_IMAGE
    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network",
        "none",
        "-v",
        f"{run_root}:/workspace:rw",
        "-w",
        "/workspace",
        image,
        "ffmpeg",
        "-y",
        "-i",
        rel_video,
        "-vf",
        f"fps=1,scale={frame_width}:-1",
        "-frames:v",
        str(max_keyframes),
        rel_output_pattern,
    ]


def _encode_image_base64(path: Path) -> str:
    """Convert image file to base64 string."""

    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract first JSON object from text."""

    raw = text.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", raw)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_feedback(feedback_obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize feedback schema for downstream coder."""

    verdict = str(feedback_obj.get("verdict", "REVISE")).strip().upper()
    if verdict not in {"OK", "REVISE"}:
        verdict = "REVISE"

    severity = str(feedback_obj.get("severity", "medium")).strip().lower()
    if severity not in {"low", "medium", "high"}:
        severity = "medium"

    issues_raw = feedback_obj.get("issues", [])
    issues: list[dict[str, str]] = []
    if isinstance(issues_raw, list):
        for item in issues_raw[:5]:
            if not isinstance(item, dict):
                continue
            issues.append(
                {
                    "target": str(item.get("target", "")).strip(),
                    "problem": str(item.get("problem", "")).strip(),
                    "fix": str(item.get("fix", "")).strip(),
                }
            )

    global_fix = str(feedback_obj.get("global_fix", "")).strip()
    return {
        "verdict": verdict,
        "severity": severity,
        "issues": issues,
        "global_fix": global_fix,
    }


def _review_images_from_state(state: AgentState) -> list[Path]:
    """Resolve video keyframes first, then fallback to a single rendered image."""

    video_path = _video_from_state(state)
    if video_path is not None:
        frames = _extract_video_keyframes(video_path, _frame_dir_from_state(state))
        if frames:
            return frames

    image_path = _image_from_state(state)
    if image_path is None:
        roots: list[Path] = []
        render_dir = state.get("render_media_dir")
        if isinstance(render_dir, str) and render_dir.strip():
            roots.append(Path(render_dir))
        roots.append(Path("media/runs"))
        image_path = _find_latest_render_image(roots)

    return [image_path] if image_path is not None else []


def _build_vision_message(task: str, encoded_images: list[str]) -> HumanMessage:
    """Build one multimodal vision request from ordered keyframes."""

    prompt = VISION_PROMPT_TEMPLATE.format(task=task)
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for idx, encoded_image in enumerate(encoded_images, start=1):
        content.append({"type": "text", "text": f"Keyframe {idx} of {len(encoded_images)}"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
            }
        )
    return HumanMessage(content=content)


def _invoke_vision_model(llm: ChatOpenAI, message: HumanMessage) -> str:
    """Invoke Qwen vision with a hard timeout guard."""

    response = invoke_with_hard_timeout(
        lambda: llm.invoke([message]),
        timeout_seconds=get_vision_timeout_seconds() + 5,
        timeout_label="vision model call",
    )
    return _extract_text_content(response.content).strip()


def _vision_failure_patch(message: str, *, failure_type: str) -> dict[str, Any]:
    """Build structured vision failure payload."""

    return {
        "vision_error": message,
        "failure_stage": "vision",
        "failure_type": failure_type,
        "failure_reason": message,
        "vision_verdict": None,
        "vision_severity": None,
        "vision_issue_count": 0,
        "vision_feedback_raw": message,
    }


def _normalized_vision_success_patch(
    normalized: dict[str, Any],
    *,
    feedback_text: str,
    vlm_score: float,
) -> dict[str, Any]:
    """Build structured success payload from normalized vision feedback."""

    severity = normalized["severity"]
    issues_count = len(normalized["issues"])
    return {
        "vision_error": None,
        "vlm_iou_score": vlm_score,
        "failure_stage": None,
        "failure_type": None,
        "failure_reason": None,
        "vision_verdict": "OK",
        "vision_severity": severity,
        "vision_issue_count": issues_count,
        "vision_feedback_raw": feedback_text,
    }


def vision_critic_node(state: AgentState) -> dict[str, Any]:
    """Vision critic node for overlap/out-of-screen/semantic feedback."""

    try:
        image_paths = _review_images_from_state(state)
        if not image_paths:
            return _vision_failure_patch(
                "No rendered image or video keyframe found for current execution run.",
                failure_type="content",
            )
    except OSError as exc:
        return _vision_failure_patch(f"Failed to read rendered visual artifact: {exc}", failure_type="infra")
    except Exception as exc:
        LOGGER.exception("vision_critic_node preprocessing failed: %s", exc)
        return _vision_failure_patch(f"Vision preprocessing failed: {exc}", failure_type="infra")

    task = str(state.get("task", "")).strip() or "No task provided."
    llm: ChatOpenAI | None = None

    try:
        llm = _build_vision_llm()
        LOGGER.info(
            "Vision critic started for strategy=%s frames=%s",
            state.get("strategy", "unknown"),
            len(image_paths),
        )
        encoded_images = [_encode_image_base64(path) for path in image_paths]
        feedback_text = _invoke_vision_model(llm, _build_vision_message(task, encoded_images))
    except Exception as exc:
        if len(image_paths) > 1 and llm is not None:
            LOGGER.warning("Multi-frame vision review failed, retrying single frame: %s", exc)
            try:
                fallback_frame = image_paths[-1]
                fallback_encoded = [_encode_image_base64(fallback_frame)]
                feedback_text = _invoke_vision_model(llm, _build_vision_message(task, fallback_encoded))
            except Exception as fallback_exc:
                LOGGER.exception("Vision model invocation failed after single-frame fallback: %s", fallback_exc)
                return _vision_failure_patch(
                    f"Vision model call failed after single-frame fallback: {fallback_exc}",
                    failure_type="infra",
                )
        else:
            LOGGER.exception("Vision model invocation failed: %s", exc)
            return _vision_failure_patch(f"Vision model call failed: {exc}", failure_type="infra")

    feedback_obj = _extract_json_object(feedback_text)
    severity_map = {"low": 0.2, "medium": 0.5, "high": 0.9}

    if feedback_obj is None:
        if feedback_text.upper() == "OK":
            return {
                "vision_error": None,
                "vlm_iou_score": 0.0,
                "vision_verdict": "OK",
                "vision_severity": None,
                "vision_issue_count": 0,
                "vision_feedback_raw": feedback_text,
            }
        return {
            "vision_error": feedback_text,
            "vlm_iou_score": 0.5,
            "failure_stage": "vision",
            "failure_type": "content",
            "failure_reason": feedback_text,
            "vision_verdict": "REVISE",
            "vision_severity": None,
            "vision_issue_count": 0,
            "vision_feedback_raw": feedback_text,
        }

    normalized = _normalize_feedback(feedback_obj)
    severity = normalized["severity"]
    issues_count = len(normalized["issues"])
    vlm_score = min(1.0, severity_map.get(severity, 0.2) + (issues_count * 0.05))
    verdict = normalized["verdict"]

    if verdict == "OK":
        return _normalized_vision_success_patch(
            normalized,
            feedback_text=feedback_text,
            vlm_score=vlm_score,
        )

    return {
        "vision_error": feedback_text,
        "vlm_iou_score": vlm_score,
        "failure_stage": "vision",
        "failure_type": "content",
        "failure_reason": feedback_text,
        "vision_verdict": verdict,
        "vision_severity": severity,
        "vision_issue_count": issues_count,
        "vision_feedback_raw": feedback_text,
    }
