from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from state import AgentState
from utils.model_provider import build_qwen_vision_model

LOGGER = logging.getLogger(__name__)

VISION_PROMPT_TEMPLATE = """
You are a strict Manim layout and semantic critic.

Task:
{task}

Review the image and output STRICT JSON only with this schema:
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
1) Return verdict=OK only when all checks pass:
   - No overlap for red-bounded elements.
   - No out-of-screen elements.
   - Core semantics align with the task.
2) If minor and acceptable issue exists, set severity=low and provide one concise fix.
3) Do not output markdown.
""".strip()


def _build_vision_llm() -> ChatOpenAI:
    """Create multimodal vision model client routed to Qwen."""

    return build_qwen_vision_model(
        model_env_name="MANIM_VISION_MODEL",
        default_model="qwen-vl-max",
        temperature_env_name="MANIM_VISION_TEMPERATURE",
        default_temperature=0.1,
    )


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


def vision_critic_node(state: AgentState) -> dict[str, str | None]:
    """Vision critic node for overlap/out-of-screen/semantic feedback."""
    # 如果当前策略是纯运行时测试，直接假装视觉完美，不调用 Qwen-VL
    if state.get("strategy") == "Runtime Only":
        return {"vision_error": None, "vlm_iou_score": 0.0}
    try:
        image_path = _image_from_state(state)
        if image_path is None:
            roots: list[Path] = []
            render_dir = state.get("render_media_dir")
            if isinstance(render_dir, str) and render_dir.strip():
                roots.append(Path(render_dir))
            roots.append(Path("media/runs"))
            image_path = _find_latest_render_image(roots)
        if image_path is None:
            return {"vision_error": "No rendered image found for current execution run."}
        encoded_image = _encode_image_base64(image_path)
    except OSError as exc:
        return {"vision_error": f"Failed to read rendered image: {exc}"}
    except Exception as exc:
        LOGGER.exception("vision_critic_node preprocessing failed: %s", exc)
        return {"vision_error": f"Vision preprocessing failed: {exc}"}

    task = str(state.get("task", "")).strip() or "No task provided."
    prompt = VISION_PROMPT_TEMPLATE.format(task=task)
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
            },
        ]
    )

    try:
        llm = _build_vision_llm()
        response = llm.invoke([message])
        feedback_text = _extract_text_content(response.content).strip()
    except Exception as exc:
        LOGGER.exception("Vision model invocation failed: %s", exc)
        return {"vision_error": f"Vision model call failed: {exc}"}

    feedback_obj = _extract_json_object(feedback_text)
    
    # 定义严重程度到分数的映射
    severity_map = {"low": 0.2, "medium": 0.5, "high": 0.9}
    
    if feedback_obj is None:
        if feedback_text.upper() == "OK":
            return {"vision_error": None, "vlm_iou_score": 0.0} # 无问题，得分为0
        return {"vision_error": feedback_text, "vlm_iou_score": 0.5}

    # 提取严重程度并计算分数
    severity = feedback_obj.get("severity", "low").lower()
    issues_count = len(feedback_obj.get("issues", []))
    
    # 结合严重程度和问题数量计算最终得分 (x_vlm)
    # 逻辑：基础分数 + 额外问题修正，上限为 1.0
    base_score = severity_map.get(severity, 0.2)
    vlm_score = min(1.0, base_score + (issues_count * 0.05))

    return {
        "vision_error": feedback_text if feedback_obj.get("verdict") != "OK" else None,
        "vlm_iou_score": vlm_score # 👈 关键：将视觉冲突分数写回状态
    }