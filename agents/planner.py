from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from state import AgentState
from utils.model_provider import build_deepseek_chat_model

LOGGER = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """
You are a storyboard planner for Manim educational videos.

Output strictly one JSON array, without markdown and without extra text.
Each item must contain:
- scene_number: int
- scene_slice: str
- action: str
- description: str
""".strip()


def _build_planner_llm() -> ChatOpenAI:
    """Create planner model client using ChatOpenAI-compatible DeepSeek route."""

    return build_deepseek_chat_model(
        model_env_name="MANIM_PLANNER_MODEL",
        default_model="deepseek-chat",
        temperature_env_name="MANIM_PLANNER_TEMPERATURE",
        default_temperature=0.1,
    )


def _extract_text_content(content: Any) -> str:
    """Extract plain text from model response payload."""

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


def _strip_code_fence(text: str) -> str:
    """Remove markdown fences from possible model output."""

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _try_parse_json_array(text: str) -> list[dict[str, Any]] | None:
    """Try parse JSON array from text."""

    try:
        value = json.loads(text)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[[\s\S]*\]", text)
    if match is None:
        return None

    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return None


def _normalize_storyboard(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize storyboard schema and enforce required fields."""

    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        normalized.append(
            {
                "scene_number": index,
                "scene_slice": str(item.get("scene_slice", f"Scene {index}")).strip() or f"Scene {index}",
                "action": str(item.get("action", "Write")).strip() or "Write",
                "description": str(item.get("description", "Explain key point.")).strip() or "Explain key point.",
            }
        )
    return normalized


def _fallback_storyboard(task: str) -> list[dict[str, Any]]:
    """Provide fallback storyboard when planner model fails."""

    concise = task.strip() or "Generate an educational animation."
    return [
        {
            "scene_number": 1,
            "scene_slice": "Opening",
            "action": "Write",
            "description": f"Introduce the topic: {concise}",
        },
        {
            "scene_number": 2,
            "scene_slice": "Core Demonstration",
            "action": "Create",
            "description": "Demonstrate the core concept visually.",
        },
        {
            "scene_number": 3,
            "scene_slice": "Summary",
            "action": "FadeOut",
            "description": "Summarize and conclude the lesson.",
        },
    ]


def planner_node(state: AgentState) -> dict[str, str | None]:
    """Planner agent: natural language task -> storyboard JSON text."""

    try:
        task = str(state.get("task", "")).strip()
    except Exception:
        task = ""

    if not task:
        fallback = _fallback_storyboard("Empty task input.")
        return {"storyboard": json.dumps(fallback, ensure_ascii=False, indent=2)}

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", PLANNER_SYSTEM_PROMPT),
            ("user", "User task:\n{task}\n\nReturn storyboard JSON array only."),
        ]
    )

    try:
        llm = _build_planner_llm()
        response = (prompt | llm).invoke({"task": task})
        raw_text = _extract_text_content(response.content)
        cleaned_text = _strip_code_fence(raw_text)
        parsed = _try_parse_json_array(cleaned_text)
        if not parsed:
            raise ValueError("Planner did not return a valid JSON array.")

        normalized = _normalize_storyboard(parsed)
        storyboard_text = json.dumps(normalized, ensure_ascii=False, indent=2)
        return {"storyboard": storyboard_text}
    except Exception as exc:
        LOGGER.exception("planner_node failed: %s", exc)
        fallback = _fallback_storyboard(task)
        return {"storyboard": json.dumps(fallback, ensure_ascii=False, indent=2)}

