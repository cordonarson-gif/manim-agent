from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from state import AgentState, MAX_RETRIES
from utils.model_provider import build_deepseek_chat_model

LOGGER = logging.getLogger(__name__)

CODER_SYSTEM_PROMPT = """
You are a senior Manim code generation and repair expert.
Return pure Python code only (no markdown fences).

Mandatory constraints:
1) Must include: from manim import *
2) Main class name must be exactly: GeneratedScene
3) Must be executable and syntactically valid Python.
4) If storyboard_json is provided, storyboard is authoritative and must be followed in order.
5) If feedback exists, apply all required fixes explicitly.
6) Keep layout conservative with shift()/next_to()/buff to avoid overlap and out-of-screen.
7) Avoid 3D objects (Cube/Sphere/Cylinder/Cone/ThreeDScene) unless explicitly requested.
8) Prefer <= 4 key scenes and <= 10 total self.play calls for reliable convergence.
""".strip()

SHAPE_KEYWORDS: list[tuple[str, str]] = [
    ("triangle", "Triangle"),
    ("\u4e09\u89d2", "Triangle"),
    ("square", "Square"),
    ("\u6b63\u65b9\u5f62", "Square"),
    ("rectangle", "Rectangle"),
    ("\u957f\u65b9\u5f62", "Rectangle"),
    ("circle", "Circle"),
    ("\u5706", "Circle"),
]

TRANSFORM_KEYWORDS: tuple[str, ...] = (
    "transform",
    "morph",
    "to ",
    "\u53d8\u6210",
    "\u53d8\u4e3a",
    "\u8f6c\u6210",
    "\u8f6c\u5316",
    "\u8f6c\u6362",
)

THREE_D_HINTS: tuple[str, ...] = (
    "cube",
    "sphere",
    "cylinder",
    "cone",
    "3d",
    "\u7acb\u4f53",
    "\u7acb\u65b9\u4f53",
    "\u4e09\u7ef4",
)


def _build_coder_llm() -> ChatOpenAI:
    """Create coder model client using ChatOpenAI-compatible DeepSeek route."""

    return build_deepseek_chat_model(
        model_env_name="MANIM_CODER_MODEL",
        default_model="deepseek-chat",
        temperature_env_name="MANIM_CODER_TEMPERATURE",
        default_temperature=0.15,
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


def _strip_markdown_fence(text: str) -> str:
    """Remove markdown code fences from model output."""

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:python)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def _ensure_required_import(code: str) -> str:
    """Ensure mandatory Manim wildcard import exists."""

    if "from manim import *" in code:
        return code
    return f"from manim import *\n\n{code.lstrip()}"


def _has_generated_scene(code: str) -> bool:
    """Check whether code contains class GeneratedScene."""

    return bool(re.search(r"^\s*class\s+GeneratedScene\s*(?:\(|:)", code, flags=re.MULTILINE))


def _parse_storyboard_text(storyboard_text: str | None) -> tuple[bool, str, list[dict[str, Any]]]:
    """Return (is_json_storyboard, normalized_text, storyboard_items)."""

    if storyboard_text is None:
        return False, "", []

    raw = storyboard_text.strip()
    if not raw:
        return False, "", []

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            items = [item for item in parsed if isinstance(item, dict)]
            return True, json.dumps(items, ensure_ascii=False, indent=2), items
    except json.JSONDecodeError:
        pass
    except Exception:
        pass

    return False, raw, []


def _normalize_vision_feedback(vision_error: str | None) -> str:
    """Normalize vision feedback into deterministic text for repair prompt."""

    if vision_error is None:
        return ""

    text = vision_error.strip()
    if not text:
        return ""

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    except Exception:
        return text

    if not isinstance(parsed, dict):
        return text

    verdict = str(parsed.get("verdict", "REVISE"))
    severity = str(parsed.get("severity", "unknown"))
    lines = [f"verdict={verdict}", f"severity={severity}"]

    issues = parsed.get("issues", [])
    if isinstance(issues, list):
        for idx, issue in enumerate(issues, start=1):
            if not isinstance(issue, dict):
                continue
            problem = str(issue.get("problem", "")).strip()
            fix = str(issue.get("fix", "")).strip()
            if problem:
                lines.append(f"issue_{idx}: {problem}")
            if fix:
                lines.append(f"fix_{idx}: {fix}")

    global_fix = str(parsed.get("global_fix", "")).strip()
    if global_fix:
        lines.append(f"global_fix: {global_fix}")

    return "\n".join(lines)


def _build_retry_strategy(retry_count: int) -> str:
    """Build round-aware repair strategy instructions."""

    if retry_count <= 1:
        return "Round policy: balanced quality mode."
    if retry_count <= 3:
        return (
            "Round policy: conservative layout mode. Use VGroup().arrange with buff>=0.6, "
            "scale_to_fit_width(<=12), and avoid edge clipping."
        )
    return (
        "Round policy: emergency convergence mode. Use minimal objects, no complex loops, "
        "and prioritize executable + visible output."
    )


def _build_feedback_block(state: AgentState) -> str:
    """Build repair feedback context from prior failures."""

    feedbacks: list[str] = []
    ast_error = state.get("ast_error")
    render_error = state.get("render_error")
    vision_error = _normalize_vision_feedback(state.get("vision_error"))

    if ast_error:
        feedbacks.append(f"[ast_error]\n{ast_error}")
    if render_error:
        feedbacks.append(f"[render_error]\n{render_error}")
    if vision_error and vision_error.strip().upper() != "OK":
        feedbacks.append(
            "[vision_error]\n"
            f"{vision_error}\n"
            "Apply each fix item explicitly in updated code."
        )

    if not feedbacks:
        return "No historical AST/render/vision errors."
    return "\n\n".join(feedbacks)


def _detected_shapes(text: str) -> list[str]:
    """Detect intended geometric shapes from task/storyboard text."""

    hay = text.lower()
    indexed: list[tuple[int, str]] = []
    for keyword, shape in SHAPE_KEYWORDS:
        idx = hay.find(keyword.lower())
        if idx >= 0:
            indexed.append((idx, shape))

    indexed.sort(key=lambda x: x[0])
    ordered: list[str] = []
    for _, shape in indexed:
        if shape not in ordered:
            ordered.append(shape)
    return ordered


def _shape_ctor(shape_name: str) -> str:
    """Return constructor expression for a target shape."""

    constructors = {
        "Triangle": "Triangle(color=BLUE)",
        "Square": "Square(color=BLUE)",
        "Rectangle": "Rectangle(width=4.0, height=2.4, color=BLUE)",
        "Circle": "Circle(color=BLUE)",
    }
    return constructors.get(shape_name, "Triangle(color=BLUE)")


def _has_transform_intent(text: str) -> bool:
    """Detect transform intent from text."""

    lower_text = text.lower()
    return any(token in lower_text for token in TRANSFORM_KEYWORDS)


def _wants_3d(text: str) -> bool:
    """Detect whether source text explicitly requests 3D content."""

    lower_text = text.lower()
    return any(token in lower_text for token in THREE_D_HINTS)


def _safe_literal(text: str, max_len: int = 80) -> str:
    """Escape text for safe single-quoted Python literal."""

    normalized = re.sub(r"\s+", " ", text).strip()
    clipped = normalized[:max_len] if normalized else "Generated educational animation"
    return clipped.replace("\\", "\\\\").replace("'", "\\'")


def _storyboard_scene_labels(items: list[dict[str, Any]], limit: int) -> list[str]:
    """Create concise scene labels from storyboard items."""

    labels: list[str] = []
    for idx, item in enumerate(items[:limit], start=1):
        action = str(item.get("action", "")).strip()
        description = str(item.get("description", "")).strip()
        if description:
            labels.append(f"Scene {idx}: {description[:36]}")
        elif action:
            labels.append(f"Scene {idx}: {action[:36]}")
        else:
            labels.append(f"Scene {idx}: Key step")
    return labels


def _fallback_code(task: str, storyboard_text: str, storyboard_items: list[dict[str, Any]], retry_count: int) -> str:
    """Build deterministic fallback code with semantic adaptation."""

    source = f"{task}\n{storyboard_text}"
    shapes = _detected_shapes(source)
    has_transform = _has_transform_intent(source)

    primary = shapes[0] if shapes else "Triangle"
    secondary = shapes[1] if len(shapes) > 1 else "Square"

    concise = _safe_literal(task, max_len=72)
    max_labels = 3 if retry_count < 4 else 1
    labels = _storyboard_scene_labels(storyboard_items, limit=max_labels)

    lines: list[str] = [
        "from manim import *",
        "",
        "class GeneratedScene(Scene):",
        "    def construct(self):",
        f"        title = Text('{concise}', font_size=34).scale_to_fit_width(12).to_edge(UP)",
        "        self.play(FadeIn(title), run_time=0.8)",
        f"        shape_a = {_shape_ctor(primary)}.scale(1.1).shift(DOWN * 0.2)",
        "        self.play(Create(shape_a), run_time=1.2)",
    ]

    if has_transform:
        lines.extend(
            [
                f"        shape_b = {_shape_ctor(secondary)}.scale(1.1).move_to(shape_a)",
                "        self.play(Transform(shape_a, shape_b), run_time=1.2)",
            ]
        )

    for idx, label in enumerate(labels, start=1):
        note_literal = _safe_literal(label, max_len=64)
        lines.extend(
            [
                f"        note_{idx} = Text('{note_literal}', font_size=24).scale_to_fit_width(12).to_edge(DOWN)",
                f"        self.play(FadeIn(note_{idx}), run_time=0.5)",
                "        self.wait(0.3)",
                f"        self.play(FadeOut(note_{idx}), run_time=0.5)",
            ]
        )

    lines.extend(
        [
            "        self.wait(0.6)",
        ]
    )
    return "\n".join(lines)


def _ensure_contract(code: str, task: str, storyboard_text: str, storyboard_items: list[dict[str, Any]], retry_count: int) -> str:
    """Enforce import + GeneratedScene contract."""

    fixed = _ensure_required_import(code)
    if _has_generated_scene(fixed):
        return fixed
    return _fallback_code(task, storyboard_text, storyboard_items, retry_count)


def _syntax_ok(code: str) -> bool:
    """Check Python syntax validity."""

    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _shape_present_in_code(code: str, shape_name: str) -> bool:
    """Check shape constructor usage in generated code."""

    return bool(re.search(rf"\b{re.escape(shape_name)}\s*\(", code))


def _semantic_alignment_ok(
    code: str,
    task: str,
    storyboard_text: str,
    storyboard_items: list[dict[str, Any]],
) -> bool:
    """Verify generated code matches requested semantics well enough."""

    source = f"{task}\n{storyboard_text}"
    source_lower = source.lower()

    requested_3d = _wants_3d(source)
    if not requested_3d:
        if any(token in code for token in ("Cube(", "Sphere(", "Cylinder(", "Cone(", "ThreeDScene")):
            return False

    shapes = _detected_shapes(source)
    if shapes:
        present = [shape for shape in shapes if _shape_present_in_code(code, shape)]
        needs_transform = _has_transform_intent(source) and len(shapes) >= 2
        min_required = 2 if needs_transform else 1
        if len(present) < min_required:
            return False

    if _has_transform_intent(source_lower) and len(shapes) >= 2:
        if not any(token in code for token in ("Transform(", "ReplacementTransform(", "FadeTransform(")):
            return False

    if len(storyboard_items) >= 2:
        play_count = len(re.findall(r"\bself\.play\s*\(", code))
        if play_count < 2:
            return False

    return True


def coder_node(state: AgentState) -> dict[str, Any]:
    """
    Coder agent:
    - If storyboard is JSON text, generate code from storyboard.
    - Else generate directly from natural language task.
    - Enforce semantic alignment with fallback safeguard.
    """

    task = str(state.get("task", "")).strip()
    storyboard_text = state.get("storyboard")
    is_storyboard_json, normalized_storyboard, storyboard_items = _parse_storyboard_text(storyboard_text)
    feedback_block = _build_feedback_block(state)

    try:
        retry_count = int(state.get("retry_count", 0))
    except Exception:
        retry_count = 0
    next_retry = retry_count + 1

    input_mode = "storyboard_json" if is_storyboard_json else "natural_language"
    storyboard_block = normalized_storyboard if normalized_storyboard else "N/A"
    strategy = _build_retry_strategy(next_retry)

    fallback = _fallback_code(task, storyboard_block, storyboard_items, next_retry)
    previous_failed = bool(state.get("ast_error") or state.get("render_error") or state.get("vision_error"))
    force_fallback = next_retry >= MAX_RETRIES and previous_failed

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CODER_SYSTEM_PROMPT),
            (
                "user",
                "Current attempt: {attempt}/{max_attempt}\n"
                "{strategy}\n\n"
                "Input mode: {input_mode}\n\n"
                "Task:\n{task}\n\n"
                "Storyboard (JSON text or raw text):\n{storyboard_block}\n\n"
                "Repair feedback:\n{feedback_block}\n\n"
                "Implementation requirements:\n"
                "- Preserve storyboard semantics in order.\n"
                "- Keep scene count compact and reliable for rendering.\n"
                "- Use only stable 2D primitives unless explicitly requested.\n"
                "- Ensure at least one clearly visible object in every key step.\n\n"
                "Return one complete executable Manim script now.",
            ),
        ]
    )

    hardened_code = fallback
    if not force_fallback:
        try:
            llm = _build_coder_llm()
            response = (prompt | llm).invoke(
                {
                    "attempt": next_retry,
                    "max_attempt": MAX_RETRIES,
                    "strategy": strategy,
                    "input_mode": input_mode,
                    "task": task,
                    "storyboard_block": storyboard_block,
                    "feedback_block": feedback_block,
                }
            )
            raw_code = _extract_text_content(response.content)
            cleaned_code = _strip_markdown_fence(raw_code)
            hardened_code = _ensure_contract(
                cleaned_code,
                task,
                storyboard_block,
                storyboard_items,
                next_retry,
            )
            if not _syntax_ok(hardened_code):
                hardened_code = fallback
            elif not _semantic_alignment_ok(hardened_code, task, storyboard_block, storyboard_items):
                hardened_code = fallback
        except Exception as exc:
            LOGGER.exception("coder_node failed: %s", exc)
            hardened_code = fallback

    return {
        "code": hardened_code,
        "ast_error": None,
        "render_error": None,
        "vision_error": None,
        "retry_count": next_retry,
    }
