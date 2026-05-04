from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from state import AgentState, MAX_RETRIES
from utils.model_provider import (
    build_deepseek_chat_model,
    get_coder_timeout_seconds,
    invoke_with_hard_timeout,
)

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
9) Use only Manim Community APIs. Do NOT use nonstandard classes such as TrailEffect.
10) For red-dot trails, prefer TracedPath, MoveAlongPath, ParametricFunction, or a VGroup of fading Dot copies.
11) Never use submodule imports like `from manim.animations.xxx import yyy` or
    `from manim.mobject.xxx import yyy`. The wildcard `from manim import *` provides
    all standard Manim CE classes.
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

RED_DOT_TOKENS: tuple[str, ...] = (
    "red dot",
    "dot",
    "trail",
    "trace",
    "tracedpath",
    "path",
    "fade",
    "dissipat",
    "\u7ea2\u70b9",
    "\u8f68\u8ff9",
    "\u8def\u5f84",
    "\u6d88\u6563",
)

TRANSFORM_LABEL_TOKENS: tuple[str, ...] = (
    "counterclockwisetransform",
    "counterclockwise transform",
    "transform",
)


def _build_coder_llm() -> ChatOpenAI:
    """Create coder model client using ChatOpenAI-compatible DeepSeek route."""

    return build_deepseek_chat_model(
        model_env_name="MANIM_CODER_MODEL",
        default_model="deepseek-v4-pro",
        temperature_env_name="MANIM_CODER_TEMPERATURE",
        default_temperature=0.15,
        timeout_seconds=get_coder_timeout_seconds(),
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


def _strip_manim_submodule_imports(code: str) -> str:
    """Remove invalid from manim.xxx import yyy submodule imports.

    from manim import * is the only valid Manim import form.
    Submodule paths like from manim.animations.transform do not resolve.
    """
    return re.sub(r"^from\s+manim\.\S+\s+import\s+.*$", "", code, flags=re.MULTILINE)


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
        return _sanitize_nonstandard_apis(text)
    except Exception:
        return _sanitize_nonstandard_apis(text)

    if not isinstance(parsed, dict):
        return _sanitize_nonstandard_apis(text)

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

    return _sanitize_nonstandard_apis("\n".join(lines))


_NONSTANDARD_API_REPLACEMENTS: list[tuple[str, str]] = [
    ("TrailEffect", "TracedPath"),
    ("TransformWithRotation", "Rotate"),
    ("PathTracer", "TracedPath"),
    ("FadingTrail", "TracedPath"),
]


def _sanitize_nonstandard_apis(text: str) -> str:
    """Replace known non-standard Manim API names with standard equivalents."""
    for bad_name, good_name in _NONSTANDARD_API_REPLACEMENTS:
        text = re.sub(rf"\b{re.escape(bad_name)}\b", good_name, text)
    return text


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
        render_text = str(render_error)
        if "TrailEffect" in render_text:
            render_text = (
                f"{render_text}\n"
                "Do not use TrailEffect; it is not a standard Manim CE API. "
                "Use TracedPath or a VGroup of fading Dot copies instead."
            )
        feedbacks.append(f"[render_error]\n{render_text}")
    if vision_error and vision_error.strip().upper() != "OK":
        if "TrailEffect" in vision_error:
            vision_error = (
                f"{vision_error}\n"
                "Note: TrailEffect is not a standard Manim CE API. "
                "Use TracedPath or a VGroup of fading Dot copies instead."
            )
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


def _extract_decimal_tokens(text: str, limit: int = 6) -> list[str]:
    """Extract visible numeric tokens from task/storyboard text."""

    ordered: list[str] = []
    for match in re.finditer(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])", text):
        token = match.group(0)
        if token not in ordered:
            ordered.append(token)
        if len(ordered) >= limit:
            break
    return ordered


def _wants_red_dot_trail(text: str) -> bool:
    """Detect red-dot path/trail animation tasks."""

    lower_text = text.lower()
    has_dot = "dot" in lower_text or "\u70b9" in lower_text
    has_red = "red" in lower_text or "\u7ea2" in lower_text
    has_trail = any(token in lower_text for token in RED_DOT_TOKENS)
    return (has_red and has_dot) or (has_dot and has_trail)


def _wants_transform_number_scene(text: str) -> bool:
    """Detect numeric transform comparison tasks."""

    lower_text = text.lower()
    numbers = _extract_decimal_tokens(text, limit=4)
    return len(numbers) >= 2 and any(token in lower_text for token in TRANSFORM_LABEL_TOKENS)


def _task_keywords(text: str, limit: int = 6) -> list[str]:
    """Extract short semantic keywords for generic fallback labels."""

    quoted = re.findall(r"[\"'`](.*?)[\"'`]", text)
    candidates: list[str] = []
    candidates.extend(item.strip() for item in quoted if item.strip())
    candidates.extend(_extract_decimal_tokens(text, limit=limit))

    for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_]{2,}\b", text):
        lower_word = word.lower()
        if lower_word in {"the", "and", "with", "for", "from", "scene", "using", "task"}:
            continue
        candidates.append(word)

    ordered: list[str] = []
    for item in candidates:
        compact = re.sub(r"\s+", " ", item).strip()
        if not compact or compact in ordered:
            continue
        ordered.append(compact[:32])
        if len(ordered) >= limit:
            break
    return ordered


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


def _fallback_number_transform_code(source: str) -> str:
    """Build deterministic fallback for numeric Transform comparison tasks."""

    numbers = _extract_decimal_tokens(source, limit=4)
    first = _safe_literal(numbers[0] if numbers else "3.141", max_len=16)
    second = _safe_literal(numbers[1] if len(numbers) > 1 else "1.618", max_len=16)
    left_label = "CounterclockwiseTransform"
    right_label = "Transform"

    lines: list[str] = [
        "from manim import *",
        "",
        "class GeneratedScene(Scene):",
        "    def construct(self):",
        f"        left_label = Text('{left_label}', color=RED, font_size=24)",
        f"        left_top = Text('{first}', font_size=34)",
        f"        left_bottom = Text('{second}', font_size=34)",
        "        left_group = VGroup(left_top, left_bottom, left_label).arrange(DOWN, buff=0.35).shift(LEFT * 3.1)",
        f"        right_label = Text('{right_label}', color=BLUE, font_size=28)",
        f"        right_top = Text('{second}', font_size=34)",
        f"        right_bottom = Text('{first}', font_size=34)",
        "        right_group = VGroup(right_top, right_bottom, right_label).arrange(DOWN, buff=0.35).shift(RIGHT * 3.1)",
        "        all_groups = VGroup(left_group, right_group).scale_to_fit_width(12).center()",
        "        self.play(FadeIn(all_groups), run_time=0.8)",
        "        left_arrow = CurvedArrow(left_top.get_bottom(), left_bottom.get_top(), angle=PI / 2, color=YELLOW)",
        "        right_arrow = Arrow(right_top.get_bottom(), right_bottom.get_top(), buff=0.12, color=YELLOW)",
        "        self.play(Create(left_arrow), Create(right_arrow), run_time=0.5)",
        "        moving_left = left_top.copy().set_color(YELLOW).set_z_index(3)",
        "        moving_right = right_top.copy().set_color(YELLOW).set_z_index(3)",
        "        self.add(moving_left, moving_right)",
        "        self.play(",
        "            CounterclockwiseTransform(moving_left, left_bottom.copy().set_color(YELLOW)),",
            "            Transform(moving_right, right_bottom.copy().set_color(YELLOW)),",
            "            run_time=1.2,",
        "        )",
        "        self.play(",
        "            left_bottom.animate.set_color(YELLOW),",
        "            right_bottom.animate.set_color(YELLOW),",
        "            run_time=0.3,",
        "        )",
        "        self.play(FadeOut(moving_left), FadeOut(moving_right), FadeOut(left_arrow), FadeOut(right_arrow), run_time=0.4)",
        "        self.wait(1.0)",
    ]
    return "\n".join(lines)


def _fallback_red_dot_trail_code(source: str) -> str:
    """Build deterministic fallback for red-dot trail/path tasks."""

    title = _safe_literal(source, max_len=44)
    lines: list[str] = [
        "from manim import *",
        "",
        "class GeneratedScene(Scene):",
        "    def construct(self):",
        f"        title = Text('{title}', font_size=26).scale_to_fit_width(11).to_edge(UP)",
        "        start_point = RIGHT * 5 + DOWN * 0.5",
        "        middle_point = ORIGIN + UP * 1.45",
        "        end_point = LEFT * 5 + DOWN * 0.5",
        "        arc_one = ArcBetweenPoints(start_point, middle_point, angle=PI / 2, color=RED, stroke_width=3)",
        "        arc_two = ArcBetweenPoints(middle_point, end_point, angle=-PI / 2, color=RED, stroke_width=3)",
        "        dot = Dot(color=RED, radius=0.14).move_to(start_point)",
        "        trail = TracedPath(",
        "            dot.get_center,",
        "            stroke_color=RED,",
        "            stroke_width=5,",
        "            dissipating_time=0.5,",
        "        )",
        "        self.add(trail, dot)",
        "        self.play(FadeIn(title), run_time=0.5)",
        "        self.play(Create(arc_one), MoveAlongPath(dot, arc_one), run_time=1.3)",
        "        self.play(Create(arc_two), MoveAlongPath(dot, arc_two), run_time=1.3)",
        "        self.wait(0.6)",
    ]
    return "\n".join(lines)


def _fallback_shape_code(
    task: str,
    source: str,
    storyboard_items: list[dict[str, Any]],
    retry_count: int,
) -> str:
    """Build deterministic fallback for explicit shape tasks."""

    shapes = _detected_shapes(source)
    has_transform = _has_transform_intent(source)
    primary = shapes[0] if shapes else "Circle"
    secondary = shapes[1] if len(shapes) > 1 else primary
    concise = _safe_literal(task, max_len=44)
    labels = _storyboard_scene_labels(storyboard_items, limit=2 if retry_count < 4 else 1)

    lines: list[str] = [
        "from manim import *",
        "",
        "class GeneratedScene(Scene):",
        "    def construct(self):",
        f"        title = Text('{concise}', font_size=30).scale_to_fit_width(11).to_edge(UP)",
        "        self.play(FadeIn(title), run_time=0.8)",
        f"        shape_a = {_shape_ctor(primary)}.scale(1.1).shift(DOWN * 0.2)",
        "        self.play(Create(shape_a), run_time=1.2)",
    ]

    if has_transform and primary != secondary:
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


def _fallback_keyword_board_code(
    task: str,
    source: str,
    storyboard_items: list[dict[str, Any]],
    *,
    storyboard_required: bool,
) -> str:
    """Build a generic semantic board instead of unrelated geometry."""

    title = _safe_literal(task, max_len=48)
    keywords = _task_keywords(source, limit=6)
    if not keywords and storyboard_items:
        keywords = _storyboard_scene_labels(storyboard_items, limit=4)
    if not keywords:
        keywords = ["Key visual object", "Main relation", "Final state"]

    lines: list[str] = [
        "from manim import *",
        "",
        "class GeneratedScene(Scene):",
        "    def construct(self):",
        f"        title = Text('{title}', font_size=30).scale_to_fit_width(11).to_edge(UP)",
        "        cards = VGroup()",
    ]
    for idx, keyword in enumerate(keywords[:6], start=1):
        literal = _safe_literal(keyword, max_len=36)
        lines.extend(
            [
                f"        card_{idx} = Text('{literal}', font_size=26).scale_to_fit_width(5.4)",
                f"        box_{idx} = SurroundingRectangle(card_{idx}, color=BLUE, buff=0.18)",
                f"        group_{idx} = VGroup(box_{idx}, card_{idx})",
                f"        cards.add(group_{idx})",
            ]
        )
    lines.extend(
        [
            "        cards.arrange_in_grid(rows=2, buff=0.45).scale_to_fit_width(11).next_to(title, DOWN, buff=0.6)",
            "        self.play(FadeIn(title), run_time=0.6)",
            "        self.play(LaggedStart(*[FadeIn(card) for card in cards], lag_ratio=0.12), run_time=1.4)",
            "        self.wait(1.0)",
        ]
    )

    if storyboard_required and storyboard_items:
        labels = _storyboard_scene_labels(storyboard_items, limit=min(4, len(storyboard_items)))
        for idx, label in enumerate(labels, start=1):
            literal = _safe_literal(label, max_len=52)
            lines.extend(
                [
                    f"        scene_note_{idx} = Text('{literal}', font_size=22).scale_to_fit_width(11.5).to_edge(DOWN)",
                    f"        self.play(FadeIn(scene_note_{idx}), run_time=0.3)",
                    "        self.wait(0.2)",
                    f"        self.play(FadeOut(scene_note_{idx}), run_time=0.3)",
                ]
            )

    return "\n".join(lines)


def _fallback_code(
    task: str,
    storyboard_text: str,
    storyboard_items: list[dict[str, Any]],
    retry_count: int,
    *,
    storyboard_required: bool = False,
) -> str:
    """Build deterministic fallback code while preserving task objects."""

    source = f"{task}\n{storyboard_text}"
    if _wants_transform_number_scene(source):
        return _fallback_number_transform_code(source)
    if _wants_red_dot_trail(source):
        return _fallback_red_dot_trail_code(source)
    if _detected_shapes(source):
        return _fallback_shape_code(task, source, storyboard_items, retry_count)
    return _fallback_keyword_board_code(
        task,
        source,
        storyboard_items,
        storyboard_required=storyboard_required,
    )


def _ensure_contract(
    code: str,
    task: str,
    storyboard_text: str,
    storyboard_items: list[dict[str, Any]],
    retry_count: int,
    *,
    storyboard_required: bool = False,
) -> str:
    """Enforce import + GeneratedScene contract."""

    fixed = _ensure_required_import(code)
    fixed = _strip_manim_submodule_imports(fixed)
    if _has_generated_scene(fixed):
        return fixed
    return _fallback_code(
        task,
        storyboard_text,
        storyboard_items,
        retry_count,
        storyboard_required=storyboard_required,
    )


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

    if _wants_transform_number_scene(source):
        numbers = _extract_decimal_tokens(source, limit=2)
        has_numbers = all(number in code for number in numbers)
        has_transform_label = "Transform" in code
        has_transform_motion = any(token in code for token in ("Transform(", "ReplacementTransform(", "Rotate("))
        if not (has_numbers and has_transform_label and has_transform_motion):
            return False

    if _wants_red_dot_trail(source):
        has_red_dot = "Dot(" in code and "RED" in code
        has_motion_or_trail = any(
            token in code
            for token in ("TracedPath", "MoveAlongPath", "ParametricFunction", "trail", "path")
        )
        if not (has_red_dot and has_motion_or_trail):
            return False

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
    strategy_name = str(state.get("strategy", "")).strip()
    storyboard_required = strategy_name == "Ours"
    storyboard_text = state.get("storyboard")
    is_storyboard_json, normalized_storyboard, storyboard_items = _parse_storyboard_text(storyboard_text)
    feedback_block = _build_feedback_block(state)

    if storyboard_required and not is_storyboard_json:
        return {
            "code": "",
            "ast_error": "Ours strategy requires valid storyboard JSON before coder execution.",
            "render_error": None,
            "vision_error": None,
            "retry_count": int(state.get("retry_count", 0) or 0),
            "coder_input_mode": "natural_language",
            "coder_storyboard_used": False,
            "storyboard_present": False,
            "retry_count_reason": "coder attempt count",
        }

    try:
        retry_count = int(state.get("retry_count", 0))
    except Exception:
        retry_count = 0
    next_retry = retry_count + 1

    input_mode = "storyboard_json" if is_storyboard_json else "natural_language"
    storyboard_block = normalized_storyboard if normalized_storyboard else "N/A"
    strategy = _build_retry_strategy(next_retry)

    fallback = _fallback_code(
        task,
        storyboard_block,
        storyboard_items,
        next_retry,
        storyboard_required=storyboard_required,
    )
    previous_code = str(state.get("code", "")).strip()
    had_ast_failure = bool(state.get("ast_error"))
    had_render_failure = bool(state.get("render_error"))
    had_vision_failure = bool(state.get("vision_error"))
    vision_only_retry = had_vision_failure and not (had_ast_failure or had_render_failure)
    if vision_only_retry and previous_code and _syntax_ok(previous_code):
        if not storyboard_required or _semantic_alignment_ok(previous_code, task, storyboard_block, storyboard_items):
            backup_code = previous_code
        else:
            backup_code = fallback
    else:
        backup_code = fallback
    force_fallback = next_retry >= MAX_RETRIES and (had_ast_failure or had_render_failure)

    storyboard_guidance = ""
    if is_storyboard_json:
        storyboard_guidance = (
            "- Storyboard is present: stay conservative with simple 2D primitives.\n"
            "- Map each storyboard scene to 1-2 self.play() calls maximum.\n"
            "- Use only objects explicitly named in the storyboard.\n"
        )

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
                "{storyboard_guidance}"
                "- Preserve storyboard semantics in order.\n"
                "- Keep scene count compact and reliable for rendering.\n"
                "- Use only stable 2D primitives unless explicitly requested.\n"
                "- Ensure at least one clearly visible object in every key step.\n\n"
                "Return one complete executable Manim script now.",
            ),
        ]
    )

    hardened_code = backup_code
    resolved_retry_count = next_retry
    if not force_fallback:
        try:
            LOGGER.info(
                "Coder started for strategy=%s attempt=%s/%s",
                state.get("strategy", "unknown"),
                next_retry,
                MAX_RETRIES,
            )
            llm = _build_coder_llm()
            response = invoke_with_hard_timeout(
                lambda: (prompt | llm).invoke(
                    {
                        "attempt": next_retry,
                        "max_attempt": MAX_RETRIES,
                        "strategy": strategy,
                        "input_mode": input_mode,
                        "task": task,
                        "storyboard_block": storyboard_block,
                        "feedback_block": feedback_block,
                        "storyboard_guidance": storyboard_guidance,
                    }
                ),
                timeout_seconds=get_coder_timeout_seconds() + 5,
                timeout_label="coder model call",
            )
            raw_code = _extract_text_content(response.content)
            cleaned_code = _strip_markdown_fence(raw_code)
            hardened_code = _ensure_contract(
                cleaned_code,
                task,
                storyboard_block,
                storyboard_items,
                next_retry,
                storyboard_required=storyboard_required,
            )
            if not _syntax_ok(hardened_code):
                hardened_code = backup_code if _syntax_ok(backup_code) else fallback
            elif not _semantic_alignment_ok(hardened_code, task, storyboard_block, storyboard_items):
                if _semantic_alignment_ok(backup_code, task, storyboard_block, storyboard_items):
                    hardened_code = backup_code
                else:
                    hardened_code = fallback
        except Exception as exc:
            LOGGER.exception("coder_node failed: %s", exc)
            hardened_code = backup_code
            if isinstance(exc, TimeoutError) and vision_only_retry and previous_code and hardened_code == previous_code:
                LOGGER.warning(
                    "Coder timed out while repairing prior vision feedback; finishing after this fallback replay."
                )
                resolved_retry_count = MAX_RETRIES

    return {
        "code": hardened_code,
        "ast_error": None,
        "render_error": None,
        "vision_error": None,
        "retry_count": resolved_retry_count,
        "render_media_dir": None,
        "render_image_path": None,
        "render_video_path": None,
        "coder_input_mode": input_mode,
        "coder_storyboard_used": bool(storyboard_required and is_storyboard_json),
        "storyboard_present": bool(is_storyboard_json),
        "retry_count_reason": "coder attempt count",
    }
