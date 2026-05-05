from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from state import AgentState, create_initial_state

try:
    from workflow import generate_app, plan_only_app
except Exception as exc:  # pragma: no cover - runtime dependency guard.
    generate_app = None
    plan_only_app = None
    _WORKFLOW_IMPORT_ERROR = exc
else:
    _WORKFLOW_IMPORT_ERROR = None


def _merge_state(runtime_state: AgentState, patch: dict[str, Any]) -> None:
    """In-place merge for incremental node updates from LangGraph stream."""

    for key, value in patch.items():
        runtime_state[key] = value


def _latest_artifact() -> Path | None:
    """Return latest render artifact path (prefer mp4, fallback png)."""

    video_root = Path("media/videos")
    if video_root.exists():
        videos: list[Path] = [
            p
            for p in video_root.rglob("*.mp4")
            if p.is_file() and "partial_movie_files" not in p.parts
        ]
        if videos:
            videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return videos[0]

    image_root = Path("media/images")
    if image_root.exists():
        images: list[Path] = [p for p in image_root.rglob("*.png") if p.is_file()]
        if images:
            images.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return images[0]

    return None


def _short(text: str, limit: int = 600) -> str:
    """Limit console payload length for readability."""

    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return f"{stripped[:limit]} ...<truncated>"


def _print_node_update(node_name: str, update: dict[str, Any]) -> None:
    """Print concise event trace for each agent node."""

    print(f"\n[{node_name}]")

    if node_name == "planner":
        storyboard = update.get("storyboard")
        print(_short(json.dumps(storyboard, ensure_ascii=False, indent=2), 1000))
        return

    if node_name == "coder":
        code: str = str(update.get("code", ""))
        print(_short(code, 1000))
        return

    if node_name == "ast_reviewer":
        print(f"ast_error: {update.get('ast_error')}")
        return

    if node_name == "execution":
        print(f"render_error: {update.get('render_error')}")
        return

    if node_name == "vision_critic":
        print(f"vision_error: {update.get('vision_error')}")
        return

    print(_short(str(update)))


def run_workflow(task: str, quiet: bool = False) -> AgentState:
    """Execute planning + generation workflow and return the final state snapshot."""

    if generate_app is None or plan_only_app is None:
        raise RuntimeError(f"Failed to import workflow: {_WORKFLOW_IMPORT_ERROR}")

    state: AgentState = create_initial_state(task)

    storyboard = state.get("storyboard")
    if not isinstance(storyboard, str) or not storyboard.strip():
        for event in plan_only_app.stream(state):
            for node_name, node_update in event.items():
                if isinstance(node_update, dict):
                    _merge_state(state, node_update)
                if not quiet and isinstance(node_update, dict):
                    _print_node_update(node_name, node_update)

    for event in generate_app.stream(state):
        for node_name, node_update in event.items():
            if isinstance(node_update, dict):
                _merge_state(state, node_update)
            if not quiet and isinstance(node_update, dict):
                _print_node_update(node_name, node_update)

    return state


def _parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run the Manim dual-feedback LangGraph pipeline."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="",
        help="Natural language request for animation generation.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-node logs and print only final summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    args = _parse_args(argv or sys.argv[1:])
    task: str = args.task.strip()

    if not task:
        task = input("Please enter your Manim task: ").strip()

    if not task:
        print("Task is empty. Abort.")
        return 2

    try:
        final_state: AgentState = run_workflow(task=task, quiet=args.quiet)
    except Exception as exc:
        print(f"Workflow execution failed: {exc}")
        return 1

    artifact = _latest_artifact()
    print("\n=== Final State Summary ===")
    print(f"retry_count: {final_state.get('retry_count')}")
    print(f"retry_count_reason: {final_state.get('retry_count_reason')}")
    print(f"final_verdict: {final_state.get('final_verdict')}")
    print(f"failure_stage: {final_state.get('failure_stage')}")
    print(f"failure_type: {final_state.get('failure_type')}")
    print(f"success_reason: {final_state.get('success_reason')}")
    print(f"failure_reason: {final_state.get('failure_reason')}")
    print(f"execution_environment: {final_state.get('execution_environment')}")
    print(f"planner_used: {final_state.get('planner_used')}")
    print(f"storyboard_present: {final_state.get('storyboard_present')}")
    print(f"coder_input_mode: {final_state.get('coder_input_mode')}")
    print(f"coder_storyboard_used: {final_state.get('coder_storyboard_used')}")
    print(f"ast_error: {final_state.get('ast_error')}")
    print(f"render_error: {final_state.get('render_error')}")
    print(f"vision_error: {final_state.get('vision_error')}")
    if artifact is not None:
        print(f"artifact: {artifact}")
    else:
        print("artifact: not found")

    has_error: bool = bool(
        final_state.get("ast_error")
        or final_state.get("render_error")
        or final_state.get("vision_error")
    )
    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
