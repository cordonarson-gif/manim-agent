from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from state import AgentState
from utils.manim_injector import inject_bounding_boxes

TEMP_FILE = Path("temp.py")
SCENE_CLASS = "GeneratedScene"
RUN_MEDIA_ROOT = Path("media/runs")
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_ERROR_LIMIT = 1800


def _safe_int_env(name: str, default: int) -> int:
    """Read integer env var with fallback."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _truncate_error(message: str, limit: int) -> str:
    """Truncate long stderr text."""

    return message if len(message) <= limit else f"{message[:limit]} ...<truncated>"


def _new_run_media_dir(retry_count: int) -> Path:
    """Create per-attempt media directory."""

    run_id = f"run_{int(time.time() * 1000)}_r{retry_count}"
    run_dir = RUN_MEDIA_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _find_latest_file(root: Path, suffix: str, *, exclude_parts: tuple[str, ...] = ()) -> Path | None:
    """Find latest file under root by suffix."""

    if not root.exists():
        return None
    candidates: list[Path] = []
    try:
        for path in root.rglob(f"*{suffix}"):
            if not path.is_file():
                continue
            if any(part in path.parts for part in exclude_parts):
                continue
            candidates.append(path)
    except Exception:
        return None

    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _run_manim(
    command: list[str],
    *,
    timeout_seconds: int,
    error_limit: int,
    stage: str,
) -> str | None:
    """Run manim command and return error text when failed."""

    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return "Manim command not found. Please ensure `manim` is in PATH."
    except subprocess.TimeoutExpired as exc:
        msg = (exc.stderr or exc.stdout or "").strip() or f"{stage} timed out."
        return _truncate_error(msg, error_limit)
    except Exception as exc:
        return _truncate_error(f"{stage} execution failure: {exc}", error_limit)

    if result.returncode == 0:
        return None

    stderr_text = (result.stderr or "").strip()
    if not stderr_text:
        stderr_text = (result.stdout or "").strip()
    if not stderr_text:
        stderr_text = f"Manim exited with code {result.returncode}."
    return _truncate_error(f"[{stage}] {stderr_text}", error_limit)


def execution_node(state: AgentState) -> dict[str, Any]:
    """
    Sandbox execution node.

    1) Inject visual anchors.
    2) Write temp file.
    3) Snapshot render for vision review.
    4) Video render for final playback.
    5) Return current-run artifact paths to avoid stale-file selection.
    """

    try:
        code = str(state.get("code", ""))
    except Exception:
        code = ""
    if not code.strip():
        return {"render_error": "No source code available for execution."}

    try:
        injected_code = inject_bounding_boxes(code)
    except Exception as exc:
        return {"render_error": f"Failed to inject bounding boxes: {exc}"}

    try:
        TEMP_FILE.write_text(injected_code, encoding="utf-8")
    except OSError as exc:
        return {"render_error": f"Failed to write temp.py: {exc}"}
    except Exception as exc:
        return {"render_error": f"Unexpected file write failure: {exc}"}

    timeout_seconds = _safe_int_env("MANIM_EXEC_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
    error_limit = _safe_int_env("MANIM_ERROR_LIMIT", DEFAULT_ERROR_LIMIT)
    retry_count = int(state.get("retry_count", 0)) if str(state.get("retry_count", "")).isdigit() else 0
    run_media_dir = _new_run_media_dir(retry_count)

    snapshot_cmd = [
        "manim",
        "-ql",
        "-s",
        "--disable_caching",
        "--media_dir",
        str(run_media_dir),
        str(TEMP_FILE),
        SCENE_CLASS,
    ]
    snapshot_err = _run_manim(
        snapshot_cmd,
        timeout_seconds=timeout_seconds,
        error_limit=error_limit,
        stage="snapshot",
    )
    if snapshot_err is not None:
        return {
            "render_error": snapshot_err,
            "render_media_dir": str(run_media_dir),
            "render_image_path": None,
            "render_video_path": None,
        }

    video_cmd = [
        "manim",
        "-ql",
        "--disable_caching",
        "--media_dir",
        str(run_media_dir),
        str(TEMP_FILE),
        SCENE_CLASS,
    ]
    video_err = _run_manim(
        video_cmd,
        timeout_seconds=timeout_seconds,
        error_limit=error_limit,
        stage="video",
    )
    if video_err is not None:
        image_path = _find_latest_file(run_media_dir, ".png")
        return {
            "render_error": video_err,
            "render_media_dir": str(run_media_dir),
            "render_image_path": str(image_path) if image_path else None,
            "render_video_path": None,
        }

    image_path = _find_latest_file(run_media_dir, ".png")
    video_path = _find_latest_file(run_media_dir, ".mp4", exclude_parts=("partial_movie_files",))
    return {
        "render_error": None,
        "render_media_dir": str(run_media_dir),
        "render_image_path": str(image_path) if image_path else None,
        "render_video_path": str(video_path) if video_path else None,
    }
