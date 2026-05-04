from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from state import AgentState
from utils.manim_injector import inject_bounding_boxes

LOGGER = logging.getLogger(__name__)

GENERATED_FILE_NAME = "generated_scene.py"
SCENE_CLASS = "GeneratedScene"
RUN_MEDIA_ROOT = Path("media/runs")
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_ERROR_LIMIT = 1800
DEFAULT_SANDBOX_MODE = "docker"
DEFAULT_DOCKER_IMAGE = "manimcommunity/manim:stable"
DEFAULT_DOCKER_CPUS = "1.0"
DEFAULT_DOCKER_MEMORY = "1g"
DEFAULT_DOCKER_PIDS_LIMIT = "256"
LOCAL_SANDBOX_ACK = "MANIM_ALLOW_UNSANDBOXED"


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


def _safe_text_env(name: str, default: str) -> str:
    """Read non-empty text env var with fallback."""

    raw = os.getenv(name, "").strip()
    return raw or default


def _truncate_error(message: str, limit: int) -> str:
    """Truncate long stderr text."""

    return message if len(message) <= limit else f"{message[:limit]} ...<truncated>"


def _safe_process_text(value: str | bytes | None) -> str:
    """Normalize subprocess output without trusting the OS default encoding."""

    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _new_run_media_dir(retry_count: int) -> Path:
    """Create per-attempt media directory."""

    run_id = f"run_{int(time.time() * 1000)}_r{retry_count}"
    run_dir = RUN_MEDIA_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _find_latest_file(
    root: Path,
    suffix: str,
    *,
    exclude_parts: tuple[str, ...] = (),
) -> Path | None:
    """Return the newest matching artifact under one run directory."""

    candidates: list[Path] = []
    try:
        for path in root.rglob(f"*{suffix}"):
            if not path.is_file():
                continue
            if exclude_parts and any(part in path.parts for part in exclude_parts):
                continue
            candidates.append(path)
    except Exception:
        return None

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _sandbox_mode() -> str:
    """Resolve execution sandbox mode."""

    mode = _safe_text_env("MANIM_SANDBOX_MODE", DEFAULT_SANDBOX_MODE).lower()
    if mode not in {"docker", "local"}:
        return DEFAULT_SANDBOX_MODE
    return mode


def _safe_local_env() -> dict[str, str]:
    """Build a minimal env for explicit local fallback without API secrets."""

    allowed_names = {
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "HOME",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
        "PROGRAMDATA",
    }
    return {key: value for key, value in os.environ.items() if key.upper() in allowed_names}


def _execution_environment(mode: str) -> str:
    """Return human-readable execution environment."""

    return "docker" if mode == "docker" else "local"


def _docker_context() -> str | None:
    """Return execution context string for experiment reporting."""

    if os.getenv("RUNNING_IN_DOCKER", "").strip() == "1":
        return "docker-container"
    return os.getenv("MANIM_SANDBOX_IMAGE", "").strip() or None


def _execution_failure_patch(
    *,
    message: str,
    stage: str = "execution",
    failure_type: str = "infra",
    run_media_dir: Path | None = None,
    image_path: Path | None = None,
    video_path: Path | None = None,
) -> dict[str, Any]:
    """Build structured execution failure payload."""

    return {
        "render_error": message,
        "failure_stage": stage,
        "failure_type": failure_type,
        "failure_reason": message,
        "execution_environment": _execution_environment(_sandbox_mode()),
        "docker_context": _docker_context(),
        "render_media_dir": str(run_media_dir) if run_media_dir else None,
        "render_image_path": str(image_path) if image_path else None,
        "render_video_path": str(video_path) if video_path else None,
    }


def _looks_like_infra_failure(message: str) -> bool:
    """Detect environment/tooling failures that coder retries cannot fix."""

    lowered = message.strip().lower()
    infra_tokens = (
        "docker command not found",
        "manim command not found",
        "timed out",
        "execution failure",
        "failed to write generated code",
        "local generated-code execution is disabled",
        "failed to build sandboxed manim command",
    )
    return any(token in lowered for token in infra_tokens)


def _run_manim(
    command: list[str],
    *,
    timeout_seconds: int,
    error_limit: int,
    stage: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str | None:
    """Run manim command and return error text when failed."""

    try:
        result: subprocess.CompletedProcess[str] = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
        )
    except FileNotFoundError:
        executable = command[0] if command else "command"
        if executable == "docker":
            return (
                "Docker command not found. Secure generated-code execution requires Docker "
                "with a local Manim image. Set MANIM_SANDBOX_MODE=local and "
                f"{LOCAL_SANDBOX_ACK}=1 only for trusted development runs."
            )
        return "Manim command not found. Please ensure `manim` is in PATH."
    except subprocess.TimeoutExpired as exc:
        msg = (_safe_process_text(exc.stderr) or _safe_process_text(exc.stdout)).strip()
        if not msg:
            msg = f"{stage} timed out."
        return _truncate_error(msg, error_limit)
    except Exception as exc:
        return _truncate_error(f"{stage} execution failure: {exc}", error_limit)

    if result.returncode == 0:
        return None

    stderr_text = _safe_process_text(result.stderr).strip()
    if not stderr_text:
        stderr_text = _safe_process_text(result.stdout).strip()
    if not stderr_text:
        stderr_text = f"Manim exited with code {result.returncode}."
    return _truncate_error(f"[{stage}] {stderr_text}", error_limit)


def _docker_manim_command(
    run_dir: Path,
    *,
    still_image: bool,
) -> list[str]:
    """Build a dockerized Manim command with no network and constrained resources."""

    image = _safe_text_env("MANIM_SANDBOX_IMAGE", DEFAULT_DOCKER_IMAGE)
    cpus = _safe_text_env("MANIM_SANDBOX_CPUS", DEFAULT_DOCKER_CPUS)
    memory = _safe_text_env("MANIM_SANDBOX_MEMORY", DEFAULT_DOCKER_MEMORY)
    pids_limit = _safe_text_env("MANIM_SANDBOX_PIDS_LIMIT", DEFAULT_DOCKER_PIDS_LIMIT)
    host_workspace = str(run_dir.resolve())
    quality_args = ["-ql", "-s"] if still_image else ["-ql"]

    return [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--network",
        "none",
        "--cpus",
        cpus,
        "--memory",
        memory,
        "--pids-limit",
        pids_limit,
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=256m",
        "-v",
        f"{host_workspace}:/workspace:rw",
        "-w",
        "/workspace",
        "--env",
        "HOME=/workspace",
        "--env",
        "XDG_CACHE_HOME=/workspace/.cache",
        "--env",
        "MPLCONFIGDIR=/workspace/.cache/matplotlib",
        image,
        "manim",
        *quality_args,
        "--disable_caching",
        "--media_dir",
        "media",
        GENERATED_FILE_NAME,
        SCENE_CLASS,
    ]


def _local_manim_command(
    source_file: Path,
    media_dir: Path,
    *,
    still_image: bool,
) -> list[str]:
    """Build explicit opt-in local Manim command."""

    quality_args = ["-ql", "-s"] if still_image else ["-ql"]
    return [
        "manim",
        *quality_args,
        "--disable_caching",
        "--media_dir",
        str(media_dir),
        str(source_file),
        SCENE_CLASS,
    ]


def _build_manim_command(
    run_dir: Path,
    source_file: Path,
    media_dir: Path,
    *,
    still_image: bool,
) -> tuple[list[str] | None, Path | None, dict[str, str] | None, str | None]:
    """Return command/cwd/env or an error when sandbox settings are unsafe."""

    mode = _sandbox_mode()
    if mode == "docker":
        return _docker_manim_command(run_dir, still_image=still_image), None, None, None

    if os.getenv(LOCAL_SANDBOX_ACK, "").strip() != "1":
        return (
            None,
            None,
            None,
            "Local generated-code execution is disabled by default. Use Docker sandboxing, "
            f"or set MANIM_SANDBOX_MODE=local and {LOCAL_SANDBOX_ACK}=1 only for trusted code.",
        )

    return (
        _local_manim_command(source_file, media_dir, still_image=still_image),
        run_dir,
        _safe_local_env(),
        None,
    )


def execution_node(state: AgentState) -> dict[str, Any]:
    """
    Sandbox execution node.

    1) Inject visual anchors.
    2) Write generated code into a per-run workspace.
    3) Snapshot render for vision review.
    4) Video render for final playback.
    5) Return current-run artifact paths to avoid stale-file selection.
    """

    mode = _sandbox_mode()
    execution_environment = _execution_environment(mode)
    docker_context = _docker_context()

    try:
        code = str(state.get("code", ""))
    except Exception:
        code = ""
    if not code.strip():
        return _execution_failure_patch(
            message="No source code available for execution.",
            failure_type="content",
        )

    try:
        injected_code = inject_bounding_boxes(code)
    except Exception as exc:
        return _execution_failure_patch(
            message=f"Failed to inject bounding boxes: {exc}",
            failure_type="content",
        )

    timeout_seconds = _safe_int_env("MANIM_EXEC_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
    error_limit = _safe_int_env("MANIM_ERROR_LIMIT", DEFAULT_ERROR_LIMIT)
    retry_count = int(state.get("retry_count", 0)) if str(state.get("retry_count", "")).isdigit() else 0
    run_media_dir = _new_run_media_dir(retry_count)
    source_file = run_media_dir / GENERATED_FILE_NAME
    media_dir = run_media_dir / "media"

    try:
        source_file.write_text(injected_code, encoding="utf-8")
    except OSError as exc:
        return _execution_failure_patch(message=f"Failed to write generated code: {exc}", run_media_dir=run_media_dir)
    except Exception as exc:
        return _execution_failure_patch(message=f"Unexpected file write failure: {exc}", run_media_dir=run_media_dir)

    LOGGER.info(
        "Execution started for strategy=%s attempt=%s sandbox=%s",
        state.get("strategy", "unknown"),
        retry_count,
        mode,
    )

    snapshot_cmd, snapshot_cwd, snapshot_env, command_error = _build_manim_command(
        run_media_dir,
        source_file,
        media_dir,
        still_image=True,
    )
    if command_error is not None or snapshot_cmd is None:
        return _execution_failure_patch(
            message=command_error or "Failed to build sandboxed Manim command.",
            run_media_dir=run_media_dir,
        )

    snapshot_err = _run_manim(
        snapshot_cmd,
        timeout_seconds=timeout_seconds,
        error_limit=error_limit,
        stage="snapshot",
        cwd=snapshot_cwd,
        env=snapshot_env,
    )
    LOGGER.info(
        "Execution snapshot finished for strategy=%s attempt=%s error=%s",
        state.get("strategy", "unknown"),
        retry_count,
        bool(snapshot_err),
    )
    if snapshot_err is not None:
        return _execution_failure_patch(
            message=snapshot_err,
            failure_type="infra" if _looks_like_infra_failure(snapshot_err) else "content",
            run_media_dir=run_media_dir,
        )

    video_cmd, video_cwd, video_env, command_error = _build_manim_command(
        run_media_dir,
        source_file,
        media_dir,
        still_image=False,
    )
    if command_error is not None or video_cmd is None:
        image_path = _find_latest_file(run_media_dir, ".png")
        return _execution_failure_patch(
            message=command_error or "Failed to build sandboxed Manim command.",
            run_media_dir=run_media_dir,
            image_path=image_path,
        )

    video_err = _run_manim(
        video_cmd,
        timeout_seconds=timeout_seconds,
        error_limit=error_limit,
        stage="video",
        cwd=video_cwd,
        env=video_env,
    )
    LOGGER.info(
        "Execution video finished for strategy=%s attempt=%s error=%s",
        state.get("strategy", "unknown"),
        retry_count,
        bool(video_err),
    )
    if video_err is not None:
        image_path = _find_latest_file(run_media_dir, ".png")
        return _execution_failure_patch(
            message=video_err,
            failure_type="infra" if _looks_like_infra_failure(video_err) else "content",
            run_media_dir=run_media_dir,
            image_path=image_path,
        )

    image_path = _find_latest_file(run_media_dir, ".png")
    video_path = _find_latest_file(run_media_dir, ".mp4", exclude_parts=("partial_movie_files",))
    return {
        "render_error": None,
        "render_media_dir": str(run_media_dir),
        "render_image_path": str(image_path) if image_path else None,
        "render_video_path": str(video_path) if video_path else None,
        "execution_environment": execution_environment,
        "docker_context": docker_context,
        "failure_stage": None,
        "failure_type": None,
        "failure_reason": None,
    }
