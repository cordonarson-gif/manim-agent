from __future__ import annotations

import os
import threading
from typing import Callable, TypeVar

from langchain_openai import ChatOpenAI

T = TypeVar("T")

# ============================================================
# API configuration placeholders
# ============================================================
# TODO(USER): Keep this empty for published code and use environment variable
# `DEEPSEEK_API_KEY` during local development.
DEEPSEEK_API_KEY_PLACEHOLDER: str = ""

# TODO(USER): If needed, modify DeepSeek base URL here.
DEEPSEEK_BASE_URL_PLACEHOLDER: str = "https://api.deepseek.com/v1"

# Default DeepSeek text model used by planner/coder.
DEEPSEEK_PRO_MODEL_PLACEHOLDER: str = "deepseek-v4-pro"

DEEPSEEK_FLASH_ALIASES: set[str] = {
    "deepseekv4flash",
    "deepseek-v4-flash",
    "deepseek_v4_flash",
}

DEEPSEEK_PRO_ALIASES: set[str] = {
    "deepseekv4pro",
    "deepseek-v4-pro",
    "deepseek_v4_pro",
}

# TODO(USER): Keep this empty for published code and use environment variable
# `QWEN_API_KEY` during local development.
QWEN_API_KEY_PLACEHOLDER: str = ""

# TODO(USER): If needed, modify Qwen base URL here.
QWEN_BASE_URL_PLACEHOLDER: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def _safe_float_env(name: str, default: float) -> float:
    """Read float env value with fallback."""

    raw: str | None = os.getenv(name)
    if raw is None:
        return default

    try:
        return float(raw)
    except ValueError:
        return default


def _safe_int_env(name: str, default: int) -> int:
    """Read int env value with fallback."""

    raw: str | None = os.getenv(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _timeout_from_env(primary_env: str, default: int, *, fallback_env: str | None = None) -> int:
    """Resolve timeout from a primary env, then an optional fallback env."""

    primary_value = os.getenv(primary_env)
    if primary_value is not None:
        return _safe_int_env(primary_env, default)
    if fallback_env:
        return _safe_int_env(fallback_env, default)
    return default


def get_llm_timeout_seconds() -> int:
    """Return shared text-model timeout in seconds."""

    return _safe_int_env("MANIM_LLM_TIMEOUT_SECONDS", 120)


def get_planner_timeout_seconds() -> int:
    """Return planner-model timeout in seconds."""

    return _timeout_from_env("MANIM_PLANNER_TIMEOUT_SECONDS", 120, fallback_env="MANIM_LLM_TIMEOUT_SECONDS")


def get_coder_timeout_seconds() -> int:
    """Return coder-model timeout in seconds."""

    return _timeout_from_env("MANIM_CODER_TIMEOUT_SECONDS", 180, fallback_env="MANIM_LLM_TIMEOUT_SECONDS")


def get_vision_timeout_seconds() -> int:
    """Return vision-model timeout in seconds."""

    return _safe_int_env("MANIM_VISION_TIMEOUT_SECONDS", 60)


def invoke_with_hard_timeout(
    callback: Callable[[], T],
    *,
    timeout_seconds: int,
    timeout_label: str,
) -> T:
    """Run a blocking model call with a hard wall-clock timeout."""

    effective_timeout = max(1, int(timeout_seconds))
    result_box: dict[str, T] = {}
    error_box: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            result_box["value"] = callback()
        except BaseException as exc:
            error_box["error"] = exc

    worker = threading.Thread(
        target=_worker,
        daemon=True,
        name=f"{timeout_label.replace(' ', '_')}_worker",
    )
    worker.start()
    worker.join(effective_timeout)

    if worker.is_alive():
        raise TimeoutError(f"{timeout_label} exceeded hard timeout of {effective_timeout} seconds.")
    if "error" in error_box:
        raise error_box["error"]
    return result_box["value"]


def _resolve_credential(env_name: str, placeholder: str) -> str:
    """Resolve credential from env first, then placeholder."""

    env_value: str = os.getenv(env_name, "").strip()
    if env_value:
        return env_value
    return placeholder.strip()


def _require_non_empty(value: str, message: str) -> str:
    """Raise clear error if value is empty."""

    if not value.strip():
        raise RuntimeError(message)
    return value.strip()


def _normalize_deepseek_model(model_name: str) -> str:
    """Map legacy flash model names to the project default pro model."""

    normalized = model_name.strip()
    lowered = normalized.lower()
    if lowered in DEEPSEEK_FLASH_ALIASES or lowered in DEEPSEEK_PRO_ALIASES:
        return os.getenv("DEEPSEEK_PRO_MODEL", "").strip() or DEEPSEEK_PRO_MODEL_PLACEHOLDER
    return normalized


def build_deepseek_chat_model(
    *,
    model_env_name: str,
    default_model: str,
    temperature_env_name: str,
    default_temperature: float,
    timeout_seconds: int | None = None,
) -> ChatOpenAI:
    """Build ChatOpenAI client that is strictly routed to DeepSeek."""

    raw_model_name: str = os.getenv(model_env_name, default_model).strip() or default_model
    model_name: str = _normalize_deepseek_model(raw_model_name)
    temperature: float = _safe_float_env(temperature_env_name, default_temperature)
    resolved_timeout_seconds = timeout_seconds if timeout_seconds is not None else get_llm_timeout_seconds()
    max_retries: int = _safe_int_env("MANIM_LLM_MAX_RETRIES", 2)

    api_key: str = _require_non_empty(
        _resolve_credential("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY_PLACEHOLDER),
        (
            "Missing DeepSeek API key. "
            "Set environment variable DEEPSEEK_API_KEY or edit "
            "DEEPSEEK_API_KEY_PLACEHOLDER in utils/model_provider.py."
        ),
    )
    base_url: str = (
        os.getenv("DEEPSEEK_BASE_URL", "").strip()
        or DEEPSEEK_BASE_URL_PLACEHOLDER
    )

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        timeout=resolved_timeout_seconds,
        max_retries=max_retries,
    )


def build_qwen_vision_model(
    *,
    model_env_name: str = "MANIM_VISION_MODEL",
    default_model: str = "qwen-vl-max",
    temperature_env_name: str = "MANIM_VISION_TEMPERATURE",
    default_temperature: float = 0.1,
) -> ChatOpenAI:
    """Build ChatOpenAI client that is strictly routed to Qwen vision endpoint."""

    model_name: str = os.getenv(model_env_name, default_model).strip() or default_model
    temperature: float = _safe_float_env(temperature_env_name, default_temperature)
    timeout_seconds: int = get_vision_timeout_seconds()
    max_retries: int = _safe_int_env("MANIM_VISION_MAX_RETRIES", 1)

    api_key: str = _require_non_empty(
        _resolve_credential("QWEN_API_KEY", QWEN_API_KEY_PLACEHOLDER),
        (
            "Missing Qwen API key. "
            "Set environment variable QWEN_API_KEY or edit "
            "QWEN_API_KEY_PLACEHOLDER in utils/model_provider.py."
        ),
    )
    base_url: str = (
        os.getenv("QWEN_BASE_URL", "").strip()
        or QWEN_BASE_URL_PLACEHOLDER
    )

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout_seconds,
        max_retries=max_retries,
    )
