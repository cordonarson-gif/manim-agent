from __future__ import annotations

import os

from langchain_openai import ChatOpenAI

# ============================================================
# API configuration placeholders
# ============================================================
# TODO(USER): Keep this empty for published code and use environment variable
# `DEEPSEEK_API_KEY` during local development.
DEEPSEEK_API_KEY_PLACEHOLDER: str = ""

# TODO(USER): If needed, modify DeepSeek base URL here.
DEEPSEEK_BASE_URL_PLACEHOLDER: str = "https://api.deepseek.com/v1"

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


def build_deepseek_chat_model(
    *,
    model_env_name: str,
    default_model: str,
    temperature_env_name: str,
    default_temperature: float,
) -> ChatOpenAI:
    """Build ChatOpenAI client that is strictly routed to DeepSeek."""

    model_name: str = os.getenv(model_env_name, default_model).strip() or default_model
    temperature: float = _safe_float_env(temperature_env_name, default_temperature)

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
    )
