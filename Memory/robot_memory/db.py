"""
robot_memory/db.py
──────────────────
Shared async database connection pool (asyncpg) + LLM client factory.

LLM support
───────────
Reads ROBOT_LLM_PROVIDER from .env:
  'groq'     — Groq API (llama-3.3-70b-versatile or similar)
  'together' — Together AI (meta-llama/Llama-3-70b-chat-hf)
  'ollama'   — local Ollama instance (default: llama3)

Set ROBOT_LLM_API_KEY and ROBOT_LLM_MODEL in .env.
The LLM client is used exclusively by consolidator.py.
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Optional

import asyncpg

# ── Load .env from project root ──────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

# ── Configuration ─────────────────────────────────────────────────────────────
DSN       = os.getenv("ROBOT_DB_DSN",        "postgresql://robot:robot@localhost:5432/robot_memory")
POOL_MIN  = int(os.getenv("ROBOT_DB_POOL_MIN",  "2"))
POOL_MAX  = int(os.getenv("ROBOT_DB_POOL_MAX",  "10"))
LOG_LEVEL = os.getenv("ROBOT_LOG_LEVEL",     "INFO").upper()

# LLM settings (used by consolidator.py)
LLM_PROVIDER = os.getenv("ROBOT_LLM_PROVIDER", "groq")           # groq | together | ollama
LLM_API_KEY  = os.getenv("ROBOT_LLM_API_KEY",  "")
LLM_MODEL    = os.getenv("ROBOT_LLM_MODEL",    "llama-3.3-70b-versatile")
LLM_BASE_URL = os.getenv("ROBOT_LLM_BASE_URL", "")               # override base URL if needed

HEAVY_ENTITY_COLS = {"image_ptrs", "video_ptr", "audio_ptr"}
HEAVY_INFO_COLS   = {"full_data", "embedding", "image_ptr", "video_ptr", "audio_ptr"}

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
log = logging.getLogger("robot_memory")

# ── Pool singleton ─────────────────────────────────────────────────────────────
_pool: Optional[asyncpg.Pool] = None


async def init_pool(
    dsn:      str = DSN,
    min_size: int = POOL_MIN,
    max_size: int = POOL_MAX,
) -> asyncpg.Pool:
    global _pool
    log.info("Connecting to database (pool min=%d max=%d)", min_size, max_size)
    _pool = await asyncpg.create_pool(dsn, min_size=min_size, max_size=max_size)
    log.info("Database pool ready")
    return _pool


async def get_pool() -> asyncpg.Pool:
    if _pool is None:
        await init_pool()
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        log.info("Database pool closed")


# ── LLM client factory ────────────────────────────────────────────────────────

def get_llm_client():
    """
    Return an OpenAI-compatible async client for the configured LLM provider.

    Supported providers:
      groq     → api.groq.com/openai/v1  (llama-3.3-70b-versatile)
      together → api.together.xyz/v1     (meta-llama/Llama-3-70b-chat-hf)
      ollama   → localhost:11434/v1      (any locally running model)

    Returns an openai.AsyncOpenAI instance. Raises ImportError if openai
    package is not installed.
    """
    try:
        from openai import AsyncOpenAI
    except ImportError as e:
        raise ImportError(
            "openai package required for LLM calls. "
            "Install with: pip install openai"
        ) from e

    provider = LLM_PROVIDER.lower()

    if provider == "groq":
        base_url = LLM_BASE_URL or "https://api.groq.com/openai/v1"
        api_key  = LLM_API_KEY or os.getenv("GROQ_API_KEY", "")
    elif provider == "together":
        base_url = LLM_BASE_URL or "https://api.together.xyz/v1"
        api_key  = LLM_API_KEY or os.getenv("TOGETHER_API_KEY", "")
    elif provider == "ollama":
        base_url = LLM_BASE_URL or "http://localhost:11434/v1"
        api_key  = "ollama"   # Ollama accepts any non-empty key
    else:
        raise ValueError(
            f"Unknown ROBOT_LLM_PROVIDER='{provider}'. "
            "Use 'groq', 'together', or 'ollama'."
        )

    if not api_key and provider != "ollama":
        log.warning(
            "ROBOT_LLM_API_KEY not set for provider '%s'. "
            "LLM consolidation calls will likely fail.", provider
        )

    return AsyncOpenAI(api_key=api_key or "dummy", base_url=base_url)


def get_llm_model() -> str:
    """Return the model string to use for LLM calls."""
    return LLM_MODEL