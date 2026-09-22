"""Codex non-stream watchdogs honour the per-model reasoning floor.

Thinking models (glm-5.3-flash: thinking cannot be disabled, first token
routinely past the 120s TTFB / 180s idle defaults) were killed mid-think by the
implicit fuses, surfacing as BrokenPipeError. The reasoning floor now raises the
IMPLICIT TTFB/idle cutoffs; explicit operator values (and 0=disable) still win.
And the wall-clock stale kill is skipped while the TTFB/event-idle watchdogs own
progress — they already cover "nothing arrives" and "stall after progress".
"""
from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pytest


def _codex_agent(tmp_path: Path, monkeypatch, model: str = "glm-5.3-flash"):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    for var in ("HERMES_API_CALL_STALE_TIMEOUT", "HERMES_CODEX_TTFB_TIMEOUT_SECONDS",
                "HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "HERMES_CODEX_TTFB_MAX_SECONDS",
                "HERMES_CODEX_HARD_TIMEOUT_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    from run_agent import AIAgent

    agent = AIAgent(
        model=model, provider="openai-codex", api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex", quiet_mode=True,
        skip_context_files=True, skip_memory=True, platform="cli",
    )
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    return agent


def test_reasoning_floor_raises_implicit_ttfb_and_idle(tmp_path, monkeypatch):
    """glm-5.3-flash (300s floor) lifts the small-prompt 120s TTFB / 12s idle
    tiers; a non-reasoning model keeps them."""
    from agent.chat_completion_helpers import _resolve_nonstream_watchdogs

    thinker = _resolve_nonstream_watchdogs(
        _codex_agent(tmp_path, monkeypatch, "glm-5.3-flash"), {"model": "glm-5.3-flash", "input": "hi"})
    assert thinker.ttfb_enabled and thinker.idle_enabled
    assert thinker.ttfb_timeout == 300.0
    assert thinker.idle_timeout == 300.0

    chat = _resolve_nonstream_watchdogs(
        _codex_agent(tmp_path, monkeypatch, "gpt-4o"), {"model": "gpt-4o", "input": "hi"})
    assert (chat.idle_timeout, chat.ttfb_timeout) == (12.0, 120.0)


def test_explicit_ttfb_and_idle_win_over_reasoning_floor(tmp_path, monkeypatch):
    """An operator-set fuse (or 0=disable) is never raised by the floor."""
    from agent.chat_completion_helpers import _resolve_nonstream_watchdogs

    agent = _codex_agent(tmp_path, monkeypatch, "glm-5.3-flash")
    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "45")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "20")
    explicit = _resolve_nonstream_watchdogs(agent, {"model": "glm-5.3-flash", "input": "hi"})
    assert (explicit.idle_timeout, explicit.ttfb_timeout) == (20.0, 45.0)

    monkeypatch.setenv("HERMES_CODEX_TTFB_TIMEOUT_SECONDS", "0")
    disabled = _resolve_nonstream_watchdogs(agent, {"model": "glm-5.3-flash", "input": "hi"})
    assert disabled.ttfb_enabled is False


def test_owned_stream_skips_wall_clock_stale_kill(tmp_path, monkeypatch):
    """Events arrive, then the stream hangs: the tiny wall-clock stale fuse must
    NOT fire (it is skipped while TTFB/idle own progress) — the idle fuse does."""
    from agent import chat_completion_helpers as h

    agent = _codex_agent(tmp_path, monkeypatch, "glm-5.3-flash")
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "0.3")
    monkeypatch.setenv("HERMES_CODEX_EVENT_STALE_TIMEOUT_SECONDS", "2")
    monkeypatch.setenv("HERMES_CODEX_HARD_TIMEOUT_SECONDS", "0")

    closes: list = []
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_kwargs: _hanging_stream(agent)))
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **_k: client)
    monkeypatch.setattr(agent, "_abort_request_openai_client",
                         lambda _client, reason=None: closes.append(reason))
    monkeypatch.setattr(agent, "_close_request_openai_client",
                         lambda _client, reason=None: closes.append(reason))

    with pytest.raises(TimeoutError, match="no SSE events"):
        h.interruptible_api_call(agent, {"model": "glm-5.3-flash", "input": "hi"})
    assert "codex_stream_idle_kill" in closes
    assert "stale_call_kill" not in closes


def _hanging_stream(agent):
    yield SimpleNamespace(type="response.created")
    yield SimpleNamespace(type="response.reasoning_text.delta", delta="thinking")
    while getattr(agent, "_active_codex_stream_request_token", None) is not None:
        time.sleep(0.02)
    raise ConnectionError("retired stalled stream")
