"""Reasoning-effort over ACP: model ``_meta`` advertisement, ``session/set_model``
application, per-session persistence, and rebuild survival.

The wire contract mirrors what ACP clients already read: ``supportsReasoningEffort``
gates the picker, ``reasoningEfforts`` lists levels, ``reasoningEffort`` is the
session's current level. Levels ride ``session/set_model`` ``_meta`` back and land
on the live agent's ``reasoning_config`` — never on global config.
"""

from types import SimpleNamespace

import pytest
from unittest.mock import patch

from acp_adapter.model_catalog import (
    ACP_REASONING_EFFORTS,
    _reasoning_meta,
    build_model_state,
)
from acp_adapter.server import HermesACPAgent, _session_reasoning_effort
from acp_adapter.session import SessionManager, _apply_session_reasoning


def _agent(**overrides):
    """AIAgent stand-in with the attributes the reasoning path touches."""
    base = {
        "model": "model-a",
        "provider": "openrouter",
        "base_url": "",
        "api_mode": None,
        "reasoning_config": None,
        "context_compressor": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _manager():
    return SessionManager(agent_factory=lambda: _agent())


# ---------------------------------------------------------------------------
# _reasoning_meta / build_model_state
# ---------------------------------------------------------------------------


class TestReasoningMeta:
    def test_unsupported_model_marks_the_flag_off(self):
        meta = _reasoning_meta(False, "high")
        assert meta == {"supportsReasoningEffort": False}

    def test_supported_model_advertises_full_ladder(self):
        meta = _reasoning_meta(True, None)
        assert meta["supportsReasoningEffort"] is True
        assert "reasoningEffort" not in meta
        assert [e["value"] for e in meta["reasoningEfforts"]] == [
            value for value, _label in ACP_REASONING_EFFORTS
        ]
        assert all(e["default"] is False for e in meta["reasoningEfforts"])

    def test_current_effort_marks_default_and_top_level_key(self):
        meta = _reasoning_meta(True, "High")
        assert meta["reasoningEffort"] == "high"
        defaults = {e["value"]: e["default"] for e in meta["reasoningEfforts"]}
        assert defaults == {**{v: False for v, _ in ACP_REASONING_EFFORTS}, "high": True}

    def test_none_effort_marks_off_as_current(self):
        meta = _reasoning_meta(True, "none")
        assert meta["reasoningEffort"] == "none"
        assert next(e for e in meta["reasoningEfforts"] if e["value"] == "none")["default"]


class TestBuildModelStateMeta:
    def _payload(self, reasoning: bool):
        return {
            "providers": [
                {
                    "id": "openrouter",
                    "models": [{"id": "model-a"}],
                    "capabilities": {"model-a": {"reasoning": reasoning}},
                }
            ]
        }

    def test_reasoning_capable_rows_carry_the_ladder(self):
        with patch(
            "hermes_cli.inventory.build_models_payload", return_value=self._payload(True)
        ), patch(
            "acp_adapter.model_catalog._named_custom_provider_catalogs", return_value=[]
        ):
            state = build_model_state("model-a", "openrouter", "", reasoning_effort="low")
        assert state is not None
        model = next(m for m in state.available_models if m.model_id == "openrouter:model-a")
        meta = model.field_meta
        assert meta["supportsReasoningEffort"] is True
        assert meta["reasoningEffort"] == "low"

    def test_non_reasoning_rows_opt_out(self):
        with patch(
            "hermes_cli.inventory.build_models_payload", return_value=self._payload(False)
        ), patch(
            "acp_adapter.model_catalog._named_custom_provider_catalogs", return_value=[]
        ):
            state = build_model_state("model-a", "openrouter", "")
        model = next(m for m in state.available_models if m.model_id == "openrouter:model-a")
        assert model.field_meta == {"supportsReasoningEffort": False}


# ---------------------------------------------------------------------------
# _session_reasoning_effort — the advertised "current" value
# ---------------------------------------------------------------------------


class TestSessionReasoningEffort:
    def test_session_override_wins_over_agent_config(self):
        state = SimpleNamespace(
            reasoning_effort="ultra",
            agent=_agent(reasoning_config={"enabled": True, "effort": "low"}),
        )
        assert _session_reasoning_effort(state) == "ultra"

    def test_agent_disabled_reports_none(self):
        state = SimpleNamespace(
            reasoning_effort=None,
            agent=_agent(reasoning_config={"enabled": False}),
        )
        assert _session_reasoning_effort(state) == "none"

    def test_unconfigured_reports_missing(self):
        state = SimpleNamespace(reasoning_effort=None, agent=_agent(reasoning_config=None))
        assert _session_reasoning_effort(state) is None


# ---------------------------------------------------------------------------
# set_session_model — effort meta application
# ---------------------------------------------------------------------------


class TestSetSessionModelReasoning:
    @pytest.mark.asyncio
    async def test_effort_meta_applies_to_live_agent(self):
        manager = _manager()
        acp_agent = HermesACPAgent(session_manager=manager)
        resp = await acp_agent.new_session(cwd="/tmp")
        state = manager.get_session(resp.session_id)

        await acp_agent.set_session_model(
            "openrouter:model-b", resp.session_id, reasoningEffort="high"
        )

        assert state.reasoning_effort == "high"
        assert state.agent.reasoning_config == {"enabled": True, "effort": "high"}

    @pytest.mark.asyncio
    async def test_none_disables_reasoning(self):
        manager = _manager()
        acp_agent = HermesACPAgent(session_manager=manager)
        resp = await acp_agent.new_session(cwd="/tmp")
        state = manager.get_session(resp.session_id)

        await acp_agent.set_session_model(
            "openrouter:model-a", resp.session_id, reasoningEffort="none"
        )

        assert state.reasoning_effort == "none"
        assert state.agent.reasoning_config == {"enabled": False}

    @pytest.mark.asyncio
    async def test_junk_effort_is_rejected_and_keeps_prior_override(self):
        manager = _manager()
        acp_agent = HermesACPAgent(session_manager=manager)
        resp = await acp_agent.new_session(cwd="/tmp")
        state = manager.get_session(resp.session_id)

        await acp_agent.set_session_model(
            "openrouter:model-a", resp.session_id, reasoningEffort="medium"
        )
        await acp_agent.set_session_model(
            "openrouter:model-a", resp.session_id, reasoningEffort="bogus!!"
        )

        assert state.reasoning_effort == "medium"
        assert state.agent.reasoning_config == {"enabled": True, "effort": "medium"}

    @pytest.mark.asyncio
    async def test_override_survives_a_later_model_switch(self):
        """A session override must re-apply after the agent rebuild, or the
        advertised current effort would lie about the live request config."""
        manager = _manager()
        acp_agent = HermesACPAgent(session_manager=manager)
        resp = await acp_agent.new_session(cwd="/tmp")
        state = manager.get_session(resp.session_id)

        await acp_agent.set_session_model(
            "openrouter:model-a", resp.session_id, reasoningEffort="xhigh"
        )
        # Second switch carries no effort meta: the override must persist.
        await acp_agent.set_session_model("openrouter:model-c", resp.session_id)

        assert state.model == "model-c"
        assert state.reasoning_effort == "xhigh"
        assert state.agent.reasoning_config == {"enabled": True, "effort": "xhigh"}


# ---------------------------------------------------------------------------
# Persistence round-trip
# ---------------------------------------------------------------------------


class TestReasoningPersistence:
    def test_session_meta_carries_and_restores_effort(self):
        manager = _manager()
        state = manager.create_session(cwd="/tmp")
        state.history.append({"role": "user", "content": "hello"})
        state.reasoning_effort = "low"
        manager.save_session(state.session_id)

        db = manager._get_db()
        import json

        meta = json.loads(db.get_session(state.session_id)["model_config"])
        assert meta["reasoning_effort"] == "low"

        # Restore path: a fresh manager rebuilds the agent and re-applies.
        manager2 = _manager()
        restored = manager2._restore(state.session_id)
        assert restored is not None
        assert restored.reasoning_effort == "low"
        assert restored.agent.reasoning_config == {"enabled": True, "effort": "low"}

    def test_apply_session_reasoning_ignores_unparseable(self):
        agent = _agent(reasoning_config={"enabled": True, "effort": "low"})
        state = SimpleNamespace(reasoning_effort="bogus", agent=agent)
        _apply_session_reasoning(state)
        assert agent.reasoning_config == {"enabled": True, "effort": "low"}
