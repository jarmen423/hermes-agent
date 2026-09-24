"""Unauthorized Discord slash/button clicks can be made fully silent.

Default remains the existing ephemeral denial. With
``DISCORD_UNAUTHORIZED_INTERACTION_BEHAVIOR=ignore``, unauthorized
interactions send nothing, schedule no admin notify, and do not run
the button action. Authorized users still receive the normal path.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from plugins.platforms.discord.adapter import (  # noqa: E402
    ChoicePickerView,
    ClarifyChoiceView,
    DiscordAdapter,
    ExecApprovalView,
    ModelPickerView,
    SlashConfirmView,
    UpdatePromptView,
)
from gateway.config import PlatformConfig


@pytest.fixture(autouse=True)
def _clear_silent_env(monkeypatch):
    for name in (
        "DISCORD_UNAUTHORIZED_INTERACTION_BEHAVIOR",
        "DISCORD_ALLOW_ALL_USERS",
        "GATEWAY_ALLOW_ALL_USERS",
        "GATEWAY_ALLOWED_USERS",
    ):
        monkeypatch.delenv(name, raising=False)


def _interaction(user_id=99999):
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, name=f"user_{user_id}", display_name="alice", roles=[]),
        response=SimpleNamespace(send_message=AsyncMock(), edit_message=AsyncMock(), defer=AsyncMock()),
        message=SimpleNamespace(embeds=[]),
        data={"values": ["x"]},
        channel_id=5,
        guild=SimpleNamespace(id=42, owner_id=1, get_member=lambda _uid: None),
        guild_id=42,
        channel=SimpleNamespace(id=5),
    )


async def _noop(*_a, **_k):
    return ""


def _views():
    return {
        "exec": ExecApprovalView(session_key="s", allowed_user_ids={"1"}),
        "slash": SlashConfirmView(session_key="s", confirm_id="c", allowed_user_ids={"1"}),
        "update": UpdatePromptView(session_key="s", allowed_user_ids={"1"}),
        "clarify": ClarifyChoiceView(choices=["a"], clarify_id="c", allowed_user_ids={"1"}),
        "model": ModelPickerView(
            providers=[], current_model="m", current_provider="p", session_key="s",
            on_model_selected=_noop, allowed_user_ids={"1"},
        ),
        "choice": ChoicePickerView(
            choices=[{"value": "v"}], on_choice_selected=_noop, allowed_user_ids={"1"},
        ),
    }


_UNAUTH_CALLS = [
    ("exec", lambda v, i: v._resolve(i, "once", None, "x")),
    ("slash", lambda v, i: v._resolve(i, "once", None, "x")),
    ("update", lambda v, i: v._respond(i, "y", None, "x")),
    ("clarify", lambda v, i: v._resolve_choice(i, 0, "a")),
    ("clarify_other", lambda v, i: v._on_other(i)),
    ("model", lambda v, i: v._on_provider_selected(i)),
    ("choice", lambda v, i: v._on_select(i)),
]


def _view_for(name):
    views = _views()
    return views["clarify"] if name == "clarify_other" else views[name]


@pytest.mark.asyncio
@pytest.mark.parametrize("name,call", _UNAUTH_CALLS)
async def test_silent_mode_sends_nothing_on_unauthorized_click(monkeypatch, name, call):
    monkeypatch.setenv("DISCORD_UNAUTHORIZED_INTERACTION_BEHAVIOR", "ignore")
    view = _view_for(name)
    interaction = _interaction()
    await call(view, interaction)
    interaction.response.send_message.assert_not_awaited()
    interaction.response.edit_message.assert_not_called()
    interaction.response.defer.assert_not_awaited()
    assert view.resolved is False


@pytest.mark.asyncio
@pytest.mark.parametrize("name,call", _UNAUTH_CALLS)
async def test_silent_mode_hides_already_resolved_from_strangers(monkeypatch, name, call):
    monkeypatch.setenv("DISCORD_UNAUTHORIZED_INTERACTION_BEHAVIOR", "ignore")
    view = _view_for(name)
    view.resolved = True
    interaction = _interaction()
    await call(view, interaction)
    interaction.response.send_message.assert_not_awaited()
    interaction.response.edit_message.assert_not_called()


@pytest.mark.asyncio
async def test_silent_mode_slash_sends_nothing_and_skips_notify(monkeypatch):
    monkeypatch.setenv("DISCORD_UNAUTHORIZED_INTERACTION_BEHAVIOR", "ignore")
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._allowed_user_ids = {"1"}
    adapter._notify_unauthorized_slash = AsyncMock()
    interaction = _interaction()
    assert await adapter._check_slash_authorization(interaction, "/help") is False
    interaction.response.send_message.assert_not_awaited()
    await asyncio.sleep(0)
    adapter._notify_unauthorized_slash.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_mode_still_sends_slash_denial():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._allowed_user_ids = {"1"}
    interaction = _interaction()
    assert await adapter._check_slash_authorization(interaction, "/help") is False
    interaction.response.send_message.assert_awaited_once_with(
        "You're not authorized to use this command.", ephemeral=True,
    )
