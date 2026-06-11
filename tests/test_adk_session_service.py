"""Tests for ``JsonlSessionService`` — ADK sessions over the nano JSONL format."""

from __future__ import annotations

from google.adk.events import Event
from google.genai import types

from nano_claude.adk.session_service import JsonlSessionService
from nano_claude.session.restore import load_session
from nano_claude.session.storage import SessionStorage, session_file


def _service(tmp_path) -> JsonlSessionService:
    return JsonlSessionService(str(tmp_path), root=tmp_path / "root")


def _model_event(parts: list[types.Part], partial: bool = False) -> Event:
    return Event(
        author="nano_claude",
        invocation_id="inv",
        partial=partial or None,
        content=types.Content(role="model", parts=parts),
    )


async def test_append_event_persists_model_and_tool_events(tmp_path):
    service = _service(tmp_path)
    session = await service.create_session(app_name="nano-claude", user_id="u", session_id="sid")

    await service.append_event(
        session,
        _model_event(
            [
                types.Part(text="checking"),
                types.Part(
                    function_call=types.FunctionCall(id="c1", name="Bash", args={"command": "ls"})
                ),
            ]
        ),
    )
    await service.append_event(
        session,
        Event(
            author="nano_claude",
            invocation_id="inv",
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            id="c1", name="Bash", response={"result": "a.py"}
                        )
                    )
                ],
            ),
        ),
    )
    await service.append_event(session, _model_event([types.Part(text="done")]))
    storage = service._storage_for("sid")
    await storage.flush()

    restored = load_session(storage.path)
    assert [m["role"] for m in restored] == ["assistant", "tool", "assistant"]
    assert restored[0]["tool_calls"][0]["id"] == "c1"
    assert restored[1] == {"role": "tool", "tool_call_id": "c1", "content": "a.py"}
    assert restored[2] == {"role": "assistant", "content": "done"}
    assert len(session.events) == 3


async def test_append_event_skips_user_echo_and_partials(tmp_path):
    """User-authored events (the new_message echo) and partials never persist."""
    service = _service(tmp_path)
    session = await service.create_session(app_name="nano-claude", user_id="u", session_id="sid")

    await service.append_event(
        session,
        Event(
            author="user",
            invocation_id="inv",
            content=types.Content(role="user", parts=[types.Part(text="hi")]),
        ),
    )
    await service.append_event(session, _model_event([types.Part(text="par")], partial=True))
    storage = service._storage_for("sid")
    await storage.flush()

    assert load_session(storage.path) == []
    # The user echo still joins the in-memory transport session; partials don't.
    assert len(session.events) == 1


async def test_get_session_loads_and_repairs_from_disk(tmp_path):
    """An uncached session id loads the JSONL, crash-repaired, as ADK events."""
    path = session_file(str(tmp_path), "old", root=tmp_path / "root")
    storage = SessionStorage(path, "old")
    storage.append_message({"role": "user", "content": "read it"})
    storage.append_message(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c9", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
            ],
        }
    )
    await storage.flush()  # crash before the tool result landed

    service = _service(tmp_path)
    session = await service.get_session(app_name="nano-claude", user_id="u", session_id="old")

    assert session is not None
    calls = [fc for e in session.events for fc in e.get_function_calls()]
    responses = [fr for e in session.events for fr in e.get_function_responses()]
    assert [c.id for c in calls] == ["c9"]
    # Crash repair injected the synthetic [Interrupted] result.
    assert [r.id for r in responses] == ["c9"]
    assert responses[0].response == "[Interrupted]"


async def test_get_session_unknown_id_returns_none(tmp_path):
    service = _service(tmp_path)
    assert (
        await service.get_session(app_name="nano-claude", user_id="u", session_id="nope")
    ) is None


async def test_list_and_delete_sessions(tmp_path):
    service = _service(tmp_path)
    session = await service.create_session(app_name="nano-claude", user_id="u", session_id="sid")
    await service.append_event(session, _model_event([types.Part(text="hello")]))
    storage = service._storage_for("sid")
    await storage.flush()

    listed = await service.list_sessions(app_name="nano-claude")
    assert [s.id for s in listed.sessions] == ["sid"]

    await service.delete_session(app_name="nano-claude", user_id="u", session_id="sid")
    assert not storage.path.exists()
    assert (await service.list_sessions(app_name="nano-claude")).sessions == []


async def test_abort_race_keeps_session_resumable(tmp_path, monkeypatch):
    """A tool result persisted by the service but dropped by an abort must not
    be double-answered by the driver's repair — the append-only JSONL would be
    permanently API-invalid on --resume (two tool messages for one call id).
    """
    import litellm

    from nano_claude.adk.driver import run_turn
    from nano_claude.agent.types import AgentConfig, LoopState, StopReason
    from nano_claude.permissions.modes import PermissionMode
    from nano_claude.permissions.settings import Settings
    from nano_claude.tools.base import PermissionDecision, Tool, ToolResult
    from nano_claude.tools.registry import clear_dynamic_tools, register_tools
    from tests.conftest import FakeStream, text_chunk, tool_call_chunk, usage_chunk

    class CancelOnFinish(Tool):
        """Completes successfully, then the user's Esc lands immediately."""

        name = "CancelOnFinish"
        description = "test tool"
        reads_raw_args = True

        def to_api_schema(self):
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {"type": "object", "properties": {}},
                },
            }

        async def call(self, args, context) -> ToolResult:
            context.cancel_event.set()  # abort observed only after completion
            return ToolResult(output="real output")

        async def check_permissions(self, args, context) -> PermissionDecision:
            return PermissionDecision(behavior="allow")

    register_tools([CancelOnFinish()])
    try:
        streams = iter(
            [
                FakeStream([tool_call_chunk(0, "c1", "CancelOnFinish", "{}"), usage_chunk(5, 2)]),
                FakeStream([text_chunk("never"), usage_chunk(6, 1)]),
            ]
        )

        async def _acompletion(*args, **kwargs):
            return next(streams)

        monkeypatch.setattr(litellm, "acompletion", _acompletion)

        storage = SessionStorage(session_file(str(tmp_path), "sid", root=tmp_path / "root"), "sid")
        state = LoopState(messages=[{"role": "user", "content": "go"}], storage=storage)
        storage.append_message(state.messages[0])  # main.py persists the user msg
        config = AgentConfig(cwd=str(tmp_path), permission_mode=PermissionMode.BYPASS)

        result = await run_turn(state, config, settings=Settings(path=tmp_path / "s.json"))
        await storage.flush()

        assert result.reason is StopReason.ABORTED
        restored = load_session(storage.path)
        tool_msgs = [m for m in restored if m.get("role") == "tool"]
        # Exactly one result for the call — the real one, not a duplicate repair.
        assert tool_msgs == [{"role": "tool", "tool_call_id": "c1", "content": "real output"}]
        # Live state and the resumable JSONL agree.
        assert restored == state.messages
    finally:
        clear_dynamic_tools()
