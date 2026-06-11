"""Round-trip tests for the OpenAI-dict ⇄ genai Content/Event converters.

The final test drives a representative conversation through ADK's own
``lite_llm`` request converter (a private API of the *pinned* google-adk) and
asserts the resulting OpenAI messages are byte-identical to the originals —
that is the actual wire-fidelity guarantee the migration rests on, and the
test that should fail loudly on an ADK upgrade.
"""

from __future__ import annotations

import json

from google.adk.events import Event
from google.genai import types

from nano_claude.adk.convert import event_to_messages, view_to_contents


def _model_event(parts: list[types.Part]) -> Event:
    return Event(
        author="nano_claude",
        invocation_id="inv",
        content=types.Content(role="model", parts=parts),
    )


def test_system_message_becomes_instruction():
    instruction, contents = view_to_contents(
        [
            {"role": "system", "content": "You are nano-claude."},
            {"role": "user", "content": "hi"},
        ]
    )
    assert instruction == "You are nano-claude."
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert contents[0].parts[0].text == "hi"


def test_no_system_message_yields_none_instruction():
    instruction, contents = view_to_contents([{"role": "user", "content": "hi"}])
    assert instruction is None
    assert len(contents) == 1


def test_assistant_with_multiple_tool_calls():
    _, contents = view_to_contents(
        [
            {
                "role": "assistant",
                "content": "Running two tools.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {"name": "Read", "arguments": '{"file_path": "/x"}'},
                    },
                ],
            }
        ]
    )
    (content,) = contents
    assert content.role == "model"
    assert content.parts[0].text == "Running two tools."
    fc1, fc2 = content.parts[1].function_call, content.parts[2].function_call
    assert (fc1.id, fc1.name, fc1.args) == ("call_1", "Bash", {"command": "ls"})
    assert (fc2.id, fc2.name, fc2.args) == ("call_2", "Read", {"file_path": "/x"})


def test_tool_message_carries_raw_string_and_resolved_name():
    _, contents = view_to_contents(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "line1\nline2"},
        ]
    )
    fr = contents[1].parts[0].function_response
    assert fr.id == "call_1"
    assert fr.name == "Bash"
    # Raw string, not a {"result": ...} dict — the wire-fidelity invariant.
    assert fr.response == "line1\nline2"


def test_interrupted_tool_result_round_trips():
    """The crash-repair sentinel must survive conversion unchanged."""
    _, contents = view_to_contents(
        [{"role": "tool", "tool_call_id": "call_9", "content": "[Interrupted]"}]
    )
    assert contents[0].parts[0].function_response.response == "[Interrupted]"


def test_malformed_tool_call_arguments_fall_back_to_empty_dict():
    _, contents = view_to_contents(
        [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "Bash", "arguments": '{"oops": '},
                    }
                ],
            }
        ]
    )
    assert contents[0].parts[0].function_call.args == {}


def test_system_reminder_user_message_stays_user_text():
    _, contents = view_to_contents(
        [{"role": "user", "content": "<system-reminder>\nnudge\n</system-reminder>"}]
    )
    assert contents[0].role == "user"
    assert "<system-reminder>" in contents[0].parts[0].text


def test_event_with_text_and_tool_calls():
    event = _model_event(
        [
            types.Part(text="Let me check."),
            types.Part(
                function_call=types.FunctionCall(id="call_1", name="Bash", args={"command": "ls"})
            ),
        ]
    )
    (msg,) = event_to_messages(event)
    assert msg["role"] == "assistant"
    assert msg["content"] == "Let me check."
    (tc,) = msg["tool_calls"]
    assert tc["id"] == "call_1"
    assert tc["function"]["name"] == "Bash"
    assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}


def test_event_text_only():
    (msg,) = event_to_messages(_model_event([types.Part(text="done")]))
    assert msg == {"role": "assistant", "content": "done"}


def test_event_thought_parts_dropped():
    event = _model_event(
        [types.Part(text="secret reasoning", thought=True), types.Part(text="answer")]
    )
    (msg,) = event_to_messages(event)
    assert msg["content"] == "answer"


def test_function_response_event_unwraps_result_dict():
    event = Event(
        author="nano_claude",
        invocation_id="inv",
        content=types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id="call_1", name="Bash", response={"result": "out"}
                    )
                )
            ],
        ),
    )
    (msg,) = event_to_messages(event)
    assert msg == {"role": "tool", "tool_call_id": "call_1", "content": "out"}


def test_empty_event_yields_no_messages():
    assert event_to_messages(Event(author="nano_claude", invocation_id="inv")) == []


def test_wire_fidelity_through_adk_litellm_converter():
    """Drive a conversation through ADK's own request converter end-to-end.

    Uses the pinned ADK's private ``_content_to_message_param`` deliberately:
    if an upgrade changes how contents render to OpenAI messages, this is the
    test that must catch it.
    """
    import asyncio

    from google.adk.models import lite_llm as ll

    original = [
        {"role": "user", "content": "run ls"},
        {
            "role": "assistant",
            "content": "On it.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "Bash", "arguments": '{"command": "ls"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py\nb.py"},
        {"role": "assistant", "content": "Two files."},
    ]
    _, contents = view_to_contents(original)

    async def convert_all():
        out = []
        for content in contents:
            msg = await ll._content_to_message_param(
                content, provider="openai", model="deepseek/deepseek-v4-flash"
            )
            out.extend(msg if isinstance(msg, list) else [msg])
        return out

    wire = asyncio.run(convert_all())

    assert wire[0] == {"role": "user", "content": "run ls"}
    assert wire[1]["role"] == "assistant"
    assert wire[1]["content"] == "On it."
    (tc,) = wire[1]["tool_calls"]
    assert tc["id"] == "call_1"
    assert tc["function"]["name"] == "Bash"
    assert json.loads(tc["function"]["arguments"]) == {"command": "ls"}
    assert wire[2] == {"role": "tool", "tool_call_id": "call_1", "content": "a.py\nb.py"}
    assert wire[3]["role"] == "assistant"
    assert wire[3]["content"] == "Two files."
