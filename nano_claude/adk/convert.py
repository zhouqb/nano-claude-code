"""OpenAI-format message dicts ⇄ google.genai ``Content`` / ADK ``Event``.

``LoopState.messages`` (and the session JSONL) keep the OpenAI shape —
``{"role": ..., "content": ..., "tool_calls": [...], "tool_call_id": ...}`` —
as the canonical representation; see ``session/storage.py``. These converters
translate at the two ADK boundaries only:

- :func:`view_to_contents` turns the compaction pipeline's view into the
  ``(system_instruction, contents)`` pair that ``before_model_callback``
  installs on the outgoing ``LlmRequest``.
- :func:`event_to_messages` turns a (non-partial) ADK ``Event`` back into
  message dicts for recording into state + storage.

Fidelity notes, verified against the pinned ADK's ``lite_llm`` converter:

- Tool-call ids round-trip in both directions (``FunctionCall.id`` /
  ``FunctionResponse.id`` ⇄ ``tool_call_id``); crash repair keys on them.
- Tool results are carried as *raw strings*, not ``{"result": ...}`` dicts.
  ``types.FunctionResponse`` validates ``response`` as a dict, but ADK's
  converter has an explicit ``isinstance(response, str)`` branch that emits
  the string as the tool-message content verbatim — exactly today's wire
  format, with none of the JSON-escaping overhead a dict wrapper would add
  to multi-KB tool outputs. We use ``model_construct`` to carry the string.
"""

from __future__ import annotations

import json
from typing import Any

from google.adk.events import Event
from google.genai import types


def _function_call_part(tool_call: dict) -> types.Part:
    """Build a ``function_call`` Part from an OpenAI assistant ``tool_calls`` entry."""
    fn = tool_call.get("function") or {}
    raw_args = fn.get("arguments") or "{}"
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = None
    if not isinstance(args, dict):
        # genai FunctionCall.args must be a dict. Malformed/non-object argument
        # JSON can't be represented; the loop already answered such calls with a
        # fixed error result, so an empty dict here only affects what the model
        # re-reads of its own bad call.
        args = {}
    return types.Part(
        function_call=types.FunctionCall(id=tool_call.get("id"), name=fn.get("name"), args=args)
    )


def _function_response_part(call_id: str | None, name: str | None, output: str) -> types.Part:
    """Build a ``function_response`` Part carrying ``output`` as a raw string.

    ``model_construct`` bypasses the dict-only validation on ``response``; the
    ADK LiteLlm converter's string branch then emits ``output`` verbatim as the
    tool-message content (see module docstring).
    """
    response = types.FunctionResponse.model_construct(id=call_id, name=name or "", response=output)
    return types.Part.model_construct(function_response=response)


def view_to_contents(messages: list[dict]) -> tuple[str | None, list[types.Content]]:
    """Convert an OpenAI-format message list into ADK request inputs.

    Returns ``(system_instruction, contents)``: ``system`` messages (in
    practice the leading one) are folded into the instruction string; the
    rest become ``types.Content`` entries in order. Tool messages resolve
    their function *name* from the preceding assistant ``tool_calls`` entry
    with the same id.
    """
    system_parts: list[str] = []
    contents: list[types.Content] = []
    call_names: dict[str, str] = {}

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        if role == "assistant":
            parts: list[types.Part] = []
            if content:
                parts.append(types.Part(text=content))
            for tc in msg.get("tool_calls") or []:
                if tc.get("id") and (tc.get("function") or {}).get("name"):
                    call_names[tc["id"]] = tc["function"]["name"]
                parts.append(_function_call_part(tc))
            if parts:
                contents.append(types.Content(role="model", parts=parts))
            continue
        if role == "tool":
            call_id = msg.get("tool_call_id")
            parts = [_function_response_part(call_id, call_names.get(call_id or ""), content or "")]
            contents.append(types.Content(role="user", parts=parts))
            continue
        # "user" (and anything unrecognized, defensively) → plain user text.
        contents.append(types.Content(role="user", parts=[types.Part(text=content or "")]))

    return ("\n\n".join(system_parts) or None), contents


def _response_to_text(response: Any) -> str:
    """Extract the tool-output string from a ``FunctionResponse.response`` value.

    Our tool adapter returns raw strings (carried via ``model_construct``) or
    ``{"result": <str>}``; anything else (e.g. a future ADK-native tool) is
    serialized as JSON so nothing is silently dropped.
    """
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        result = response.get("result")
        if isinstance(result, str) and set(response) == {"result"}:
            return result
        return json.dumps(response)
    return "" if response is None else json.dumps(response)


def event_to_messages(event: Event) -> list[dict]:
    """Convert a non-partial ADK ``Event`` into OpenAI-format message dicts.

    A model event yields one assistant message (text and/or ``tool_calls``);
    an event carrying function responses yields one ``tool`` message per
    response. Thought parts (reasoning) are deliberately dropped — they never
    enter ``state.messages`` (see the old loop's contract), so they are never
    sent back to the model.
    """
    content = event.content
    if content is None or not content.parts:
        return []

    text_parts: list[str] = []
    tool_calls: list[dict] = []
    tool_messages: list[dict] = []
    for part in content.parts:
        if part.function_response is not None:
            fr = part.function_response
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": fr.id,
                    "content": _response_to_text(fr.response),
                }
            )
        elif part.function_call is not None:
            fc = part.function_call
            tool_calls.append(
                {
                    "id": fc.id,
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(fc.args or {}),
                    },
                }
            )
        elif part.text and not part.thought:
            text_parts.append(part.text)

    messages: list[dict] = []
    text = "".join(text_parts)
    if content.role == "user" and not tool_calls and not tool_messages:
        if text:
            messages.append({"role": "user", "content": text})
    elif text or tool_calls:
        assistant: dict = {"role": "assistant", "content": text or None}
        if tool_calls:
            assistant["tool_calls"] = tool_calls
        else:
            assistant["content"] = text
        messages.append(assistant)
    messages.extend(tool_messages)
    return messages
