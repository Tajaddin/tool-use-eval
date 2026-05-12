"""Backend protocol + Mock + Anthropic adapters.

Every backend takes ``(scenario_text, tools, system_prompt)`` and returns
either a :class:`ToolCall` (with name + arguments) or a plain text reply.
Backends are responsible for normalizing the underlying SDK's response
shape into this single representation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolCall:
    """One tool invocation extracted from a model response."""

    name: str
    arguments: dict
    raw: Any = None


@dataclass
class ModelResponse:
    """Backend output: either a tool call or a freeform text reply."""

    tool_call: ToolCall | None
    text: str
    raw: Any = None

    @property
    def used_tool(self) -> bool:
        return self.tool_call is not None


class ToolUseBackend(Protocol):
    model: str

    def call(self, prompt: str, tools: list[dict], system: str | None = None) -> ModelResponse: ...


# ---------------------------------------------------------------------------
# MockBackend
# ---------------------------------------------------------------------------


class MockBackend:
    """Deterministic backend for tests. The handler returns a ``ModelResponse``
    directly so we can simulate every failure mode without an API."""

    def __init__(self, model: str = "mock", handler=None) -> None:
        self.model = model
        self.handler = handler
        self.calls: list[dict] = []

    def call(self, prompt: str, tools: list[dict], system: str | None = None) -> ModelResponse:
        self.calls.append({"prompt": prompt, "tools": tools, "system": system})
        if self.handler is None:
            return ModelResponse(tool_call=None, text="")
        return self.handler(prompt, tools, system)


# ---------------------------------------------------------------------------
# AnthropicBackend
# ---------------------------------------------------------------------------


class AnthropicBackend:
    """Wraps Anthropic's tools API. Tools must be in Anthropic format
    (``input_schema`` + ``name`` + ``description``)."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        from anthropic import Anthropic  # lazy

        self.model = model or os.environ.get("TOOL_USE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def call(self, prompt: str, tools: list[dict], system: str | None = None) -> ModelResponse:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 1024,
            "tools": tools,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        resp = self.client.messages.create(**kwargs)
        tool_call: ToolCall | None = None
        text_parts: list[str] = []
        for block in resp.content:
            btype = getattr(block, "type", "")
            if btype == "tool_use":
                tool_call = ToolCall(
                    name=block.name,
                    arguments=dict(block.input),
                    raw=block,
                )
            elif btype == "text":
                text_parts.append(block.text)
        return ModelResponse(tool_call=tool_call, text="".join(text_parts), raw=resp)
