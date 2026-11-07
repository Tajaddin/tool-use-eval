"""A response can hold several tool_use blocks; every one of them is scored."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from tool_use_eval import (
    AnthropicBackend,
    FailureMode,
    MockBackend,
    Scenario,
    ToolCall,
    builtin_scenarios,
    evaluate,
    score_call,
)
from tool_use_eval.backends import ModelResponse

WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get current weather.",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}

WEATHER = Scenario(
    id="weather",
    prompt="weather in paris?",
    tools=[WEATHER_TOOL],
    expected_tool="get_weather",
    required_args=["city"],
    arg_value_checks={"city": lambda v: v.lower() == "paris"},
)

CHITCHAT = Scenario(id="joke", prompt="tell me a joke", tools=[WEATHER_TOOL], expected_tool=None)


def _block(name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=args)


def _anthropic_backend(monkeypatch, content: list) -> AnthropicBackend:
    resp = SimpleNamespace(content=content)
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **_: resp))
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=lambda api_key=None: client))
    return AnthropicBackend(model="fake")


def test_anthropic_backend_keeps_every_tool_use_block_in_order(monkeypatch) -> None:
    backend = _anthropic_backend(
        monkeypatch,
        [
            SimpleNamespace(type="text", text="Let me check."),
            _block("calculate", {"expression": "1+1"}),
            _block("get_weather", {"city": "Paris"}),
        ],
    )
    resp = backend.call("weather in paris?", [WEATHER_TOOL])
    assert [c.name for c in resp.tool_calls] == ["calculate", "get_weather"]
    assert resp.tool_call.name == "calculate"
    assert resp.text == "Let me check."


def test_wrong_call_followed_by_right_call_fails(monkeypatch) -> None:
    backend = _anthropic_backend(
        monkeypatch, [_block("calculate", {"expression": "1+1"}), _block("get_weather", {"city": "Paris"})]
    )
    summary = evaluate(backend, [WEATHER])
    assert summary.pass_count == 0
    assert summary.failure_counts[FailureMode.WRONG_TOOL_NAME.value] == 1
    assert [c.name for c in summary.results[0].tool_calls] == ["calculate", "get_weather"]


def test_extra_call_after_the_expected_one_is_its_own_mode() -> None:
    search = next(s for s in builtin_scenarios() if s.id == "multi-search-then-email")
    resp = ModelResponse(
        tool_calls=[
            ToolCall(name="web_search", arguments={"query": "OpenAI CEO"}),
            ToolCall(name="send_email", arguments={"to": "bob@example.com"}),
        ]
    )
    card = score_call(search, resp)
    assert not card.passed
    assert card.failures == [FailureMode.EXTRA_TOOL_CALL]
    assert "'send_email'" in card.notes[0]


def test_right_call_followed_by_wrong_args_fails() -> None:
    resp = ModelResponse(
        tool_calls=[
            ToolCall(name="get_weather", arguments={"city": "Paris"}),
            ToolCall(name="get_weather", arguments={"city": "Berlin"}),
        ]
    )
    card = score_call(WEATHER, resp)
    assert card.failures == [FailureMode.WRONG_ARG_VALUE]


def test_failure_counts_count_scenarios_not_calls() -> None:
    no_city = ToolCall(name="get_weather", arguments={})
    backend = MockBackend(handler=lambda *_: ModelResponse(tool_calls=[no_city, no_city]))
    summary = evaluate(backend, [WEATHER])
    assert summary.results[0].scorecard.failures == [FailureMode.MISSING_ARG] * 2
    assert summary.failure_counts[FailureMode.MISSING_ARG.value] == 1


def test_every_call_correct_passes() -> None:
    call = ToolCall(name="get_weather", arguments={"city": "Paris"})
    assert score_call(WEATHER, ModelResponse(tool_calls=[call, call])).passed


def test_tool_when_not_expected_counted_once_and_names_every_call() -> None:
    resp = ModelResponse(
        tool_calls=[
            ToolCall(name="get_weather", arguments={"city": "a"}),
            ToolCall(name="calculate", arguments={}),
        ]
    )
    card = score_call(CHITCHAT, resp)
    assert card.failures == [FailureMode.TOOL_WHEN_NOT_EXPECTED]
    assert "get_weather" in card.notes[0] and "calculate" in card.notes[0]


def test_single_tool_call_keyword_still_works() -> None:
    resp = ModelResponse(tool_call=ToolCall(name="get_weather", arguments={"city": "Paris"}), text="")
    assert [c.name for c in resp.tool_calls] == ["get_weather"]
    assert score_call(WEATHER, resp).passed
