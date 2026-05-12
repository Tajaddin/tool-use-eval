"""Tool-use-eval runner tests with MockBackend simulating each failure mode."""

from __future__ import annotations

from tool_use_eval import (
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
        "properties": {
            "city": {"type": "string"},
            "units": {"type": "string"},
        },
        "required": ["city"],
    },
}


def _scenario(*, expected_tool=None, required_args=None, value_checks=None) -> Scenario:
    return Scenario(
        id="t",
        prompt="weather in paris?",
        tools=[WEATHER_TOOL],
        expected_tool=expected_tool,
        required_args=required_args or [],
        arg_value_checks=value_checks or {},
    )


def test_pass_when_no_tool_expected_and_text_only() -> None:
    sc = _scenario(expected_tool=None)
    resp = ModelResponse(tool_call=None, text="A joke.")
    card = score_call(sc, resp)
    assert card.passed is True
    assert card.failures == []


def test_fail_tool_when_not_expected() -> None:
    sc = _scenario(expected_tool=None)
    resp = ModelResponse(tool_call=ToolCall(name="get_weather", arguments={"city": "x"}), text="")
    card = score_call(sc, resp)
    assert card.passed is False
    assert FailureMode.TOOL_WHEN_NOT_EXPECTED in card.failures


def test_fail_no_tool_when_expected() -> None:
    sc = _scenario(expected_tool="get_weather", required_args=["city"])
    resp = ModelResponse(tool_call=None, text="The weather is sunny.")
    card = score_call(sc, resp)
    assert FailureMode.NO_TOOL_WHEN_EXPECTED in card.failures


def test_fail_wrong_tool_name() -> None:
    sc = _scenario(expected_tool="get_weather", required_args=["city"])
    resp = ModelResponse(tool_call=ToolCall(name="calculate", arguments={"x": 1}), text="")
    card = score_call(sc, resp)
    assert FailureMode.WRONG_TOOL_NAME in card.failures


def test_fail_missing_arg() -> None:
    sc = _scenario(expected_tool="get_weather", required_args=["city", "units"])
    resp = ModelResponse(
        tool_call=ToolCall(name="get_weather", arguments={"city": "paris"}), text="",
    )
    card = score_call(sc, resp)
    assert FailureMode.MISSING_ARG in card.failures


def test_fail_wrong_arg_type() -> None:
    sc = _scenario(expected_tool="get_weather", required_args=["city"])
    resp = ModelResponse(
        tool_call=ToolCall(name="get_weather", arguments={"city": 123}), text="",
    )
    card = score_call(sc, resp)
    assert FailureMode.WRONG_ARG_TYPE in card.failures


def test_fail_wrong_arg_value() -> None:
    sc = _scenario(
        expected_tool="get_weather",
        required_args=["city"],
        value_checks={"city": lambda v: v.lower() == "paris"},
    )
    resp = ModelResponse(
        tool_call=ToolCall(name="get_weather", arguments={"city": "Berlin"}), text="",
    )
    card = score_call(sc, resp)
    assert FailureMode.WRONG_ARG_VALUE in card.failures


def test_pass_correct_tool_with_required_args() -> None:
    sc = _scenario(expected_tool="get_weather", required_args=["city"])
    resp = ModelResponse(
        tool_call=ToolCall(name="get_weather", arguments={"city": "Paris"}), text="",
    )
    card = score_call(sc, resp)
    assert card.passed is True


def test_evaluate_end_to_end_with_mock() -> None:
    def handler(prompt, tools, system):
        if "weather" in prompt.lower():
            return ModelResponse(
                tool_call=ToolCall(name="get_weather", arguments={"city": "Paris"}), text=""
            )
        return ModelResponse(tool_call=None, text="No tool needed.")

    backend = MockBackend(model="mock-correct", handler=handler)
    scenarios = builtin_scenarios()[:5]
    summary = evaluate(backend, scenarios)
    assert summary.n == 5
    assert isinstance(summary.pass_rate, float)
    assert set(summary.failure_counts.keys()) == {f.value for f in FailureMode}


def test_builtin_scenarios_have_30_pairs() -> None:
    scenarios = builtin_scenarios()
    assert len(scenarios) == 30
    # Every scenario references one of the five built-in tools or has expected_tool=None.
    tool_names = {t["name"] for t in scenarios[0].tools}
    for s in scenarios:
        assert s.expected_tool is None or s.expected_tool in tool_names
