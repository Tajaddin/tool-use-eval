"""Scenario, scoring, and evaluation runner.

Six named failure modes:

* ``NO_TOOL_WHEN_EXPECTED`` — model replied in text when a tool was the right answer
* ``TOOL_WHEN_NOT_EXPECTED`` — model invoked a tool on a chit-chat prompt
* ``WRONG_TOOL_NAME`` — model picked the wrong tool from the available list (or named a tool not in the list)
* ``MISSING_ARG`` — model invoked the correct tool but omitted a required argument
* ``WRONG_ARG_VALUE`` — required arg present but failed the scenario's value check
* ``WRONG_ARG_TYPE`` — arg present but not the type the tool's schema requires
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from tool_use_eval.backends import ModelResponse, ToolCall, ToolUseBackend


class FailureMode(str, Enum):
    NO_TOOL_WHEN_EXPECTED = "no_tool_when_expected"
    TOOL_WHEN_NOT_EXPECTED = "tool_when_not_expected"
    WRONG_TOOL_NAME = "wrong_tool_name"
    MISSING_ARG = "missing_arg"
    WRONG_ARG_VALUE = "wrong_arg_value"
    WRONG_ARG_TYPE = "wrong_arg_type"


@dataclass
class Scenario:
    id: str
    prompt: str
    tools: list[dict]
    expected_tool: str | None
    required_args: list[str] = field(default_factory=list)
    arg_value_checks: dict[str, Callable[[object], bool]] = field(default_factory=dict)


@dataclass
class Scorecard:
    """Per-scenario score breakdown."""

    scenario_id: str
    passed: bool
    failures: list[FailureMode] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _type_of_schema(schema: dict, arg_name: str) -> str | None:
    props = (schema or {}).get("properties", {})
    spec = props.get(arg_name) or {}
    return spec.get("type")


def _matches_schema_type(value, jsonschema_type: str | None) -> bool:
    if jsonschema_type is None:
        return True
    if jsonschema_type == "string":
        return isinstance(value, str)
    if jsonschema_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if jsonschema_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if jsonschema_type == "boolean":
        return isinstance(value, bool)
    if jsonschema_type == "object":
        return isinstance(value, dict)
    if jsonschema_type == "array":
        return isinstance(value, list)
    return True


def score_call(scenario: Scenario, response: ModelResponse) -> Scorecard:
    """Score every tool call in the response; it passes only if all of them are right."""
    failures: list[FailureMode] = []
    notes: list[str] = []

    expected = scenario.expected_tool

    if expected is None:
        if response.used_tool:
            failures.append(FailureMode.TOOL_WHEN_NOT_EXPECTED)
            names = ", ".join(repr(c.name) for c in response.tool_calls)
            notes.append(f"called {names} when none was expected")
        return Scorecard(scenario_id=scenario.id, passed=not failures, failures=failures, notes=notes)

    # expected_tool is not None — we want exactly that tool.
    if not response.tool_calls:
        failures.append(FailureMode.NO_TOOL_WHEN_EXPECTED)
        notes.append("replied in text when a tool call was required")
        return Scorecard(scenario_id=scenario.id, passed=False, failures=failures, notes=notes)

    tool_schema = next((t["input_schema"] for t in scenario.tools if t["name"] == expected), {})

    for call in response.tool_calls:
        if call.name != expected:
            failures.append(FailureMode.WRONG_TOOL_NAME)
            notes.append(f"called {call.name!r}, expected {expected!r}")
            # Wrong tool means the call is wrong regardless of its args.
            continue

        args = call.arguments or {}
        for req in scenario.required_args:
            if req not in args:
                failures.append(FailureMode.MISSING_ARG)
                notes.append(f"missing arg {req!r}")
                continue
            # Type check.
            expected_type = _type_of_schema(tool_schema, req)
            if not _matches_schema_type(args[req], expected_type):
                failures.append(FailureMode.WRONG_ARG_TYPE)
                notes.append(
                    f"arg {req!r} has wrong type: got {type(args[req]).__name__}, "
                    f"schema wants {expected_type}"
                )

        for arg_name, check in scenario.arg_value_checks.items():
            if arg_name in args:
                try:
                    ok = bool(check(args[arg_name]))
                except Exception as exc:  # noqa: BLE001
                    ok = False
                    notes.append(f"value check for {arg_name!r} raised {type(exc).__name__}")
                if not ok:
                    failures.append(FailureMode.WRONG_ARG_VALUE)
                    notes.append(f"arg {arg_name!r}={args[arg_name]!r} failed value check")

    passed = not failures
    return Scorecard(scenario_id=scenario.id, passed=passed, failures=failures, notes=notes)


@dataclass
class EvalResult:
    scenario_id: str
    backend_model: str
    used_tool: bool
    chosen_tool: str | None
    arguments: dict
    text_excerpt: str
    scorecard: Scorecard
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class EvalSummary:
    backend_model: str
    n: int
    pass_count: int
    pass_rate: float
    failure_counts: dict[str, int]
    results: list[EvalResult]


def evaluate(backend: ToolUseBackend, scenarios: list[Scenario], *, system: str | None = None) -> EvalSummary:
    results: list[EvalResult] = []
    failure_counts: dict[str, int] = {f.value: 0 for f in FailureMode}
    n_pass = 0
    for sc in scenarios:
        resp = backend.call(sc.prompt, sc.tools, system=system)
        sc_card = score_call(sc, resp)
        # Counts are scenarios per mode; the scorecard keeps one entry per failing call.
        for f in set(sc_card.failures):
            failure_counts[f.value] += 1
        if sc_card.passed:
            n_pass += 1
        results.append(
            EvalResult(
                scenario_id=sc.id,
                backend_model=backend.model,
                used_tool=resp.used_tool,
                chosen_tool=resp.tool_call.name if resp.tool_call else None,
                arguments=resp.tool_call.arguments if resp.tool_call else {},
                text_excerpt=(resp.text or "")[:140],
                scorecard=sc_card,
                tool_calls=list(resp.tool_calls),
            )
        )
    return EvalSummary(
        backend_model=backend.model,
        n=len(scenarios),
        pass_count=n_pass,
        pass_rate=round(n_pass / max(len(scenarios), 1), 4),
        failure_counts=failure_counts,
        results=results,
    )
