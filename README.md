# tool-use-eval

> Function-calling benchmark for LLMs with **six named failure modes**. Real Claude Haiku 4.5 run on 30 hand-curated scenarios: **93.3% pass rate**, both failures of type `tool_when_not_expected` (used a tool where a text answer was fine).

[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) [![Tests](https://img.shields.io/badge/tests-10%20passing-brightgreen)](#tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

## Hero benchmark

`bench/run_anthropic.py` against `claude-haiku-4-5-20251001`:

| Metric | Value |
|---|---:|
| Scenarios | 30 |
| Pass count | 28 |
| **Pass rate** | **93.3%** |
| Cost (one full run) | ~$0.02 |

Failure breakdown:

| Failure mode | Count | What it means |
|---|---:|---|
| `no_tool_when_expected` | 0 | Model replied in text where a tool was required |
| `tool_when_not_expected` | **2** | Model called a tool on a chit-chat / general-knowledge prompt |
| `wrong_tool_name` | 0 | Picked the wrong tool from the available list |
| `missing_arg` | 0 | Invoked the correct tool but omitted a required argument |
| `wrong_arg_value` | 0 | Required arg present but failed the scenario's value check |
| `wrong_arg_type` | 0 | Arg present but not the JSON-Schema type the tool requires |

The two failures are both judgment-call cases:

1. `preference`: "Recommend a thriller movie I might enjoy." → Haiku called `web_search(query="best thriller movies to watch 2024", limit=5)`. Arguably reasonable behavior, scored as a fail because the scenario marked `expected_tool=None` (assume the model has training data on movies).
2. `math-no-need`: "What is 2 + 2?" → Haiku called `calculate(expression="2 + 2")`. Same pattern.

Raw output: [`bench/anthropic_results.json`](bench/anthropic_results.json). Reproduce with `python bench/run_anthropic.py`.

## Why this exists

"Tool-use accuracy" appears in every model release post as a single percentage. That's not enough for an engineering decision. Knowing *which* failure modes a model is biased toward — does it hallucinate tools, misuse argument types, miss multi-step planning, or just over-eagerly tool-call on small talk? — changes how you scaffold the rest of your system.

`tool-use-eval` is a tiny harness that:

* Ships a **30-scenario built-in dataset** with five common tool families (weather, calculate, web search, ticket creation, send email).
* Scores each model response against **six named failure modes**, not just a pass/fail.
* Provides **pluggable backends** (Mock / Anthropic) and a swap-in protocol for OpenAI / Groq.

The result is one comparable table per model with the same denominator and the same failure taxonomy.

## Quickstart

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
python bench/run_anthropic.py
```

Programmatic:

```python
from tool_use_eval import AnthropicBackend, builtin_scenarios, evaluate

backend = AnthropicBackend(model="claude-haiku-4-5-20251001")
summary = evaluate(backend, builtin_scenarios())

print(summary.pass_rate)             # 0.9333
print(summary.failure_counts)        # {'tool_when_not_expected': 2, ...}
for r in summary.results:
    if not r.scorecard.passed:
        print(r.scenario_id, r.scorecard.failures, r.scorecard.notes)
```

## Bring-your-own scenarios

```python
from tool_use_eval import Scenario, evaluate

my_scenarios = [
    Scenario(
        id="open-bug-ticket",
        prompt="Open a ticket for issue: the login page crashes on Safari 17.",
        tools=[my_ticket_tool_def],
        expected_tool="create_ticket",
        required_args=["title", "body", "priority"],
        arg_value_checks={
            "priority": lambda v: v.lower() in {"medium", "high"},
        },
    ),
    # ...
]
summary = evaluate(backend, my_scenarios)
```

The `Scenario` model is intentionally minimal — drop in any of your own tools.

## Six named failure modes

| Mode | Triggered when… |
|---|---|
| `NO_TOOL_WHEN_EXPECTED` | scenario had `expected_tool`, model replied with text only |
| `TOOL_WHEN_NOT_EXPECTED` | scenario had `expected_tool=None`, model called a tool anyway |
| `WRONG_TOOL_NAME` | model picked a different tool than `expected_tool` (or a tool name not in the list) |
| `MISSING_ARG` | correct tool, but a `required_args` entry was missing |
| `WRONG_ARG_TYPE` | required arg present but its Python type didn't match the tool's JSON-Schema `type` |
| `WRONG_ARG_VALUE` | required arg present but a `arg_value_checks[name](value)` returned False |

A single response can hit multiple modes (e.g., correct tool but missing arg + wrong type on another arg). The scorecard lists them all.

## Built-in scenario coverage

The 30 built-in scenarios are split:

| Category | Count |
|---|---:|
| Pick the correct tool | 12 |
| Should NOT use any tool (chit-chat / general knowledge) | 8 |
| User asked for a tool *not in the list* (tool-name hallucination tempt) | 4 |
| Argument-type stress (numbers as words, typos in city name) | 3 |
| Multi-step (first call must be the *first* tool in a chain) | 3 |

## Tests

```bash
pip install -e ".[dev]"
pytest -q
```

```
10 passed in 0.06s
```

Each failure mode has a dedicated test that produces it via `MockBackend`. An end-to-end test runs the first 5 built-in scenarios through a deterministic mock.

## Project layout

```
.
├── src/tool_use_eval/
│   ├── __init__.py
│   ├── runner.py      # Scenario, FailureMode, score_call, evaluate, EvalSummary
│   ├── backends.py    # ToolUseBackend protocol + Mock + Anthropic adapter
│   └── dataset.py     # built-in 30 scenarios + 5 tool definitions
├── tests/             # 10 pytest cases, one per failure mode + end-to-end
└── bench/
    ├── run_anthropic.py     # real Haiku 4.5 run
    └── anthropic_results.json
```

## Limitations

**Single model in published results.** Only Haiku 4.5 in the committed `anthropic_results.json`. Adding a Groq Llama 3.1-8B or 3.3-70B comparison is straightforward (the protocol is the same; OpenAI / Groq tool-call shapes differ but the parsing is mechanical) and was deferred for v0.1.

**No multi-turn tool execution.** The harness scores the *first* tool call only. For multi-step scenarios in the built-in set, we test that the first call is correct — verifying the full chain would require actually executing the tools and feeding results back, which is a different scope.

**Scenario set is hand-curated, not crowd-labeled.** 30 scenarios is enough to surface big differences between models but not enough for confidence intervals. For peer-reviewed claims, expand to several hundred scenarios with multiple annotators.

**Argument-type checking is JSON-Schema-flat.** Nested object / array constraints aren't enforced beyond top-level types. Most function-calling failures happen at the top level so this is usually fine, but deeply-nested schemas need their own validator.

## License

MIT — see [LICENSE](LICENSE).
