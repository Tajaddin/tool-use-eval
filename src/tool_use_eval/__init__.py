"""LLM function-calling benchmark with named failure-mode breakdown."""

from tool_use_eval.backends import (
    AnthropicBackend,
    MockBackend,
    ToolCall,
    ToolUseBackend,
)
from tool_use_eval.dataset import builtin_scenarios
from tool_use_eval.runner import (
    EvalResult,
    EvalSummary,
    FailureMode,
    Scenario,
    Scorecard,
    evaluate,
    score_call,
)

__version__ = "0.1.0"

__all__ = [
    "Scenario",
    "FailureMode",
    "EvalResult",
    "EvalSummary",
    "Scorecard",
    "score_call",
    "evaluate",
    "ToolUseBackend",
    "MockBackend",
    "AnthropicBackend",
    "ToolCall",
    "builtin_scenarios",
]
