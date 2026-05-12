"""Built-in 30-scenario benchmark.

Each scenario provides:
* ``prompt``: the user message
* ``tools``: list of available tools (Anthropic-formatted)
* ``expected_tool``: name of the tool the model SHOULD pick (None = no tool needed)
* ``expected_arg_types``: type constraints on the arguments (e.g. ``{"city": str}``)
* ``required_args``: arguments that MUST appear with reasonable values
* ``arg_value_checks``: optional per-arg predicates for stricter scoring
"""

from __future__ import annotations

from tool_use_eval.runner import Scenario


# Shared tool definitions used across scenarios.
WEATHER_TOOL = {
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["city"],
    },
}

CALC_TOOL = {
    "name": "calculate",
    "description": "Evaluate a math expression and return its numeric result.",
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression like '12 * 34 + 5'"},
        },
        "required": ["expression"],
    },
}

SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web and return top results.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
}

CREATE_TICKET_TOOL = {
    "name": "create_ticket",
    "description": "Open a support ticket in the help-desk system.",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "body": {"type": "string"},
            "priority": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
        },
        "required": ["title", "body", "priority"],
    },
}

SEND_EMAIL_TOOL = {
    "name": "send_email",
    "description": "Send an email to a recipient.",
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {"type": "string", "description": "Recipient email"},
            "subject": {"type": "string"},
            "body": {"type": "string"},
        },
        "required": ["to", "subject", "body"],
    },
}

ALL_TOOLS = [WEATHER_TOOL, CALC_TOOL, SEARCH_TOOL, CREATE_TICKET_TOOL, SEND_EMAIL_TOOL]


_SCENARIOS = [
    # Pick correct tool (12).
    {"id": "weather-paris", "prompt": "What's the weather in Paris in Celsius?",
     "expected_tool": "get_weather",
     "required_args": ["city"], "arg_value_checks": {"city": lambda v: "paris" in v.lower()}},
    {"id": "weather-tokyo", "prompt": "Tell me Tokyo weather in fahrenheit please.",
     "expected_tool": "get_weather", "required_args": ["city", "units"],
     "arg_value_checks": {"units": lambda v: v.lower() == "fahrenheit"}},
    {"id": "weather-no-units", "prompt": "Weather for Berlin?",
     "expected_tool": "get_weather", "required_args": ["city"]},
    {"id": "calc-basic", "prompt": "Compute 1234 * 5678.",
     "expected_tool": "calculate", "required_args": ["expression"],
     "arg_value_checks": {"expression": lambda v: "1234" in v and "5678" in v}},
    {"id": "calc-mixed", "prompt": "What is (1024 + 256) / 2?",
     "expected_tool": "calculate", "required_args": ["expression"]},
    {"id": "search-news", "prompt": "Search the web for the latest news about quantum computing.",
     "expected_tool": "web_search", "required_args": ["query"]},
    {"id": "search-with-limit", "prompt": "Search for 'rust programming tutorials' and give me 5 results.",
     "expected_tool": "web_search", "required_args": ["query", "limit"],
     "arg_value_checks": {"limit": lambda v: int(v) == 5}},
    {"id": "ticket-printer", "prompt": "Please open a HIGH priority support ticket for printer queue stuck.",
     "expected_tool": "create_ticket", "required_args": ["title", "body", "priority"],
     "arg_value_checks": {"priority": lambda v: v.lower() == "high"}},
    {"id": "ticket-low", "prompt": "Open a low priority ticket: keyboard sticky keys, will replace tomorrow.",
     "expected_tool": "create_ticket", "required_args": ["title", "body", "priority"],
     "arg_value_checks": {"priority": lambda v: v.lower() == "low"}},
    {"id": "ticket-critical-outage", "prompt": "URGENT: open a CRITICAL ticket — production DB is down.",
     "expected_tool": "create_ticket", "required_args": ["title", "body", "priority"],
     "arg_value_checks": {"priority": lambda v: v.lower() == "critical"}},
    {"id": "email-followup", "prompt": "Send an email to bob@example.com with subject 'Follow up' and body 'Are we still on for tomorrow?'",
     "expected_tool": "send_email", "required_args": ["to", "subject", "body"],
     "arg_value_checks": {"to": lambda v: v.lower() == "bob@example.com"}},
    {"id": "email-thank-you", "prompt": "Email alice@example.com a thank you note for the demo yesterday.",
     "expected_tool": "send_email", "required_args": ["to", "subject", "body"]},

    # Should NOT use a tool (8).
    {"id": "joke", "prompt": "Tell me a joke about cats.", "expected_tool": None},
    {"id": "definition", "prompt": "What does 'photosynthesis' mean in one sentence?", "expected_tool": None},
    {"id": "preference", "prompt": "Recommend a thriller movie I might enjoy.", "expected_tool": None},
    {"id": "small-talk", "prompt": "How are you today?", "expected_tool": None},
    {"id": "math-no-need", "prompt": "What is 2 + 2?", "expected_tool": None},
    {"id": "general-knowledge", "prompt": "Who was the first president of the United States?", "expected_tool": None},
    {"id": "creative", "prompt": "Write a haiku about the ocean.", "expected_tool": None},
    {"id": "explain", "prompt": "Briefly explain what a black hole is.", "expected_tool": None},

    # Should pick a tool not in the list (4) — testing tool-name hallucination.
    {"id": "unsupported-translation", "prompt": "Translate 'good morning' to French.", "expected_tool": None},
    {"id": "unsupported-image", "prompt": "Generate an image of a cat in space.", "expected_tool": None},
    {"id": "unsupported-shell", "prompt": "Run 'ls -la' on the home directory.", "expected_tool": None},
    {"id": "unsupported-stock", "prompt": "What's the stock price of AAPL right now?", "expected_tool": None},

    # Argument-type stress (3) — the model should still pick the right tool with right types.
    {"id": "calc-as-words", "prompt": "Compute one thousand two hundred plus seventeen.",
     "expected_tool": "calculate", "required_args": ["expression"]},
    {"id": "weather-with-typo", "prompt": "Weather for Parris in celsius",
     "expected_tool": "get_weather", "required_args": ["city", "units"]},
    {"id": "ticket-no-explicit-priority", "prompt": "Open a ticket: the wifi is flaky in conference room B.",
     "expected_tool": "create_ticket", "required_args": ["title", "body", "priority"]},

    # Multi-step (3) — first call must be one specific tool; expected_tool is the FIRST.
    {"id": "multi-search-then-email", "prompt": "Look up the current CEO of OpenAI and email it to bob@example.com.",
     "expected_tool": "web_search", "required_args": ["query"]},
    {"id": "multi-calc-then-ticket", "prompt": "Compute 84 * 3 and open a low priority ticket with the result as the title.",
     "expected_tool": "calculate", "required_args": ["expression"]},
    {"id": "multi-weather-then-search", "prompt": "Check the weather in Madrid then search for outdoor cafés that are good in that weather.",
     "expected_tool": "get_weather", "required_args": ["city"]},
]


def builtin_scenarios() -> list[Scenario]:
    out: list[Scenario] = []
    for s in _SCENARIOS:
        out.append(
            Scenario(
                id=s["id"],
                prompt=s["prompt"],
                tools=ALL_TOOLS,
                expected_tool=s.get("expected_tool"),
                required_args=s.get("required_args", []),
                arg_value_checks=s.get("arg_value_checks", {}),
            )
        )
    return out
