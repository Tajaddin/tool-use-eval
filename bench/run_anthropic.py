"""Real Claude Haiku 4.5 evaluation over the built-in 30 scenarios."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_use_eval import AnthropicBackend, builtin_scenarios, evaluate

BENCH = Path(__file__).resolve().parent
RESULTS = BENCH / "anthropic_results.json"


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 2

    backend = AnthropicBackend(model="claude-haiku-4-5-20251001")
    scenarios = builtin_scenarios()
    print(f"Running {len(scenarios)} scenarios against {backend.model}...")
    summary = evaluate(backend, scenarios)

    out = {
        "backend_model": summary.backend_model,
        "n": summary.n,
        "pass_count": summary.pass_count,
        "pass_rate": summary.pass_rate,
        "failure_counts": summary.failure_counts,
        "results": [
            {
                "scenario_id": r.scenario_id,
                "used_tool": r.used_tool,
                "chosen_tool": r.chosen_tool,
                "arguments": r.arguments,
                "text_excerpt": r.text_excerpt,
                "passed": r.scorecard.passed,
                "failures": [f.value for f in r.scorecard.failures],
                "notes": r.scorecard.notes,
            }
            for r in summary.results
        ],
    }
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
    print(f"\nWrote {RESULTS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
