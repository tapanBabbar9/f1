"""Faithfulness: numeric claims in rationale should match tool returns."""

from __future__ import annotations

import re
from typing import Any, Sequence

from race_engineer.tools import ToolResult, flatten_numbers

# Numbers like 1.001, 17, 0.4231 — skip years-looking 4-digit if needed later.
_NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")


def extract_rationale_numbers(text: str) -> list[float]:
    vals: list[float] = []
    for m in _NUM_RE.finditer(text or ""):
        raw = m.group(0)
        # Skip bare integers that are likely ordinals in prose only if huge;
        # keep small ints (laps, positions).
        try:
            vals.append(float(raw))
        except ValueError:
            continue
    return vals


def _close(a: float, b: float, *, rel: float = 0.02, abs_tol: float = 0.05) -> bool:
    if abs(a - b) <= abs_tol:
        return True
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / scale <= rel


def faithfulness_score(
    rationale: str,
    tool_results: Sequence[ToolResult],
) -> dict[str, Any]:
    """
    Fraction of numeric tokens in rationale that match some tool-returned number.

    If the rationale cites no numbers, score is 1.0 when tools were used
    (vacuous pass) and 0.0 if tools were not used.
    """
    cited = extract_rationale_numbers(rationale)
    allowed = flatten_numbers([t.payload for t in tool_results])
    tools_used = len(tool_results) > 0

    if not cited:
        return {
            "faithfulness": 1.0 if tools_used else 0.0,
            "n_cited": 0,
            "n_matched": 0,
            "tools_used": tools_used,
            "n_tools": len(tool_results),
            "unmatched": [],
        }

    matched = 0
    unmatched: list[float] = []
    for x in cited:
        if any(_close(x, y) for y in allowed):
            matched += 1
        else:
            unmatched.append(x)

    return {
        "faithfulness": matched / len(cited),
        "n_cited": len(cited),
        "n_matched": matched,
        "tools_used": tools_used,
        "n_tools": len(tool_results),
        "unmatched": unmatched[:10],
    }
