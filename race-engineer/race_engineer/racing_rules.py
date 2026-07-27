"""v0 sporting constraints (prompt + sim); full rules RAG is a later phase."""

from __future__ import annotations

from race_engineer.state import RaceState

# Appended to LLM system prompts (Phases 2–6).
DRY_MANDATORY_PIT_RULES = """
Dry-race sporting (v0):
- At least one pit stop is required before the chequered flag, using a different
  dry compound on the second stint (soft / medium / hard).
- If pit_stops so far is 0, do NOT plan to finish the race without pitting.
  Only pit_next_lap or stay_N_then_pit options are legal until the first stop.
- When recommending a pit stop, pick a different dry compound than the current set.
""".strip()


def is_dry_compound(compound: str | None) -> bool:
    c = (compound or "UNKNOWN").strip().upper()
    return c not in {"WET", "INTERMEDIATE"}


def mandatory_dry_pit_pending(state: RaceState) -> bool:
    """True when driver still owes the mandatory dry-race pit stop."""
    if state.pit_count > 0:
        return False
    return is_dry_compound(state.tyre_compound)


def alternate_dry_compound_id(current_compound_id: float) -> float:
    """Pick a different dry compound for the post-pit stint (sim default)."""
    if current_compound_id == 2.0:  # MEDIUM
        return 3.0  # HARD
    if current_compound_id == 3.0:  # HARD
        return 2.0  # MEDIUM
    if current_compound_id == 1.0:  # SOFT
        return 3.0  # HARD
    return 3.0
