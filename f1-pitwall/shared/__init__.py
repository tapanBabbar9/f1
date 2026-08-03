"""Shared world model, harness, and A2A pipeline."""

__all__ = [
    "RaceReplay",
    "RaceState",
]


def __getattr__(name: str):
    if name == "RaceReplay":
        from shared.replay import RaceReplay

        return RaceReplay
    if name == "RaceState":
        from shared.state import RaceState

        return RaceState
    raise AttributeError(name)
