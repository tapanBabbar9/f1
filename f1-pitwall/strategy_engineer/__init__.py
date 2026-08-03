"""Strategy engineer (phases 1–7): owns the race plan and memory."""

__all__ = ["get_strategy_backend"]


def __getattr__(name: str):
    if name == "get_strategy_backend":
        from strategy_engineer.strategy import get_strategy_backend

        return get_strategy_backend
    raise AttributeError(name)
