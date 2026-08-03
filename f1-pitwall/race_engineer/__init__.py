"""Race Engineer (phase 9): read-only on Strategy plan; emits driver radio."""

__all__ = ["RADIO_STYLE_GUIDE", "apply_radio", "get_radio_backend"]


def __getattr__(name: str):
    if name in {"RADIO_STYLE_GUIDE", "apply_radio", "get_radio_backend"}:
        from race_engineer import radio as _radio

        return getattr(_radio, name)
    raise AttributeError(name)
