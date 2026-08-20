"""Global quiet flag for evaluation runs (suppresses loader / training chatter)."""

_VERBOSE: bool = True


def set_verbose(on: bool = True) -> None:
    global _VERBOSE
    _VERBOSE = on


def is_verbose() -> bool:
    return _VERBOSE


def vprint(*args, **kwargs) -> None:
    if _VERBOSE:
        print(*args, **kwargs)
