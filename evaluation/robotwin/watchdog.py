"""Fail fast with a Python/native stack location when a simulator call stalls."""
import faulthandler
import functools
import os

_active = False


def guarded_call(label, function, *args, **kwargs):
    global _active
    seconds = float(os.environ.get("NOISE_INVERSION_STEP_TIMEOUT", "0"))
    if seconds <= 0 or _active:
        return function(*args, **kwargs)
    # The faulthandler watchdog runs independently of the Python main thread;
    # exit=True also stops calls blocked in native planning/physics/rendering.
    print(f"[WATCHDOG] BEGIN {label}; deadline={seconds:g}s", flush=True)
    faulthandler.dump_traceback_later(seconds, exit=True)
    _active = True
    try:
        return function(*args, **kwargs)
    finally:
        faulthandler.cancel_dump_traceback_later()
        _active = False


def guard_method(instance, name):
    original = getattr(instance, name)
    if getattr(original, "_noise_watchdog", False):
        return
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        return guarded_call(name, original, *args, **kwargs)
    wrapped._noise_watchdog = True
    setattr(instance, name, wrapped)
