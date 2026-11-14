"""Local interruption intent; detach must never become remote cancellation."""
from __future__ import annotations

import time
from threading import Event

from .agents import Cancelled


class Detached(RuntimeError):
    pass


class Interrupt:
    def __init__(self, cancel: Event, detach: Event | None = None):
        self.cancel = cancel
        self.detach = detach or Event()

    def is_set(self) -> bool:
        return self.cancel.is_set() or self.detach.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            if deadline is not None and time.monotonic() >= deadline:
                return False
            self.detach.wait(min(0.1, max(0, deadline - time.monotonic())) if deadline else 0.1)
        return True

    def raise_if_set(self) -> None:
        if self.detach.is_set():
            raise Detached("Detached. Cloud work is preserved; Resume reconnects to the saved session.")
        if self.cancel.is_set():
            raise Cancelled("Cancelled by user.")
