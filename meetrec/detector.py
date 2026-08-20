"""Meeting detector state machine with debounce.

IDLE → CANDIDATE → ACTIVE → COOLDOWN → IDLE

Transition logic lives in tick(), pure over (clock, scan result), so it is
testable with a synthetic clock and scanner. The real polling loop is run().
"""

import time
from collections.abc import Callable, Iterable
from enum import Enum

from .registry_scan import MicUsage, scan_mic_usage


class State(Enum):
    IDLE = "IDLE"
    CANDIDATE = "CANDIDATE"
    ACTIVE = "ACTIVE"
    COOLDOWN = "COOLDOWN"


class MeetingDetector:
    def __init__(
        self,
        *,
        scan_fn: Callable[[], list[MicUsage]] = scan_mic_usage,
        clock: Callable[[], float] = time.monotonic,
        start_debounce_s: float = 15.0,
        stop_debounce_s: float = 30.0,
        ignore_apps: Iterable[str] = (),
        on_meeting_started: Callable[[set[str]], None] | None = None,
        on_meeting_ended: Callable[[float], None] | None = None,
    ) -> None:
        self._scan_fn = scan_fn
        self._clock = clock
        self._start_debounce_s = start_debounce_s
        self._stop_debounce_s = stop_debounce_s
        self._ignored = {a.lower() for a in ignore_apps}
        self._on_started = on_meeting_started
        self._on_ended = on_meeting_ended

        self.state = State.IDLE
        self.apps: set[str] = set()      # executables/app_ids seen this session
        self._capture_since = 0.0        # start of continuous capture
        self._cooldown_since = 0.0       # start of the no-capture period

    def _is_ignored(self, usage: MicUsage) -> bool:
        key = usage.app_id if usage.packaged else usage.exe_name
        return key.lower() in self._ignored

    def _relevant(self, usages: list[MicUsage]) -> set[str]:
        return {
            u.exe_path
            for u in usages
            if u.is_active and not self._is_ignored(u)
        }

    def tick(self, now: float, usages: list[MicUsage]) -> None:
        active = self._relevant(usages)

        if self.state is State.IDLE:
            if active:
                self.state = State.CANDIDATE
                self._capture_since = now
                self.apps = set(active)

        elif self.state is State.CANDIDATE:
            if not active:
                self.state = State.IDLE
                self.apps = set()
            else:
                self.apps |= active
                if now - self._capture_since >= self._start_debounce_s:
                    self.state = State.ACTIVE
                    if self._on_started:
                        self._on_started(set(self.apps))

        elif self.state is State.ACTIVE:
            if active:
                self.apps |= active
            else:
                self.state = State.COOLDOWN
                self._cooldown_since = now

        elif self.state is State.COOLDOWN:
            if active:
                self.state = State.ACTIVE
                self.apps |= active
            elif now - self._cooldown_since >= self._stop_debounce_s:
                duration = self._cooldown_since - self._capture_since
                self.state = State.IDLE
                self.apps = set()
                if self._on_ended:
                    self._on_ended(duration)

    def run(self, poll_interval: float = 2.0,
            should_stop: Callable[[], bool] | None = None) -> None:
        """Blocking polling loop. Ctrl+C (or should_stop) exits."""
        while not (should_stop and should_stop()):
            self.tick(self._clock(), self._scan_fn())
            time.sleep(poll_interval)
