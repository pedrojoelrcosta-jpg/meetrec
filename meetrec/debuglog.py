"""Observability for the processing pipeline.

The pipeline deliberately survives partial failures (a broken summary must
never lose a transcript), which makes silent bugs easy to miss. This module
makes every swallowed failure visible and queryable:

- each pipeline stage runs inside StageRecorder.stage(): outcome, duration
  and full traceback are persisted to <session>/debug.json as they happen
- meta.json gets an `issues` count so `meetrec list` can flag sessions
- `debug.strict: true` in config.yaml re-raises instead of swallowing, for
  interactive debugging (`meetrec reprocess` with strict on fails fast)
"""

import json
import logging
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

DEBUG_FILE = "debug.json"


class StageError(RuntimeError):
    """Wrapped stage failure re-raised in strict mode."""


class StageRecorder:
    def __init__(self, session_dir: Path, strict: bool = False):
        self.session_dir = session_dir
        self.strict = strict
        self.stages: dict[str, dict] = {}
        self.issues: list[dict] = []
        existing = session_dir / DEBUG_FILE
        if existing.exists():
            try:
                data = json.loads(existing.read_text(encoding="utf-8"))
                self.stages = data.get("stages", {})
                # issues restart per run; stages keep history of what ran
            except (OSError, json.JSONDecodeError):
                pass

    @contextmanager
    def stage(self, name: str, fatal: bool = False):
        """Run one pipeline stage. Non-fatal failures are recorded and
        swallowed (unless strict); fatal ones are recorded and re-raised."""
        log.debug("stage %s: started", name)
        started = time.time()
        try:
            yield
        except Exception as exc:
            seconds = round(time.time() - started, 1)
            issue = {
                "stage": name,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.issues.append(issue)
            # update, don't replace: note()/skip() calls made inside the
            # stage block must survive its exit
            self.stages.setdefault(name, {}).update(
                ok=False, seconds=seconds, error=issue["error"])
            self._flush()
            log.exception("stage %s FAILED after %.1fs", name, seconds)
            if fatal or self.strict:
                raise StageError(f"stage {name} failed: {exc}") from exc
        else:
            seconds = round(time.time() - started, 1)
            entry = self.stages.setdefault(name, {})
            entry.update(seconds=seconds)
            entry.setdefault("ok", True)  # a skip() inside the block wins
            self._flush()
            log.info("stage %s: ok (%.1fs)", name, seconds)

    def skip(self, name: str, reason: str) -> None:
        """Record a stage that did not run at all (e.g. missing input) —
        skipping is a legitimate outcome but should never be invisible."""
        self.stages[name] = {"ok": None, "skipped": reason}
        self._flush()
        log.info("stage %s: skipped (%s)", name, reason)

    def note(self, name: str, **details) -> None:
        """Attach extra structured detail to a recorded stage."""
        self.stages.setdefault(name, {}).update(details)
        self._flush()

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def _flush(self) -> None:
        try:
            (self.session_dir / DEBUG_FILE).write_text(
                json.dumps({"stages": self.stages, "issues": self.issues},
                           ensure_ascii=False, indent=2),
                encoding="utf-8")
        except OSError:
            log.warning("could not write %s", DEBUG_FILE)


def load_debug(session_dir: Path) -> dict | None:
    path = session_dir / DEBUG_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
