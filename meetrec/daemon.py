"""Background daemon: detector → recorder → processing pipeline.

State machine: IDLE → RECORDING → PROCESSING → DELIVERING (processing and
delivery run on a worker thread so a new meeting can start recording while
the previous one is still processing).

Daemon state is exposed through a JSON file in the data dir so the CLI
(status/stop/pause) can talk to a daemon started elsewhere.
"""

import json
import logging
import os
import queue
import shutil
import threading
import time
from pathlib import Path

from . import notify, telegram
from .audio_capture import DualTrackRecorder
from .config import data_dir
from .detector import MeetingDetector
from .pipeline import new_session_dir, process_session

log = logging.getLogger(__name__)

STATE_FILE = "daemon_state.json"
PAUSE_FILE = "paused"
STOP_FILE = "stop_requested"


def state_path() -> Path:
    return data_dir() / STATE_FILE


def stop_path() -> Path:
    return data_dir() / STOP_FILE


def pause_path() -> Path:
    return data_dir() / PAUSE_FILE


def is_paused() -> bool:
    """True while the pause flag exists. A flag holding an ISO timestamp
    expires automatically (see `meetrec pause --for`)."""
    path = pause_path()
    if not path.exists():
        return False
    content = path.read_text(encoding="utf-8").strip()
    if content.startswith("until "):
        from datetime import datetime
        try:
            expiry = datetime.fromisoformat(content[len("until "):])
            if datetime.now() >= expiry:
                path.unlink(missing_ok=True)
                log.info("Pause expired — resuming")
                return False
        except ValueError:
            pass  # malformed flag counts as a plain manual pause
    return True


def find_unprocessed_sessions(cfg: dict) -> list[Path]:
    """Sessions with recorded audio but no transcript — e.g. after a crash
    mid-processing or mid-recording. Picked up again at daemon start."""
    out_dir = Path(cfg["output"]["dir"])
    if not out_dir.exists():
        return []
    pending = []
    for session in sorted(out_dir.iterdir()):
        if not session.is_dir():
            continue
        from .config import find_transcript
        has_audio = any((session / n).exists() for n in
                        ("track_mic.wav", "track_sys.wav", "audio.flac"))
        from .config import LEGACY_SUMMARY_MD, SUMMARY_MD
        has_summary = any((session / n).exists()
                          for n in (SUMMARY_MD, LEGACY_SUMMARY_MD))
        # no transcript: crashed before/at transcription. Transcript but no
        # summary: crashed between the merge and delivery stages — a window
        # the old check missed, silently losing the summary forever.
        if has_audio and (find_transcript(session) is None or not has_summary):
            pending.append(session)
    return pending


def read_state() -> dict | None:
    path = state_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class Daemon:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.recorder: DualTrackRecorder | None = None
        self._work: queue.Queue[Path] = queue.Queue()
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self.state = "IDLE"
        det = cfg["detector"]
        self.detector = MeetingDetector(
            start_debounce_s=det["start_debounce_s"],
            stop_debounce_s=det["stop_debounce_s"],
            ignore_apps=det["ignore_apps"],
            on_meeting_started=self._on_started,
            on_meeting_ended=self._on_ended,
        )

    # -- detector callbacks -------------------------------------------------

    def _on_started(self, apps: set[str]) -> None:
        if is_paused():
            log.info("Meeting detected but daemon is paused; not recording")
            return
        log.info("Meeting started: %s", apps)
        self.state = "RECORDING"
        self.recorder = DualTrackRecorder(new_session_dir(self.cfg))
        self.recorder.start()
        notify.recording_started(apps)  # mandatory, not disableable
        self._write_state()

    def _on_ended(self, duration: float) -> None:
        if self.recorder is None:
            return
        recorder, self.recorder = self.recorder, None
        stats = recorder.stop()
        log.info("Meeting ended: %s", stats)
        if self.cfg.get("notifications", {}).get("recording_stopped", True):
            notify.recording_stopped(stats["duration_s"])

        min_s = self.cfg["audio"]["min_session_s"]
        if stats["duration_s"] < min_s:
            log.info("Session shorter than %ss — discarding", min_s)
            shutil.rmtree(recorder.session_dir, ignore_errors=True)
            self.state = "IDLE"
            self._write_state()
            return

        meta = {"date": recorder.session_dir.name,
                "duration_s": stats["duration_s"], "tracks": stats}
        (recorder.session_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self._work.put(recorder.session_dir)
        self.state = "IDLE"
        self._write_state()

    # -- worker -------------------------------------------------------------

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                session_dir = self._work.get(timeout=1.0)
            except queue.Empty:
                continue
            self.state = "PROCESSING"
            self._write_state()
            try:
                process_session(self.cfg, session_dir)
            except Exception:
                log.exception("Processing %s failed", session_dir)
                notify.error(f"Processing failed: {session_dir.name}")
            self.state = "RECORDING" if self.recorder else "IDLE"
            self._write_state()

    # -- lifecycle ----------------------------------------------------------

    def _write_state(self) -> None:
        # written from the main loop AND the worker thread, and read by other
        # processes (`meetrec status`) — write to a temp file and atomically
        # replace so readers never see a torn write
        payload = json.dumps({
            "pid": os.getpid(),
            "state": self.state,
            "paused": pause_path().exists(),
            "recording_dir": str(self.recorder.session_dir)
            if self.recorder else None,
            "updated_at": time.time(),
        })
        with self._state_lock:
            tmp = state_path().with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(payload, encoding="utf-8")
            os.replace(tmp, state_path())

    def run(self) -> None:
        log.info("meetrec daemon started (pid %s)", os.getpid())
        stop_path().unlink(missing_ok=True)
        telegram.flush_queue()
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()
        for session in find_unprocessed_sessions(self.cfg):
            log.info("Recovering unprocessed session %s", session.name)
            self._work.put(session)
        self._write_state()
        try:
            last_beat = 0.0
            last_sweep = time.monotonic()
            while not stop_path().exists():
                self.detector.tick(time.monotonic(),
                                   self.detector._scan_fn())
                if time.monotonic() - last_beat > 30:
                    self._write_state()  # heartbeat for `meetrec status`
                    last_beat = time.monotonic()
                if time.monotonic() - last_sweep > 1800:  # every 30 min
                    from .pipeline import cleanup_expired_audio
                    try:
                        cleanup_expired_audio(self.cfg)
                    except Exception:
                        log.exception("audio retention sweep failed")
                    last_sweep = time.monotonic()
                time.sleep(self.cfg["detector"]["poll_interval_s"])
            log.info("Stop requested via `meetrec stop`")
        except KeyboardInterrupt:
            log.info("Interrupted")
        finally:
            if self.recorder:
                self._on_ended(0.0)
            self._stop.set()
            # the worker is a daemon thread: without this join, interpreter
            # shutdown kills it mid-stage and the summary/delivery of the
            # session being processed would be lost until the next start
            if worker.is_alive() and not self._work.empty() \
                    or self.state == "PROCESSING":
                log.info("Waiting for processing to finish (Ctrl+C again "
                         "to abandon; it resumes on next start)...")
            try:
                worker.join(timeout=1800)
            except KeyboardInterrupt:
                log.warning("Processing abandoned — it will resume on the "
                            "next daemon start")
            stop_path().unlink(missing_ok=True)
            state_path().unlink(missing_ok=True)
