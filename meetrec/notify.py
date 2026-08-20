"""Windows 11 toast notifications via windows-toasts (native WinRT).

The recording-start notification is mandatory and cannot be disabled — the
user must always know when recording is happening. Failures here must never
break the pipeline, so everything is wrapped.
"""

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_toaster = None


def _get_toaster():
    global _toaster
    if _toaster is None:
        from windows_toasts import InteractableWindowsToaster
        _toaster = InteractableWindowsToaster("meetrec")
    return _toaster


def _show(title: str, body: str, buttons: list[tuple[str, callable]] = ()):
    try:
        from windows_toasts import Toast, ToastActivatedEventArgs, ToastButton

        toast = Toast([title, body])
        actions = {}
        for label, action in buttons:
            toast.AddAction(ToastButton(label, label))
            actions[label] = action

        if actions:
            def on_activated(args: ToastActivatedEventArgs):
                action = actions.get(args.arguments)
                if action:
                    try:
                        action()
                    except Exception:
                        log.exception("toast action failed")
            toast.on_activated = on_activated

        _get_toaster().show_toast(toast)
    except Exception:
        log.exception("toast failed: %s", title)


def open_folder(path: Path) -> None:
    subprocess.Popen(["explorer", str(path)])


def recording_started(apps: set[str]) -> None:
    # mandatory, non-disableable by design
    names = ", ".join(sorted(a.rsplit("\\", 1)[-1] for a in apps)) or "?"
    _show("Recording meeting", f"Capturing audio ({names})")


def recording_stopped(duration_s: float) -> None:
    minutes = int(duration_s // 60)
    _show("Recording finished", f"{minutes} min captured. Processing…")


def processing_done(session_dir: Path, issues: int = 0) -> None:
    # an honest notification: partial failures are surfaced, not hidden
    body = (f"Result in {session_dir.name}" if not issues else
            f"{session_dir.name}: done with {issues} issue(s) — "
            f"run: meetrec debug {session_dir.name}")
    title = "Meeting processed" if not issues else "Meeting processed (issues)"
    _show(title, body,
          buttons=[("Open folder", lambda: open_folder(session_dir))])


def speakers_unlabeled(session_dir: Path, count: int) -> None:
    _show("Unidentified speakers",
          f"{count} unknown speaker(s). Run: meetrec label {session_dir.name}",
          buttons=[("Open folder", lambda: open_folder(session_dir))])


def error(message: str) -> None:
    _show("meetrec error", message[:200])
