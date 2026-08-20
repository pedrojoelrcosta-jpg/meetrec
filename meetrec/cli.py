"""meetrec CLI: start, stop, status, pause, reprocess, label, speakers,
doctor, test-telegram, autostart."""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import data_dir, env, load_config


def _session_dir(cfg: dict, name: str) -> Path:
    path = Path(name)
    if not path.is_absolute():
        path = Path(cfg["output"]["dir"]) / name
    if not path.is_dir():
        sys.exit(f"Session directory not found: {path}")
    return path


# -- commands ----------------------------------------------------------------

def _setup_logging(level: int = logging.INFO) -> None:
    from logging.handlers import RotatingFileHandler
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(level)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    file_handler = RotatingFileHandler(
        data_dir() / "meetrec.log", maxBytes=2_000_000, backupCount=3,
        encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def _log_level(cfg: dict, args) -> int:
    if getattr(args, "debug", False) \
            or cfg.get("debug", {}).get("level") == "debug":
        return logging.DEBUG
    return logging.INFO


def cmd_setup(cfg: dict, args) -> None:
    from .setup_wizard import run_wizard
    run_wizard(cfg)


def cmd_start(cfg: dict, args) -> None:
    from .daemon import Daemon, read_state
    state = read_state()
    if state and _pid_alive(state["pid"]):
        sys.exit(f"Daemon already running (pid {state['pid']})")
    _setup_logging(_log_level(cfg, args))
    Daemon(cfg).run()


def cmd_debug(cfg: dict, args) -> None:
    """X-ray of one session: artifacts, stage outcomes, recorded issues."""
    from .debuglog import load_debug
    session_dir = _session_dir(cfg, args.session)
    print(f"Session {session_dir}\n")

    print("Artifacts:")
    for name in ("track_mic.wav", "track_sys.wav", "audio.flac",
                 "transcript_mic.json", "transcript_sys.json",
                 "diarization.json", "embeddings.npz", "speaker_map.json",
                 "transcript.txt", "transcript.md", "summary.md",
                 "meta.json", "debug.json"):
        path = session_dir / name
        size = f"{path.stat().st_size:,} B" if path.exists() else "missing"
        print(f"  {'[x]' if path.exists() else '[ ]'} {name:<22} {size}")

    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tracks = meta.get("tracks", {})
        for side in ("mic", "sys"):
            err = (tracks.get(side) or {}).get("error")
            if err:
                print(f"\nCapture error on {side} track: {err}")

    data = load_debug(session_dir)
    if not data:
        print("\nNo debug.json — session was processed before debug "
              "recording existed. Re-run: meetrec reprocess "
              f"{session_dir.name}")
        return
    print("\nStages:")
    for name, info in data.get("stages", {}).items():
        if info.get("ok") is None:
            status = f"skipped ({info.get('skipped', '?')})"
        elif info["ok"]:
            status = f"ok in {info.get('seconds', '?')}s"
        else:
            status = f"FAILED after {info.get('seconds', '?')}s"
        extra = {k: v for k, v in info.items()
                 if k not in ("ok", "seconds", "skipped", "error")}
        print(f"  {name:<20} {status}"
              + (f"  {extra}" if extra else ""))
        if info.get("error"):
            print(f"      {info['error']}")
    issues = data.get("issues", [])
    if issues:
        print(f"\n{len(issues)} issue(s) recorded:")
        for issue in issues:
            print(f"  [{issue['at']}] {issue['stage']}: {issue['error']}")
        if args.traceback:
            for issue in issues:
                print(f"\n--- {issue['stage']} traceback ---")
                print(issue["traceback"])
        else:
            print("  (run with --traceback for full tracebacks)")
    else:
        print("\nNo issues recorded.")


def cmd_record(cfg: dict, args) -> None:
    """Manual recording, independent of meeting detection — e.g. in-person
    meetings. Enter stops and processing starts right away."""
    from . import notify
    from .audio_capture import DualTrackRecorder
    from .pipeline import new_session_dir, process_session
    _setup_logging()
    in_person = getattr(args, "in_person", False)
    recorder = DualTrackRecorder(new_session_dir(cfg))
    recorder.start()
    notify.recording_started({"manual recording"})
    print(f"Recording to {recorder.session_dir}"
          + (" (in-person mode: the mic track will be diarized)"
             if in_person else ""))

    time.sleep(1.5)  # let the capture threads open their devices
    mic_dead = recorder.mic.error is not None
    sys_dead = recorder.sys.error is not None
    if mic_dead and sys_dead:
        recorder.stop()
        import shutil
        shutil.rmtree(recorder.session_dir, ignore_errors=True)
        sys.exit("Both audio devices failed to open — nothing to record.\n"
                 f"  mic: {recorder.mic.error}\n  sys: {recorder.sys.error}\n"
                 "Run `meetrec doctor` to check the audio setup.")
    if mic_dead:
        print(f"! Microphone failed ({recorder.mic.error}) — recording "
              "system audio only.")
    if sys_dead:
        print(f"! System-audio loopback failed ({recorder.sys.error}) — "
              "recording the microphone only"
              + (" (fine for in-person meetings)." if in_person else "."))

    print("Press Enter to stop...", flush=True)
    _wait_for_enter()
    print("Stopping...", flush=True)
    stats = recorder.stop()
    if stats["duration_s"] < 3:
        import shutil
        shutil.rmtree(recorder.session_dir, ignore_errors=True)
        sys.exit(f"Stopped after {stats['duration_s']}s — treating as "
                 "accidental; session discarded.")
    print(f"Recorded {stats['duration_s']}s. Processing...")
    import json as _json
    (recorder.session_dir / "meta.json").write_text(
        _json.dumps({"date": recorder.session_dir.name,
                     "duration_s": stats["duration_s"], "tracks": stats,
                     "manual": True, "in_person": in_person},
                    ensure_ascii=False, indent=2),
        encoding="utf-8")
    process_session(cfg, recorder.session_dir)
    _print_outputs(recorder.session_dir)


def cmd_list(cfg: dict, args) -> None:
    out_dir = Path(cfg["output"]["dir"])
    if not out_dir.exists():
        print(f"No sessions yet ({out_dir} does not exist).")
        return
    sessions = sorted(d for d in out_dir.iterdir() if d.is_dir())
    if not sessions:
        print("No sessions yet.")
        return
    for session in sessions:
        meta_path = session / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            duration = int(meta.get("duration_s") or 0)
            from .config import (LEGACY_SUMMARY_MD, SUMMARY_MD,
                                 find_transcript)
            has_summary = any((session / n).exists()
                              for n in (SUMMARY_MD, LEGACY_SUMMARY_MD))
            state = "ok" if has_summary else (
                "no summary" if find_transcript(session) else "unprocessed")
            issues = meta.get("issues", 0)
            if issues:
                state += f", {issues} issue(s)"
            speakers = ", ".join(meta.get("speakers", [])) or "-"
            print(f"  {session.name}  {duration // 60:3d}min  "
                  f"lang={meta.get('language', '?')}  [{state}]  {speakers}")
        else:
            print(f"  {session.name}  (no metadata — unprocessed)")


def cmd_summary(cfg: dict, args) -> None:
    """Regenerate (and optionally resend) just the summary of a session."""
    from . import telegram
    from .config import SUMMARY_MD, find_transcript
    from .summarize import summarize
    session_dir = _session_dir(cfg, args.session)
    transcript = find_transcript(session_dir)
    if transcript is None:
        sys.exit("Session has no transcript yet — run reprocess first.")
    if args.backend:
        cfg["summary"]["backend"] = args.backend
    meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
    lang_cfg = args.language or cfg["summary"].get("language", "auto")
    language = meta.get("language", "en") if lang_cfg == "auto" else lang_cfg
    text = summarize(cfg, language, transcript.read_text(encoding="utf-8"))
    (session_dir / SUMMARY_MD).write_text(text, encoding="utf-8")
    print(f"Summary written ({cfg['summary']['backend']}, {language}).")
    if args.resend:
        telegram.deliver(cfg, session_dir, text,
                         f"Meeting {session_dir.name}")
        print("Sent to Telegram.")


def cmd_stop(cfg: dict, args) -> None:
    from .daemon import read_state, state_path, stop_path
    state = read_state()
    if not state or not _pid_alive(state["pid"]):
        state_path().unlink(missing_ok=True)
        stop_path().unlink(missing_ok=True)
        print("Daemon is not running.")
        return
    if getattr(args, "force", False):
        # taskkill /F gives the daemon no chance to run cleanup — an
        # in-flight processing job is resumed on the next start
        subprocess.run(["taskkill", "/PID", str(state["pid"]), "/F"],
                       check=False)
        state_path().unlink(missing_ok=True)
        print(f"Daemon (pid {state['pid']}) killed.")
        return
    stop_path().write_text(datetime.now().isoformat(), encoding="utf-8")
    print("Stop requested — waiting for the daemon to exit cleanly", end="")
    for _ in range(15):
        time.sleep(1)
        print(".", end="", flush=True)
        if not _pid_alive(state["pid"]):
            print("\nDaemon stopped.")
            return
    print("\nDaemon is still finishing (likely processing a meeting). It "
          "will exit when done; use `meetrec stop --force` to kill it now.")


def cmd_status(cfg: dict, args) -> None:
    from .daemon import is_paused, read_state
    state = read_state()
    if not state or not _pid_alive(state["pid"]):
        print("Daemon: not running")
    else:
        age = time.time() - state["updated_at"]
        print(f"Daemon: running (pid {state['pid']}), state={state['state']}"
              f"{' [PAUSED]' if state['paused'] else ''}, "
              f"heartbeat {age:.0f}s ago")
        if state.get("recording_dir"):
            print(f"Recording to: {state['recording_dir']}")
    print(f"Paused: {'yes' if is_paused() else 'no'}")
    print(f"Output dir: {cfg['output']['dir']}")
    log_path = data_dir() / "meetrec.log"
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8",
                                   errors="replace").splitlines()[-5:]
        if lines:
            print("Recent log:")
            for line in lines:
                print(f"  {line}")


def _parse_duration(text: str) -> float:
    """'90m', '2h', '1h30m' → seconds."""
    import re
    matches = re.findall(r"(\d+)\s*([hm])", text.lower())
    if not matches or re.sub(r"[\dhm\s]", "", text.lower()):
        raise ValueError(f"Invalid duration: {text!r} (use e.g. 30m, 2h, 1h30m)")
    return sum(int(n) * (3600 if unit == "h" else 60) for n, unit in matches)


def cmd_pause(cfg: dict, args) -> None:
    from datetime import timedelta
    from .daemon import pause_path
    if args.duration:
        seconds = _parse_duration(args.duration)
        expiry = datetime.now() + timedelta(seconds=seconds)
        pause_path().write_text(f"until {expiry.isoformat()}", encoding="utf-8")
        print(f"Paused until {expiry:%H:%M} — then recording resumes "
              "automatically.")
    elif pause_path().exists():
        pause_path().unlink()
        print("Resumed — new meetings will be recorded.")
    else:
        pause_path().write_text("manual", encoding="utf-8")
        print("Paused — new meetings will NOT be recorded "
              "(run `meetrec pause` again to resume).")


def cmd_reprocess(cfg: dict, args) -> None:
    from .pipeline import process_session
    session_dir = _session_dir(cfg, args.session)
    if args.full:
        # drop intermediates so every stage runs again
        for name in ("transcript_mic.json", "transcript_sys.json",
                     "diarization.json", "speaker_map.json",
                     "embeddings.npz", "summary.md", "resumo.md",
                     "debug.json"):
            (session_dir / name).unlink(missing_ok=True)
    _setup_logging(_log_level(cfg, args))
    process_session(cfg, session_dir, with_notifications=False)
    _print_outputs(session_dir)


def cmd_label(cfg: dict, args) -> None:
    from .label import label_session
    label_session(cfg, _session_dir(cfg, args.session))


def cmd_speakers(cfg: dict, args) -> None:
    from .voiceprints import VoiceprintDB
    db = VoiceprintDB(data_dir() / "voiceprints.db")
    try:
        if args.rename:
            old, new = args.rename
            print("Renamed." if db.rename(old, new) else f"'{old}' not found.")
        elif args.delete:
            print("Deleted." if db.delete(args.delete)
                  else f"'{args.delete}' not found.")
        else:
            speakers = db.list_speakers()
            if not speakers:
                print("No known voices yet. Use `meetrec label <session>`.")
            for s in speakers:
                updated = datetime.fromtimestamp(s["updated_at"])
                print(f"  {s['name']}  (samples: {s['num_samples']}, "
                      f"updated {updated:%Y-%m-%d})")
    finally:
        db.close()


def cmd_doctor(cfg: dict, args) -> None:
    ok = True

    def check(label: str, fn) -> None:
        nonlocal ok
        try:
            detail = fn()
            print(f"  [OK]   {label}" + (f" — {detail}" if detail else ""))
        except Exception as exc:  # noqa: BLE001 — doctor reports, not raises
            ok = False
            print(f"  [FAIL] {label} — {exc}")

    print("meetrec doctor\n")

    def deps():
        import faster_whisper  # noqa: F401
        import pyannote.audio  # noqa: F401
        import pyaudiowpatch  # noqa: F401
        import soundfile  # noqa: F401
        import windows_toasts  # noqa: F401
        return None
    check("Python dependencies", deps)

    def registry():
        from .registry_scan import scan_mic_usage
        entries = scan_mic_usage()
        return f"{len(entries)} ConsentStore entries"
    check("Registry access (ConsentStore)", registry)

    def audio():
        import pyaudiowpatch as pyaudio
        p = pyaudio.PyAudio()
        try:
            mic = p.get_default_input_device_info()["name"]
            loop = p.get_default_wasapi_loopback()["name"]
            return f"mic='{mic}', loopback='{loop}'"
        finally:
            p.terminate()
    check("Audio devices (mic + WASAPI loopback)", audio)

    def cuda():
        from .transcribe import resolve_device
        device, compute = resolve_device(cfg)
        if device == "cpu":
            return ("no CUDA — CPU int8; expect ~1-2x meeting duration "
                    "to transcribe")
        return f"CUDA available ({compute})"
    check("CUDA / compute device", cuda)

    def hf():
        if not env("HF_TOKEN"):
            raise RuntimeError(
                "HF_TOKEN missing (needed for diarization) — see .env.example")
        return "token set"
    check("HuggingFace token", hf)

    def summary_backend():
        if env("GEMINI_API_KEY"):
            return "gemini key set"
        import requests
        url = cfg["summary"]["ollama_url"]
        requests.get(url, timeout=5)
        return f"ollama reachable at {url}"
    check("Summary backend (gemini key or ollama)", summary_backend)

    def tg():
        from .telegram import test_connection
        return f"bot @{test_connection()}"
    check("Telegram connectivity", tg)

    print("\nAll good." if ok else "\nSome checks failed — see above.")
    sys.exit(0 if ok else 1)


def cmd_cleanup(cfg: dict, args) -> None:
    from .pipeline import cleanup_expired_audio
    retention = cfg["output"].get("audio_retention_h", 0)
    if not retention:
        print("output.audio_retention_h is 0 — audio is kept forever. "
              "Set e.g. 72 in config.yaml to expire audio after 72h.")
        return
    removed = cleanup_expired_audio(cfg, dry_run=args.dry_run)
    if not removed:
        print("Nothing to delete.")
        return
    verb = "Would delete" if args.dry_run else "Deleted"
    total = sum(p.stat().st_size for p in removed if p.exists())
    for path in removed:
        print(f"  {path}")
    print(f"{verb} {len(removed)} file(s), {total / 1e6:.0f} MB.")


def cmd_test_telegram(cfg: dict, args) -> None:
    from .telegram import flush_queue, send_summary, test_connection
    username = test_connection()
    send_summary("meetrec test message — Telegram delivery works.",
                 "meetrec test")
    flushed = flush_queue()  # deliver anything queued while unconfigured/offline
    print(f"Sent via bot @{username}. Check your Telegram."
          + (f" ({flushed} queued message(s) delivered too)" if flushed else ""))


def cmd_autostart(cfg: dict, args) -> None:
    task_name = "meetrec"
    if args.mode == "on":
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        cmd = f'"{pythonw}" -m meetrec start'
        subprocess.run(
            ["schtasks", "/Create", "/F", "/SC", "ONLOGON",
             "/TN", task_name, "/TR", cmd],
            check=True)
        print(f"Scheduled task '{task_name}' created (runs at logon).")
    else:
        subprocess.run(["schtasks", "/Delete", "/F", "/TN", task_name],
                       check=False)
        print(f"Scheduled task '{task_name}' removed.")


def _print_outputs(session_dir: Path) -> None:
    """Tell the user exactly where the results are."""
    print(f"\nDone. Output files in {session_dir}:")
    for name, label in (("transcript.txt", "transcript (plain text)"),
                        ("transcript.md", "transcript (markdown)"),
                        ("summary.md", "summary"),
                        ("audio.flac", "audio"),
                        ("meta.json", "metadata")):
        path = session_dir / name
        if path.exists():
            print(f"  {label:<24} {path}")


def _wait_for_enter(min_wait_s: float = 2.0) -> None:
    """Block until the user actually presses Enter, at least min_wait_s
    after this call. Reads raw console keys so stray buffered input (e.g. a
    leftover Enter from a previous Ctrl+C) can never stop the recording
    instantly; falls back to input() when stdin is not a console."""
    if not sys.stdin.isatty():
        input()
        return
    try:
        import msvcrt
    except ImportError:
        input()
        return
    started = time.monotonic()
    while True:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\r", "\n") and time.monotonic() - started >= min_wait_s:
                return
        else:
            time.sleep(0.05)


def _pid_alive(pid: int) -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True).stdout
    return str(pid) in out


# -- entry point -------------------------------------------------------------

def main() -> None:
    # Windows consoles default to a legacy codepage; keep output UTF-8
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
    parser = argparse.ArgumentParser(
        prog="meetrec",
        description="Automatic meeting recording, transcription and "
                    "diarization — 100% local audio processing.")
    parser.add_argument("--debug", action="store_true",
                        help="verbose DEBUG logging for this run")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup", help="interactive first-run wizard: keys, "
                                 "config, Telegram pairing, autostart")

    sub.add_parser("start", help="run the daemon (foreground)")

    p = sub.add_parser("stop", help="stop a running daemon (graceful)")
    p.add_argument("--force", action="store_true",
                   help="kill immediately instead of finishing processing")

    sub.add_parser("status", help="show daemon state")

    p = sub.add_parser("pause", help="toggle pause (skip new meetings)")
    p.add_argument("--for", dest="duration", metavar="DURATION",
                   help="auto-resume after e.g. 30m, 2h, 1h30m")

    p = sub.add_parser("record",
                       help="record manually now (Enter stops), then process")
    p.add_argument("--in-person", action="store_true", dest="in_person",
                   help="in-person meeting: everyone is in the room, so the "
                        "MIC track is diarized (instead of labeling it as "
                        "you) and speakers are identified from it")
    sub.add_parser("list", help="list sessions and their processing state")

    p = sub.add_parser("summary",
                       help="regenerate just the summary of a session")
    p.add_argument("session", help="session dir (name or full path)")
    p.add_argument("--backend", choices=["gemini", "ollama", "anthropic"])
    p.add_argument("--language", help="force summary language (e.g. pt, en)")
    p.add_argument("--resend", action="store_true",
                   help="also send the new summary to Telegram")

    p = sub.add_parser("reprocess", help="re-run processing on a session")
    p.add_argument("session", help="session dir (name or full path)")
    p.add_argument("--full", action="store_true",
                   help="discard intermediates and redo everything")

    p = sub.add_parser("label", help="name unknown speakers in a session")
    p.add_argument("session", help="session dir (name or full path)")

    p = sub.add_parser("speakers", help="list/rename/delete known voices")
    p.add_argument("--rename", nargs=2, metavar=("OLD", "NEW"))
    p.add_argument("--delete", metavar="NAME")

    sub.add_parser("doctor", help="validate the whole setup")
    sub.add_parser("test-telegram", help="send a test message")

    p = sub.add_parser("cleanup",
                       help="delete audio older than output.audio_retention_h")
    p.add_argument("--dry-run", action="store_true",
                   help="show what would be deleted without deleting")

    p = sub.add_parser("debug",
                       help="inspect a session: stages, timings, issues")
    p.add_argument("session", help="session dir (name or full path)")
    p.add_argument("--traceback", action="store_true",
                   help="print full tracebacks of recorded issues")

    p = sub.add_parser("autostart", help="enable/disable start at logon")
    p.add_argument("mode", choices=["on", "off"])

    args = parser.parse_args()
    cfg = load_config()
    handlers = {
        "setup": cmd_setup, "start": cmd_start, "stop": cmd_stop,
        "status": cmd_status, "pause": cmd_pause, "record": cmd_record,
        "list": cmd_list, "summary": cmd_summary, "reprocess": cmd_reprocess,
        "label": cmd_label, "speakers": cmd_speakers, "doctor": cmd_doctor,
        "debug": cmd_debug, "cleanup": cmd_cleanup,
        "test-telegram": cmd_test_telegram, "autostart": cmd_autostart,
    }
    handlers[args.command](cfg, args)


if __name__ == "__main__":
    main()
