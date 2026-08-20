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

def _setup_logging() -> None:
    from logging.handlers import RotatingFileHandler
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    file_handler = RotatingFileHandler(
        data_dir() / "meetrec.log", maxBytes=2_000_000, backupCount=3,
        encoding="utf-8")
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def cmd_start(cfg: dict, args) -> None:
    from .daemon import Daemon, read_state
    state = read_state()
    if state and _pid_alive(state["pid"]):
        sys.exit(f"Daemon already running (pid {state['pid']})")
    _setup_logging()
    Daemon(cfg).run()


def cmd_record(cfg: dict, args) -> None:
    """Manual recording, independent of meeting detection — e.g. in-person
    meetings. Enter stops and processing starts right away."""
    from . import notify
    from .audio_capture import DualTrackRecorder
    from .pipeline import new_session_dir, process_session
    _setup_logging()
    recorder = DualTrackRecorder(new_session_dir(cfg))
    recorder.start()
    notify.recording_started({"manual recording"})
    print(f"Recording to {recorder.session_dir}")
    input("Press Enter to stop...\n")
    stats = recorder.stop()
    print(f"Recorded {stats['duration_s']}s. Processing...")
    import json as _json
    (recorder.session_dir / "meta.json").write_text(
        _json.dumps({"date": recorder.session_dir.name,
                     "duration_s": stats["duration_s"], "tracks": stats,
                     "manual": True}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    process_session(cfg, recorder.session_dir)
    print(f"Done: {recorder.session_dir}")


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
            state = "ok" if (session / "resumo.md").exists() else (
                "no summary" if (session / "transcricao.txt").exists()
                else "unprocessed")
            speakers = ", ".join(meta.get("speakers", [])) or "-"
            print(f"  {session.name}  {duration // 60:3d}min  "
                  f"lang={meta.get('language', '?')}  [{state}]  {speakers}")
        else:
            print(f"  {session.name}  (no metadata — unprocessed)")


def cmd_summary(cfg: dict, args) -> None:
    """Regenerate (and optionally resend) just the summary of a session."""
    from . import telegram
    from .summarize import summarize
    session_dir = _session_dir(cfg, args.session)
    transcript = session_dir / "transcricao.txt"
    if not transcript.exists():
        sys.exit("Session has no transcript yet — run reprocess first.")
    if args.backend:
        cfg["summary"]["backend"] = args.backend
    meta = json.loads((session_dir / "meta.json").read_text(encoding="utf-8"))
    lang_cfg = args.language or cfg["summary"].get("language", "auto")
    language = meta.get("language", "en") if lang_cfg == "auto" else lang_cfg
    text = summarize(cfg, language, transcript.read_text(encoding="utf-8"))
    (session_dir / "resumo.md").write_text(text, encoding="utf-8")
    print(f"Summary written ({cfg['summary']['backend']}, {language}).")
    if args.resend:
        telegram.deliver(cfg, session_dir, text,
                         f"Meeting {session_dir.name}")
        print("Sent to Telegram.")


def cmd_stop(cfg: dict, args) -> None:
    from .daemon import read_state, state_path
    state = read_state()
    if not state or not _pid_alive(state["pid"]):
        state_path().unlink(missing_ok=True)
        print("Daemon is not running.")
        return
    subprocess.run(["taskkill", "/PID", str(state["pid"]), "/F"], check=False)
    state_path().unlink(missing_ok=True)
    print(f"Daemon (pid {state['pid']}) stopped.")


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
                     "embeddings.npz", "resumo.md"):
            (session_dir / name).unlink(missing_ok=True)
    logging.basicConfig(level=logging.INFO)
    process_session(cfg, session_dir, with_notifications=False)
    print(f"Reprocessed {session_dir}")


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
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("start", help="run the daemon (foreground)")
    sub.add_parser("stop", help="stop a running daemon")
    sub.add_parser("status", help="show daemon state")

    p = sub.add_parser("pause", help="toggle pause (skip new meetings)")
    p.add_argument("--for", dest="duration", metavar="DURATION",
                   help="auto-resume after e.g. 30m, 2h, 1h30m")

    sub.add_parser("record",
                   help="record manually now (Enter stops), then process")
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

    p = sub.add_parser("autostart", help="enable/disable start at logon")
    p.add_argument("mode", choices=["on", "off"])

    args = parser.parse_args()
    cfg = load_config()
    handlers = {
        "start": cmd_start, "stop": cmd_stop, "status": cmd_status,
        "pause": cmd_pause, "record": cmd_record, "list": cmd_list,
        "summary": cmd_summary, "reprocess": cmd_reprocess,
        "label": cmd_label, "speakers": cmd_speakers, "doctor": cmd_doctor,
        "test-telegram": cmd_test_telegram, "autostart": cmd_autostart,
    }
    handlers[args.command](cfg, args)


if __name__ == "__main__":
    main()
