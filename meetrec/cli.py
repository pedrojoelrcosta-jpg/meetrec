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

def cmd_start(cfg: dict, args) -> None:
    from .daemon import Daemon, read_state
    state = read_state()
    if state and _pid_alive(state["pid"]):
        sys.exit(f"Daemon already running (pid {state['pid']})")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    Daemon(cfg).run()


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
    from .daemon import pause_path, read_state
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
    print(f"Paused flag: {'yes' if pause_path().exists() else 'no'}")
    print(f"Output dir: {cfg['output']['dir']}")


def cmd_pause(cfg: dict, args) -> None:
    from .daemon import pause_path
    if pause_path().exists():
        pause_path().unlink()
        print("Resumed — new meetings will be recorded.")
    else:
        pause_path().write_text(datetime.now().isoformat(), encoding="utf-8")
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
    parser = argparse.ArgumentParser(
        prog="meetrec",
        description="Automatic meeting recording, transcription and "
                    "diarization — 100% local audio processing.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("start", help="run the daemon (foreground)")
    sub.add_parser("stop", help="stop a running daemon")
    sub.add_parser("status", help="show daemon state")
    sub.add_parser("pause", help="toggle pause (skip new meetings)")

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
        "pause": cmd_pause, "reprocess": cmd_reprocess, "label": cmd_label,
        "speakers": cmd_speakers, "doctor": cmd_doctor,
        "test-telegram": cmd_test_telegram, "autostart": cmd_autostart,
    }
    handlers[args.command](cfg, args)


if __name__ == "__main__":
    main()
