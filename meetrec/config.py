"""config.yaml + .env loading, with complete defaults in code."""

import os
from copy import deepcopy
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Canonical output filenames (session folder artifacts)
TRANSCRIPT_TXT = "transcript.txt"
TRANSCRIPT_MD = "transcript.md"
SUMMARY_MD = "summary.md"
# Pre-1.0 sessions used Portuguese names; still recognized when reading
LEGACY_TRANSCRIPT_TXT = "transcricao.txt"
LEGACY_SUMMARY_MD = "resumo.md"


def find_transcript(session_dir: Path) -> Path | None:
    """The session's plain-text transcript, accepting the legacy name."""
    for name in (TRANSCRIPT_TXT, LEGACY_TRANSCRIPT_TXT):
        if (session_dir / name).exists():
            return session_dir / name
    return None

DEFAULTS: dict = {
    "detector": {
        "poll_interval_s": 2,
        "start_debounce_s": 15,
        "stop_debounce_s": 30,
        "ignore_apps": [
            "SoundRecorder.exe",
            "Microsoft.WindowsSoundRecorder_8wekyb3d8bbwe",
            # meetrec itself records through python — without these, a manual
            # `meetrec record` would trigger the daemon into a duplicate
            # recording of the same meeting
            "python.exe",
            "pythonw.exe",
        ],
    },
    "audio": {
        "min_session_s": 60,
        # drop mic segments that duplicate simultaneous system audio —
        # speaker bleed when the user listens on speakers, not headphones
        "echo_dedup": True,
    },
    "transcription": {
        "model": "large-v3-turbo",
        "device": "auto",
        "compute_type": "auto",
        # re-detect the language per segment — meetings that switch between
        # languages (e.g. PT and EN) transcribe correctly at a small speed cost
        "multilingual": True,
    },
    "diarization": {
        "similarity_threshold": 0.75,
        "self_label": "ME",  # how the mic-track speaker (you) is labeled
    },
    "summary": {
        "backend": "gemini",
        "language": "auto",
        "gemini_model": "gemini-2.0-flash",
        "ollama_model": "gemma4:26b",
        "ollama_url": "http://localhost:11434",
        "anthropic_model": "claude-sonnet-5",
    },
    "output": {
        "dir": "~/Meetings",
        "keep_wav": False,
        # 0 = keep audio.flac forever; N = delete audio N hours after
        # processing (transcripts/summary always kept). Label speakers
        # before it expires — excerpts need the audio.
        "audio_retention_h": 0,
    },
    "telegram": {"enabled": True, "send_full_transcript": False},
    "debug": {
        "level": "info",    # info | debug — logging verbosity
        "strict": False,    # True: pipeline stage errors raise instead of
                            # being recorded-and-skipped (for debugging)
    },
    "notifications": {
        # "recording started" is intentionally NOT configurable: the user
        # must always know when recording begins
        "recording_stopped": True,
        "processing_done": True,
        "speakers_unlabeled": True,
        "errors": True,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        elif value is None and isinstance(merged.get(key), dict):
            # an emptied-out YAML section ("telegram:" with no keys) parses
            # as None — keep the defaults instead of wiping the section
            continue
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None) -> dict:
    """DEFAULTS <- config.yaml (repo, commented) <- config.local.yaml
    (gitignored — written by `meetrec setup`, survives git pulls)."""
    load_dotenv(PROJECT_ROOT / ".env")
    cfg = DEFAULTS
    layers = [path] if path else [PROJECT_ROOT / "config.yaml",
                                  PROJECT_ROOT / "config.local.yaml"]
    for layer in layers:
        if layer and layer.exists():
            loaded = yaml.safe_load(layer.read_text(encoding="utf-8")) or {}
            cfg = _deep_merge(cfg, loaded)
    cfg["output"]["dir"] = str(Path(cfg["output"]["dir"]).expanduser())
    return cfg


def env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def data_dir() -> Path:
    """Persistent state (voiceprint DB, Telegram queue, daemon state)."""
    root = Path(os.environ.get("LOCALAPPDATA", "~")).expanduser() / "meetrec"
    root.mkdir(parents=True, exist_ok=True)
    return root
