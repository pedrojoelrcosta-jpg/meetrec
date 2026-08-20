"""config.yaml + .env loading, with complete defaults in code."""

import os
from copy import deepcopy
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "detector": {
        "poll_interval_s": 2,
        "start_debounce_s": 15,
        "stop_debounce_s": 30,
        "ignore_apps": [
            "SoundRecorder.exe",
            "Microsoft.WindowsSoundRecorder_8wekyb3d8bbwe",
        ],
    },
    "audio": {"min_session_s": 60},
    "transcription": {
        "model": "large-v3-turbo",
        "device": "auto",
        "compute_type": "auto",
    },
    "diarization": {"similarity_threshold": 0.75},
    "summary": {
        "backend": "gemini",
        "language": "auto",
        "gemini_model": "gemini-2.0-flash",
        "ollama_model": "gemma4:26b",
        "ollama_url": "http://localhost:11434",
        "anthropic_model": "claude-sonnet-5",
    },
    "output": {"dir": "~/Reunioes", "keep_wav": False},
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
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None = None) -> dict:
    load_dotenv(PROJECT_ROOT / ".env")
    path = path or PROJECT_ROOT / "config.yaml"
    user_cfg = {}
    if path.exists():
        user_cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = _deep_merge(DEFAULTS, user_cfg)
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
