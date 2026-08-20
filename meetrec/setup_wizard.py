"""Interactive first-run wizard: `meetrec setup`.

One shot: walks through every key and config choice in the terminal, with
the links for whatever needs to be created (HuggingFace token, Gemini key,
Telegram bot), writes .env and config.yaml, detects the Telegram chat id
automatically, runs a self-test and optionally enables autostart.

Everything can be skipped with Enter and re-run later — the wizard is
idempotent and keeps existing values as defaults.
"""

import os
import time
from pathlib import Path

import requests
import yaml

from .config import PROJECT_ROOT, load_config

LINKS = {
    "hf_terms_1": "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "hf_terms_2": "https://huggingface.co/pyannote/segmentation-3.0",
    "hf_token": "https://huggingface.co/settings/tokens",
    "gemini": "https://aistudio.google.com/apikey",
    "botfather": "https://t.me/BotFather",
}


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"  {prompt}{suffix}: ").strip()
    return value or default


def _ask_yn(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    value = input(f"  {prompt} ({hint}): ").strip().lower()
    if not value:
        return default
    return value in ("y", "yes")


def _mask(secret: str) -> str:
    return secret[:6] + "…" + secret[-4:] if len(secret) > 12 else "set"


def _read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                values[key.strip()] = val.strip()
    return values


def _write_env(path: Path, values: dict[str, str]) -> None:
    lines = ["# Written by `meetrec setup`. Never commit this file."]
    lines += [f"{k}={v}" for k, v in values.items() if v]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _detect_chat_id(bot_token: str) -> str | None:
    """Ask the user to message the bot, then read the chat id from
    getUpdates — no manual JSON digging needed."""
    try:
        me = requests.get(
            f"https://api.telegram.org/bot{bot_token}/getMe", timeout=15
        ).json()
        if not me.get("ok"):
            print("  ! That bot token was rejected by Telegram.")
            return None
        username = me["result"]["username"]
    except requests.RequestException as exc:
        print(f"  ! Could not reach Telegram: {exc}")
        return None

    print(f"\n  Now open Telegram and send any message to @{username}.")
    print("  Waiting up to 60s for it", end="", flush=True)
    deadline = time.time() + 60
    while time.time() < deadline:
        time.sleep(3)
        print(".", end="", flush=True)
        try:
            updates = requests.get(
                f"https://api.telegram.org/bot{bot_token}/getUpdates",
                timeout=15).json()
        except requests.RequestException:
            continue
        for update in reversed(updates.get("result", [])):
            chat = (update.get("message") or {}).get("chat")
            if chat:
                print(f"\n  Found chat: {chat.get('first_name', '')} "
                      f"(id {chat['id']})")
                return str(chat["id"])
    print("\n  ! No message received — you can fill TELEGRAM_CHAT_ID later.")
    return None


def run_wizard(cfg: dict) -> None:
    print("\n=== meetrec setup ===\n")
    print("Enter accepts the [default]; empty skips optional steps.\n")

    env_path = PROJECT_ROOT / ".env"
    env = _read_env(env_path)

    # 1. HuggingFace (diarization)
    print("1) Speaker diarization — HuggingFace token (free)")
    print(f"   Accept the terms on BOTH pages (same account):")
    print(f"     {LINKS['hf_terms_1']}")
    print(f"     {LINKS['hf_terms_2']}")
    print(f"   Then create a READ token at {LINKS['hf_token']}")
    current = env.get("HF_TOKEN", "")
    token = _ask("HF token (Enter keeps current)" if current
                 else "HF token (Enter skips diarization for now)",
                 current)
    env["HF_TOKEN"] = token
    print(f"   -> {_mask(token) if token else 'skipped'}\n")

    # 2. Gemini (summaries)
    print("2) Summaries — Google Gemini API key (free, no credit card)")
    print(f"   Create one at {LINKS['gemini']}")
    print("   Skipping is fine: a local Ollama model is used instead.")
    current = env.get("GEMINI_API_KEY", "")
    env["GEMINI_API_KEY"] = _ask("Gemini key", current)
    print(f"   -> {_mask(env['GEMINI_API_KEY']) if env['GEMINI_API_KEY'] else 'skipped (Ollama fallback)'}\n")

    # 3. Telegram
    print("3) Telegram delivery")
    print(f"   Create a bot with {LINKS['botfather']} (/newbot) and paste "
          "its token.")
    current = env.get("TELEGRAM_BOT_TOKEN", "")
    bot_token = _ask("Bot token", current)
    env["TELEGRAM_BOT_TOKEN"] = bot_token
    if bot_token:
        chat_id = env.get("TELEGRAM_CHAT_ID", "")
        if chat_id and _ask_yn(f"Keep current chat id ({chat_id})?"):
            pass
        else:
            detected = _detect_chat_id(bot_token)
            if detected:
                env["TELEGRAM_CHAT_ID"] = detected
            else:
                env["TELEGRAM_CHAT_ID"] = _ask(
                    "Chat id (manual entry, Enter to skip)",
                    env.get("TELEGRAM_CHAT_ID", ""))
    print()

    # 4. config.yaml choices
    print("4) Preferences (stored in config.yaml)")
    out_dir = _ask("Folder for meeting outputs", cfg["output"]["dir"])
    language = _ask("Summary language: auto = meeting language, or pt/en",
                    cfg["summary"].get("language", "auto"))
    multilingual = _ask_yn(
        "Do your meetings mix languages (e.g. PT and EN)? Enables "
        "per-segment language detection",
        cfg["transcription"].get("multilingual", True))
    send_transcript = _ask_yn(
        "Send the FULL transcript to Telegram too? (summary is always sent)",
        cfg["telegram"]["send_full_transcript"])

    config_path = PROJECT_ROOT / "config.yaml"
    file_cfg = {}
    if config_path.exists():
        file_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    file_cfg.setdefault("output", {})["dir"] = out_dir
    file_cfg.setdefault("summary", {})["language"] = language
    file_cfg.setdefault("transcription", {})["multilingual"] = multilingual
    file_cfg.setdefault("telegram", {})["send_full_transcript"] = send_transcript
    config_path.write_text(yaml.safe_dump(file_cfg, sort_keys=False,
                                          allow_unicode=True),
                           encoding="utf-8")
    _write_env(env_path, env)
    print(f"\n  Saved {env_path.name} and config.yaml.\n")
    for key, value in env.items():
        if value:
            os.environ[key] = value

    # 5. self-test
    if env.get("TELEGRAM_BOT_TOKEN") and env.get("TELEGRAM_CHAT_ID") \
            and _ask_yn("Send a Telegram test message now?"):
        from .telegram import send_summary
        try:
            send_summary("meetrec is configured. You will receive meeting "
                         "summaries here.", "meetrec setup")
            print("  Test message sent — check Telegram.\n")
        except Exception as exc:  # noqa: BLE001 — wizard must not crash
            print(f"  ! Test failed: {exc}\n")

    # 6. autostart
    if _ask_yn("Start meetrec automatically at Windows logon?", False):
        import subprocess
        import sys
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        subprocess.run(
            ["schtasks", "/Create", "/F", "/SC", "ONLOGON", "/TN", "meetrec",
             "/TR", f'"{pythonw}" -m meetrec start'], check=False)
        print("  Scheduled task created.\n")

    print("Done. Next steps:")
    print("  meetrec doctor   # full validation of this machine")
    print("  meetrec start    # run the daemon and join a meeting")
    if not env.get("HF_TOKEN"):
        print("  (diarization is off until HF_TOKEN is set — rerun "
              "`meetrec setup` anytime)")
