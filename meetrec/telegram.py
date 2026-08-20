"""Telegram delivery: summary via sendMessage, transcript via sendDocument.

- HTML parse_mode (MarkdownV2 escaping is a minefield)
- messages split at the 4096-char limit, on line boundaries when possible
- retry with backoff; on definitive failure the payload is queued on disk
  and retried on the next run — the result is never lost
"""

import html
import json
import logging
import time
import uuid
from pathlib import Path

import requests

from .config import data_dir, env, find_transcript

log = logging.getLogger(__name__)

API = "https://api.telegram.org"
MAX_MESSAGE_CHARS = 4096
RETRIES = 4
BACKOFF_BASE_S = 2.0


class TelegramNotConfigured(RuntimeError):
    pass


def _credentials() -> tuple[str, str]:
    token = env("TELEGRAM_BOT_TOKEN")
    chat_id = env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        raise TelegramNotConfigured(
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID missing from .env")
    return token, chat_id


def _queue_dir() -> Path:
    d = data_dir() / "telegram_queue"
    d.mkdir(parents=True, exist_ok=True)
    return d


def split_message(text: str, limit: int = MAX_MESSAGE_CHARS) -> list[str]:
    """Split text into <=limit chunks, preferring line boundaries."""
    if len(text) <= limit:
        return [text]
    chunks, current = [], ""
    for line in text.splitlines(keepends=True):
        while len(line) > limit:  # single pathological line
            room = limit - len(current)
            current += line[:room]
            chunks.append(current)
            current, line = "", line[room:]
        if len(current) + len(line) > limit:
            chunks.append(current)
            current = ""
        current += line
    if current:
        chunks.append(current)
    return chunks


def _post(method: str, *, data: dict, files: dict | None = None) -> None:
    token, _ = _credentials()
    url = f"{API}/bot{token}/{method}"
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        try:
            response = requests.post(url, data=data, files=files, timeout=60)
            if response.status_code == 429:
                retry_after = response.json().get("parameters", {}) \
                                             .get("retry_after", 5)
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            return
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            time.sleep(BACKOFF_BASE_S * 2 ** attempt)
    raise last_error  # type: ignore[misc]


def send_summary(summary_markdown: str, title: str) -> None:
    _, chat_id = _credentials()
    text = f"<b>{html.escape(title)}</b>\n\n{html.escape(summary_markdown)}"
    for chunk in split_message(text):
        _post("sendMessage", data={
            "chat_id": chat_id, "text": chunk, "parse_mode": "HTML"})


def send_document(path: Path, caption: str) -> None:
    _, chat_id = _credentials()
    with path.open("rb") as fh:
        _post("sendDocument",
              data={"chat_id": chat_id, "caption": caption[:1024]},
              files={"document": (path.name, fh)})


def deliver(cfg: dict, session_dir: Path, summary: str, title: str) -> bool:
    """Send summary (+ transcript if enabled). On failure, queue for later.

    Returns True if delivered now, False if queued.
    """
    if not cfg["telegram"]["enabled"]:
        return True
    try:
        send_summary(summary, title)
        if cfg["telegram"]["send_full_transcript"]:
            transcript = find_transcript(session_dir)
            if transcript:
                send_document(transcript, f"Transcript — {title}")
        return True
    except TelegramNotConfigured:
        raise
    except Exception:
        log.exception("Telegram delivery failed; queuing for retry")
        _enqueue({"session_dir": str(session_dir), "summary": summary,
                  "title": title,
                  "send_transcript": cfg["telegram"]["send_full_transcript"]})
        return False


def _enqueue(payload: dict) -> None:
    path = _queue_dir() / f"{int(time.time())}_{uuid.uuid4().hex[:8]}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def flush_queue() -> int:
    """Retry queued deliveries. Returns how many were sent."""
    sent = 0
    for item in sorted(_queue_dir().glob("*.json")):
        try:
            payload = json.loads(item.read_text(encoding="utf-8"))
            send_summary(payload["summary"], payload["title"])
            if payload.get("send_transcript"):
                transcript = find_transcript(Path(payload["session_dir"]))
                if transcript:
                    send_document(transcript,
                                  f"Transcript — {payload['title']}")
            item.unlink()
            sent += 1
        except Exception:
            log.warning("Queued delivery %s still failing; kept", item.name)
            break  # network is likely still down — stop trying
    return sent


def test_connection() -> str:
    """getMe roundtrip; returns the bot username."""
    token, _ = _credentials()
    response = requests.get(f"{API}/bot{token}/getMe", timeout=30)
    response.raise_for_status()
    return response.json()["result"]["username"]
