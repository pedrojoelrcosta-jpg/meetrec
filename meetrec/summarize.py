"""Meeting summary via Google Gemini (free API key), local Ollama, or the
Anthropic API.

The summary prompt is generated in the language detected in the transcript,
not hardcoded.

Gemini free tier: get a key at https://aistudio.google.com/apikey (no credit
card needed) and put GEMINI_API_KEY=... in .env. Free keys are heavily
rate-limited (429), so this module retries with generous backoff — and a
summary failure never blocks the pipeline: the transcript is always saved
first, and the configured fallback backends are tried in order.
"""

import logging
import time

import requests

from .config import env

log = logging.getLogger(__name__)

GEMINI_RETRIES = 6          # free-tier keys often return 429/503 when busy
GEMINI_BACKOFF_BASE_S = 5.0

_PROMPTS = {
    "pt": (
        "És um assistente que resume reuniões. A partir da transcrição "
        "abaixo, produz em português:\n\n"
        "## Sumário executivo\n(3-6 frases)\n\n"
        "## Decisões\n(lista; se nenhuma, escreve 'Nenhuma')\n\n"
        "## Ações\n(lista com responsável atribuído a partir dos nomes dos "
        "oradores, formato '- [Nome] ação')\n\n"
        "## Questões em aberto\n(lista; se nenhuma, escreve 'Nenhuma')\n\n"
        "Transcrição:\n"
    ),
    "en": (
        "You are an assistant that summarizes meetings. From the transcript "
        "below, produce in English:\n\n"
        "## Executive summary\n(3-6 sentences)\n\n"
        "## Decisions\n(list; if none, write 'None')\n\n"
        "## Action items\n(list with an owner assigned from the speaker "
        "names, format '- [Name] action')\n\n"
        "## Open questions\n(list; if none, write 'None')\n\n"
        "Transcript:\n"
    ),
}


def build_prompt(language: str, transcript_text: str) -> str:
    prompt = _PROMPTS.get(language, _PROMPTS["en"])
    return prompt + transcript_text


def summarize(cfg: dict, language: str, transcript_text: str) -> str:
    """Try the configured backend first, then the other configured ones.

    Raises only if every available backend fails — and the caller treats
    that as non-fatal (transcript is already on disk).
    """
    prompt = build_prompt(language, transcript_text)
    backends = {"gemini": _gemini, "ollama": _ollama, "anthropic": _anthropic}
    preferred = cfg["summary"]["backend"]
    order = [preferred] + [b for b in backends if b != preferred]
    last_error: Exception | None = None
    for name in order:
        try:
            result = backends[name](cfg, prompt)
            log.info("Summary produced by %s backend", name)
            return result
        except _NotConfigured as exc:
            log.debug("Backend %s not configured: %s", name, exc)
        except Exception as exc:  # noqa: BLE001 — try the next backend
            log.warning("Backend %s failed: %s", name, exc)
            last_error = exc
    raise last_error or RuntimeError(
        "No summary backend is configured. Set GEMINI_API_KEY in .env "
        "(free key: https://aistudio.google.com/apikey), run Ollama, or "
        "set ANTHROPIC_API_KEY.")


class _NotConfigured(RuntimeError):
    pass


class _Retryable(RuntimeError):
    pass


def _gemini(cfg: dict, prompt: str) -> str:
    # Free API key from Google AI Studio: https://aistudio.google.com/apikey
    # Free-tier quota is shared and often busy — retry hard before giving up.
    api_key = env("GEMINI_API_KEY")
    if not api_key:
        raise _NotConfigured("GEMINI_API_KEY not set in .env")
    model = cfg["summary"].get("gemini_model", "gemini-2.0-flash")
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent")
    last_error: Exception | None = None
    for attempt in range(GEMINI_RETRIES):
        try:
            response = requests.post(
                url,
                # key goes in a header, never in the URL: query params end
                # up verbatim in logs, tracebacks and debug.json
                headers={"x-goog-api-key": api_key},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=300,
            )
            if response.status_code in (429, 500, 502, 503, 504):
                # transient / rate-limit: worth waiting for
                raise _Retryable(f"HTTP {response.status_code}")
            response.raise_for_status()  # 400/401/403… fail immediately
            data = response.json()
            return "".join(
                part.get("text", "")
                for part in data["candidates"][0]["content"]["parts"]
            ).strip()
        except (_Retryable, requests.ConnectionError,
                requests.Timeout) as exc:
            last_error = exc
            wait = GEMINI_BACKOFF_BASE_S * 2 ** attempt
            log.warning("Gemini attempt %d/%d failed (%s); retrying in %.0fs",
                        attempt + 1, GEMINI_RETRIES, exc, wait)
            time.sleep(wait)
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Gemini returned an unexpected response "
                               f"shape: {exc}") from exc
    raise last_error  # type: ignore[misc]


def _ollama(cfg: dict, prompt: str) -> str:
    url = cfg["summary"]["ollama_url"].rstrip("/") + "/api/generate"
    try:
        response = requests.post(url, json={
            "model": cfg["summary"]["ollama_model"],
            "prompt": prompt,
            "stream": False,
        }, timeout=600)
    except requests.ConnectionError as exc:
        raise _NotConfigured(f"Ollama not reachable at {url}") from exc
    response.raise_for_status()
    return response.json()["response"].strip()


def _anthropic(cfg: dict, prompt: str) -> str:
    api_key = env("ANTHROPIC_API_KEY")
    if not api_key:
        raise _NotConfigured("ANTHROPIC_API_KEY not set in .env")
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": cfg["summary"]["anthropic_model"],
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=300,
    )
    response.raise_for_status()
    return "".join(
        part["text"] for part in response.json()["content"]
        if part["type"] == "text"
    ).strip()
