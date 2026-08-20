# meetrec

Automatic meeting recording, transcription, diarization and summarization for
Windows 11 — with **100% local audio processing**. No cloud STT, no browser
extensions, no virtual audio cables.

A background daemon detects when you are in a meeting (Google Meet, Teams,
Zoom — any app that captures the microphone), records both sides of the
conversation, transcribes and identifies who said what, summarizes it, and
delivers the result as local files + Telegram, with Windows toast
notifications along the way.

## How it works

```
IDLE ──mic capture detected──▶ RECORDING ──meeting ends──▶ PROCESSING ──▶ DELIVERING
```

- **Detection** — polls the Windows ConsentStore registry
  (`HKCU/HKLM\...\CapabilityAccessManager\ConsentStore\microphone`), where
  `LastUsedTimeStop == 0` means an app is capturing the mic right now.
  Debounced: recording starts after 15 s of continuous capture, stops after
  30 s without it. No hooks into any specific browser or app.
- **Recording** — two synchronized tracks via WASAPI:
  `track_mic.wav` (your microphone) and `track_sys.wav` (loopback of the
  default output = everyone else). Written to disk incrementally; survives
  audio-device changes mid-meeting.
- **Transcription** — [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  `large-v3-turbo`, automatic language detection (PT/EN, including mixed),
  VAD filter, word timestamps. Uses CUDA if available, otherwise CPU int8.
- **Diarization** — [pyannote speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  on the system track only (your voice is already isolated on the mic track
  and labeled `EU`). A persistent **voiceprint database** (SQLite + cosine
  similarity over speaker embeddings) recognizes people across meetings;
  unknown speakers stay `SPEAKER_XX` until you name them once with
  `meetrec label`.
- **Summary** — executive summary, decisions, action items with owners, open
  questions — generated in the language of the meeting. Backends: Google
  Gemini (free API key), local Ollama, or the Anthropic API. A summary
  failure never loses the transcript.
- **Delivery** — session folder + Telegram (summary as messages, transcript
  as a document only if you opt in) + Windows toasts. Failed Telegram sends
  are queued on disk and retried later; results are never lost.

## Output

```
~/Reunioes/2026-08-20_1400/
├── audio.flac         # stereo: mic on the left, system on the right
├── transcricao.txt    # [HH:MM:SS] Speaker: text — one block per turn
├── transcricao.md
├── resumo.md
└── meta.json          # duration, language, speakers, models, confidence
```

## Setup

Requires Windows 11 and Python 3.11+.

```powershell
git clone https://github.com/pedrojoelrcosta-jpg/meetrec
cd meetrec
py -3.11 -m venv .venv
.venv\Scripts\pip install -e .
copy .env.example .env   # then fill it in — see below
.venv\Scripts\meetrec doctor
```

### Verify the detection mechanism on your machine (recommended first step)

```powershell
py diagnose_mic.py --watch
```

Join a meeting with the mic on: you should see `STARTED capturing:
<your app>`. Leave it: `STOPPED capturing`. If that works, everything else
will.

### API keys (.env)

| Key | Needed for | How to get it |
|---|---|---|
| `HF_TOKEN` | Diarization (who spoke) | Accept the terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0), then create a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Free. |
| `GEMINI_API_KEY` | Summaries (default backend) | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — free, no credit card. Free keys hit rate limits when busy; meetrec retries with backoff and falls back to Ollama/Anthropic automatically. |
| `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` | Telegram delivery | Create a bot with [@BotFather](https://t.me/BotFather); get your chat id from `https://api.telegram.org/bot<token>/getUpdates` after messaging the bot. |
| `ANTHROPIC_API_KEY` | Optional summary backend | [console.anthropic.com](https://console.anthropic.com) |

Everything audio-related (recording, transcription, diarization,
voiceprints) is fully local. The only data that leaves the machine is the
summary text sent to the summary backend and to Telegram — and the full
transcript only if you set `telegram.send_full_transcript: true`.

## Usage

```
meetrec start            # run the daemon (foreground; see autostart below)
meetrec stop             # stop it
meetrec status           # daemon state, current recording
meetrec pause            # toggle: skip new meetings until resumed
meetrec reprocess <dir>  # re-run processing on a session (--full = from scratch)
meetrec label <dir>      # play excerpts of unknown speakers and name them
meetrec speakers         # list known voices (--rename OLD NEW | --delete NAME)
meetrec doctor           # validate deps, tokens, registry, audio, CUDA, Telegram
meetrec test-telegram    # send a test message
meetrec autostart on     # run at logon via Task Scheduler (off to remove)
```

The recording-start notification is intentionally **not** disableable — you
should always know when meetrec is recording.

## Configuration

See [config.yaml](config.yaml): Whisper model/device, debounce thresholds,
minimum session length, voice-similarity threshold, output directory,
summary backend, Telegram flags, apps to ignore.

## Development

```powershell
.venv\Scripts\python -m pytest tests/
```

The state machine and the embedding matching are tested with injected fake
scanners/clocks and synthetic vectors — no real meetings needed.
