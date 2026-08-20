# meetrec

**Automatic meeting recording, transcription, speaker identification and
summarization for Windows 11 — with 100% local audio processing.**

![tests](https://github.com/pedrojoelrcosta-jpg/meetrec/actions/workflows/ci.yml/badge.svg)
![license](https://img.shields.io/badge/license-MIT-green)
![python](https://img.shields.io/badge/python-3.11%2B-blue)

No cloud speech-to-text. No browser extensions. No virtual audio cables.
A background daemon notices when you are in a meeting (Google Meet, Microsoft
Teams, Zoom — any app that captures the microphone), records both sides of
the conversation, figures out who said what, writes a transcript and a
summary, and delivers everything as local files + Telegram messages, with
Windows toast notifications along the way.

---

## Table of contents

- [Why meetrec](#why-meetrec)
- [How it works](#how-it-works)
- [What you get](#what-you-get)
- [Minimum requirements](#minimum-requirements)
- [Installation](#installation)
- [First run: validate detection on your machine](#first-run-validate-detection-on-your-machine)
- [API keys and .env](#api-keys-and-env)
- [Usage](#usage)
- [Speaker identification workflow](#speaker-identification-workflow)
- [Configuration reference](#configuration-reference)
- [Performance expectations](#performance-expectations)
- [Privacy model](#privacy-model)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

---

## Why meetrec

Meeting-notes SaaS products send your audio to someone else's cloud. Browser
extensions break when you switch profiles or browsers, and bots that join
your calls announce themselves to everyone. meetrec takes a different
approach:

- **Profile- and browser-agnostic** — it watches the Windows microphone
  ConsentStore, so it works with any app that uses the mic, in any browser
  profile, without touching the meeting itself.
- **Audio never leaves your machine** — recording, transcription (Whisper),
  diarization and voiceprints are all local. Only the finished summary text
  goes to an LLM backend (and even that can be a local Ollama model).
- **Set and forget** — a daemon with debounced start/stop detection, crash-safe
  incremental recording, a disk-backed delivery queue, and a mandatory
  "recording started" notification so you are never recorded silently.

## How it works

```
        ┌──────────────────────────── daemon ────────────────────────────┐
        │                                                                │
 mic    │  IDLE ──15s continuous mic capture──▶ RECORDING                │
 registry│   ▲                                      │                    │
 polling │   └──────30s without capture─────────────┘                    │
        │                                            ▼                   │
        │                    PROCESSING ──▶ DELIVERING                   │
        └────────────────────────────────────────────────────────────────┘

RECORDING   track_mic.wav  (your microphone)
            track_sys.wav  (WASAPI loopback of the default output = everyone else)

PROCESSING  faster-whisper (large-v3-turbo) on both tracks, separately
            pyannote diarization on track_sys only (you are already isolated
              on the mic track, labeled with `diarization.self_label` — "ME" by default)
            speaker embeddings → matched against a persistent voiceprint DB
            chronological merge → transcript documents
            LLM summary in the meeting's language

DELIVERING  session folder + Telegram (summary; transcript only if opted in)
            Windows toasts with action buttons
```

**Detection detail:** Windows records every app's microphone usage under
`HKCU/HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\microphone`
(packaged apps) and `...\microphone\NonPackaged` (win32 apps such as Chrome,
with `#` in place of path backslashes). `LastUsedTimeStop == 0` means "this
app is capturing right now". meetrec polls this every 2 s and debounces:
recording starts only after 15 s of continuous capture and stops only after
30 s without any — so a quick mic check doesn't trigger a session, and a
network blip doesn't split one meeting into two.

## What you get

Each meeting produces a folder:

```
~/Meetings/2026-08-20_1400/
├── audio.flac         # stereo mix: your mic on the left, everyone else on the right
├── transcript.txt     # [HH:MM:SS] Speaker: text — one block per speaking turn
├── transcript.md      # same, formatted for reading
├── summary.md         # executive summary, decisions, action items with owners,
│                      #   open questions — in the meeting's language
├── meta.json          # duration, language + confidence, speakers, models used
└── (intermediates)    # transcript_*.json, diarization.json, embeddings.npz,
                       #   speaker_map.json — enable reprocess/label without
                       #   re-running everything
```

And on Telegram: the summary as one or more messages (split under the
4096-char API limit), plus `transcript.txt` as a document **only** if you
explicitly enable `telegram.send_full_transcript`.

## Minimum requirements

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| OS | Windows 11 | Windows 11 | Windows 10 21H2+ likely works (same ConsentStore/WASAPI), untested |
| Python | 3.11 | 3.11+ | |
| RAM | 8 GB | 16 GB+ | Whisper large-v3-turbo int8 uses ~2–3 GB; pyannote ~2 GB; peaks overlap |
| Disk | 10 GB free | 20 GB+ | ~4 GB of models on first run + recordings (~120 MB/h of WAV, mixed down to FLAC after) |
| CPU | 4 cores | 8+ cores | CPU-only transcription of a 1 h meeting ≈ 30–90 min (see [Performance](#performance-expectations)) |
| GPU | none | NVIDIA with 6 GB+ VRAM and CUDA 12 | Auto-detected; transcription becomes ~10× faster |
| Audio | any mic + output device | | WASAPI loopback needs no drivers or virtual cables |
| Network | only for delivery/summary | | Recording and transcription work fully offline |

## Installation

One-shot installer — creates the venv, installs everything, then walks you
through keys and preferences interactively (`meetrec setup`), including
detecting your Telegram chat id automatically:

```powershell
git clone https://github.com/pedrojoelrcosta-jpg/meetrec
cd meetrec
powershell -ExecutionPolicy Bypass -File install.ps1
```

Manual route, if you prefer:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\pip install -e .
.venv\Scripts\meetrec setup      # or: copy .env.example .env and edit it
.venv\Scripts\meetrec doctor
```

> **Long-path warning:** torch's package layout can exceed Windows' 260-char
> `MAX_PATH` if your clone lives in a deep directory. Either enable long
> paths (`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled=1`
> as admin) or create the venv somewhere short, e.g.
> `py -3.11 -m venv C:\venvs\meetrec`.

`meetrec doctor` validates everything: Python dependencies, registry access,
audio devices (mic + WASAPI loopback), CUDA availability, HuggingFace token,
summary backend, and Telegram connectivity. Fix whatever it flags before the
first real meeting.

## First run: validate detection on your machine

Before trusting the daemon, verify the registry mechanism works for you —
it takes two minutes:

```powershell
py diagnose_mic.py --watch
```

Join any meeting with the microphone on. You should see:

```
[13:53:20] STARTED capturing: C:\Program Files\Google\Chrome\Application\chrome.exe  [HKCU]
[13:53:44] STOPPED capturing: C:\Program Files\Google\Chrome\Application\chrome.exe  [HKCU]
```

`py diagnose_mic.py` (without `--watch`) prints a snapshot of every app that
ever used the mic, with timestamps. If the right executable shows up as
ACTIVE during a meeting and nothing shows up outside one, detection will
work.

## API keys and .env

Copy `.env.example` to `.env` and fill in what you need. **`.env` is
gitignored and must never be committed** — only `.env.example` (with empty
placeholders) belongs in the repository.

| Variable | Needed for | How to get it |
|---|---|---|
| `HF_TOKEN` | Diarization (who spoke when) | Free. Accept the model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) **and** [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0), then create a *read* token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Without it, meetrec still records and transcribes — everyone on the system track just appears as a single unlabeled speaker. |
| `GEMINI_API_KEY` | Summaries (default backend) | Free, no credit card: [aistudio.google.com/apikey](https://aistudio.google.com/apikey) → "Create API key". Free keys get rate-limited (HTTP 429) when busy; meetrec retries up to 6 times with exponential backoff and then falls back to Ollama/Anthropic automatically. A summary failure never loses the transcript. |
| `TELEGRAM_BOT_TOKEN` | Telegram delivery | Talk to [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token. |
| `TELEGRAM_CHAT_ID` | Telegram delivery | Send any message to your new bot, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` in a browser and read `chat.id` from the JSON. |
| `ANTHROPIC_API_KEY` | Optional summary backend | [console.anthropic.com](https://console.anthropic.com) |

Test the Telegram pair with `meetrec test-telegram`.

## Usage

```
meetrec start            # run the daemon (foreground)
meetrec stop             # stop a running daemon
meetrec status           # daemon state, current recording, recent log lines
meetrec pause            # toggle: skip new meetings until toggled back
meetrec pause --for 2h   # pause with automatic resume (30m, 2h, 1h30m, ...)
meetrec record           # record manually right now (Enter stops) — for
                         #   in-person meetings or apps detection misses
meetrec list             # all sessions with duration/language/state/speakers
meetrec summary <dir>    # regenerate just the summary (--backend, --language,
                         #   --resend to push the new one to Telegram)
meetrec reprocess <dir>  # re-run processing on a session (--full = from scratch)
meetrec label <dir>      # play excerpts of unknown speakers and name them
meetrec speakers         # list known voices (--rename OLD NEW | --delete NAME)
meetrec cleanup          # expire old audio per output.audio_retention_h
                         #   (--dry-run previews; the daemon also runs this)
meetrec setup            # re-run the interactive wizard anytime
meetrec doctor           # validate the whole setup
meetrec debug <dir>      # session x-ray: stage timings, recorded issues,
                         #   capture errors (--traceback for full detail)
meetrec test-telegram    # send a test message to your chat
meetrec autostart on     # create a Task Scheduler job that starts meetrec at
                         #   logon (meetrec autostart off removes it)
```

`<dir>` accepts either the session folder name (`2026-08-20_1400`) or a full
path.

Notifications you will see: recording started (**mandatory, not
disableable** — you should always know when you are being recorded),
recording finished, processing done (button opens the folder), unidentified
speakers (points you to `meetrec label`), and errors.

## Speaker identification workflow

1. Your own voice never goes through diarization — the mic track is isolated
   by construction and labeled **ME** (configurable via
   `diarization.self_label` — e.g. `EU` for Portuguese transcripts).
2. The system track is diarized with pyannote; each speaker gets an
   embedding (voiceprint) averaged over their longest speaking turns.
3. Each embedding is compared, by cosine similarity, against the persistent
   voiceprint database (`%LOCALAPPDATA%\meetrec\voiceprints.db`). Above the
   threshold (`diarization.similarity_threshold`, default 0.75, tunable) the
   known name is used; otherwise the speaker stays `SPEAKER_00`, `SPEAKER_01`…
4. After the meeting, run `meetrec label <session>`: meetrec plays up to 3
   audio excerpts per unknown speaker, asks for a name, stores the voiceprint,
   and regenerates the transcript with the new names. From then on that
   person is recognized automatically in every future meeting.
5. Voiceprints improve over time: labeling the same name again folds the new
   embedding into a running mean. Manage them with `meetrec speakers`.

If recognition is too eager (wrong names), raise the threshold to 0.8–0.85;
if familiar people keep coming out as `SPEAKER_XX`, lower it toward 0.65.

## Configuration reference

Everything lives in [`config.yaml`](config.yaml); every key has a default in
code, so a missing key just falls back.

| Key | Default | Meaning |
|---|---|---|
| `detector.poll_interval_s` | `2` | How often the registry is polled |
| `detector.start_debounce_s` | `15` | Continuous capture required before recording starts |
| `detector.stop_debounce_s` | `30` | Capture-free time required before recording stops |
| `detector.ignore_apps` | sound recorder, Xbox overlay | Executables/app-ids that never trigger recording (case-insensitive) |
| `audio.min_session_s` | `60` | Sessions shorter than this are discarded entirely |
| `audio.echo_dedup` | `true` | When you listen on **speakers**, your mic also captures the other participants; mic segments that overlap a system segment in time with near-identical text are dropped as echo instead of being misattributed to you. Harmless when using headphones |
| `transcription.model` | `large-v3-turbo` | Any faster-whisper model name (`medium`, `small`… for weaker CPUs) |
| `transcription.device` | `auto` | `auto` picks CUDA when available, else CPU |
| `transcription.compute_type` | `auto` | `auto` = float16 on CUDA, int8 on CPU |
| `transcription.multilingual` | `true` | Re-detect the language per segment, so meetings that switch languages mid-call (PT ↔ EN) transcribe each part correctly. Small speed cost; turn off for strictly single-language meetings |
| `diarization.similarity_threshold` | `0.75` | Cosine threshold for voiceprint recognition |
| `summary.backend` | `gemini` | `gemini` \| `ollama` \| `anthropic` — the others act as fallbacks |
| `summary.language` | `auto` | Language of the summary (file **and** Telegram message): `auto` = the language detected in the meeting; `pt` or `en` to force one — useful when meetings mix languages and you always want the summary in a single one |
| `summary.gemini_model` | `gemini-2.0-flash` | Model for the Gemini backend |
| `summary.ollama_model` | `gemma3:4b` | Local model — small, strong multilingual, CPU-friendly |
| `summary.ollama_url` | `http://localhost:11434` | Ollama endpoint |
| `output.dir` | `~/Meetings` | Where session folders are created |
| `diarization.self_label` | `ME` | Label for your own (mic-track) speech blocks |
| `output.keep_wav` | `false` | Keep the raw per-track WAVs after the FLAC mix (they cost ~120 MB/h; `label` works from the FLAC either way) |
| `output.audio_retention_h` | `0` | `0` keeps `audio.flac` forever; e.g. `72` deletes the audio 72 h after processing while keeping transcripts and summary. The daemon sweeps every 30 min; `meetrec cleanup --dry-run` previews. Label unknown speakers before the audio expires |
| `notifications.*` | all `true` | Per-event toast toggles (`recording_stopped`, `processing_done`, `speakers_unlabeled`, `errors`). The recording-start toast is not configurable by design |
| `debug.level` | `info` | `debug` for verbose logs (or `--debug` on any command) |
| `debug.strict` | `false` | `true` makes pipeline stage errors raise immediately instead of being recorded in the session's `debug.json` and skipped — use while debugging |
| `telegram.enabled` | `true` | Master switch for Telegram delivery |
| `telegram.send_full_transcript` | `false` | The transcript only leaves the machine if you set this to `true` |

## Performance expectations

Transcription dominates processing time. Rough figures for a 1-hour meeting
with `large-v3-turbo`:

| Hardware | Transcription time |
|---|---|
| NVIDIA GPU (CUDA, float16) | ~3–6 min |
| Modern 8-core CPU (int8) | ~30–60 min |
| Older 4-core CPU (int8) | ~60–120 min |

On CPU-only machines meetrec warns about this at model load. If it is too
slow for you, set `transcription.model: medium` (or `small`) — quality drops
modestly, speed improves a lot. Diarization adds roughly 5–15 min per meeting
hour on CPU. Processing runs in the background and a new meeting can start
recording while the previous one is still being processed.

First run downloads models: Whisper large-v3-turbo (~1.6 GB) and the
pyannote models (~1 GB), cached under your user profile afterwards.

## Privacy model

| Data | Where it goes |
|---|---|
| Audio (both tracks) | Never leaves the machine |
| Transcription | Local (faster-whisper) |
| Diarization + voiceprints | Local (pyannote + SQLite in `%LOCALAPPDATA%\meetrec`) |
| Transcript text | Sent to the summary LLM backend¹; sent to Telegram **only** if `send_full_transcript: true` |
| Summary text | Sent to Telegram (if enabled) |

¹ With `summary.backend: ollama` the summary is generated locally too, and
nothing but the Telegram delivery touches the network. With Gemini/Anthropic
the transcript text is submitted to that API for summarization — choose the
backend according to how sensitive your meetings are.

**Legal note:** depending on your jurisdiction, recording calls may require
the consent of the other participants. You are responsible for complying
with the laws that apply to you. The non-disableable recording notification
exists so *you* are always aware; it does not notify the other side.

## Troubleshooting

- **`WinError 206` / "filename or extension too long" during install** —
  Windows `MAX_PATH` limit. Enable long paths (admin) or use a venv at a
  short path (`C:\venvs\meetrec`). See the note in [Installation](#installation).
- **`diagnose_mic.py` shows nothing during a meeting** — check the HKLM
  branch too (some setups write there), and confirm the meeting app really
  has mic permission (Windows Settings → Privacy → Microphone).
- **Diarization fails with a token message** — accept the terms on *both*
  HuggingFace model pages (diarization *and* segmentation) with the same
  account that owns `HF_TOKEN`.
- **Gemini keeps failing with 429** — the free tier is congested; meetrec
  already retries and falls back to Ollama. Install Ollama and
  `ollama pull gemma3:4b` to always have the local fallback.
- **No toasts appear** — Windows Focus Assist / Do Not Disturb suppresses
  them; meetrec still logs and delivers to Telegram.
- **Telegram messages never arrive** — run `meetrec test-telegram`. Failed
  deliveries are queued in `%LOCALAPPDATA%\meetrec\telegram_queue` and
  retried on the next processing run or daemon start; nothing is lost.
- **Recording survived a crash?** — tracks are written incrementally, so the
  audio up to the crash moment is on disk. The daemon re-queues any
  recorded-but-unprocessed session automatically at startup; you can also
  run `meetrec reprocess <dir>` yourself.

## Development

```powershell
.venv\Scripts\python -m pytest tests/
```

The detector state machine is tested with an injected fake scanner and fake
clock, and voiceprint matching with synthetic embedding vectors — no real
meetings (or GPUs) needed for the logic tests. `diagnose_mic.py` is
stdlib-only on purpose, so detection can be validated before installing
anything. CI runs the logic tests on `windows-latest` — the heavy ML
dependencies are imported lazily, so the suite needs only the light ones.

Project layout:

```
meetrec/
├── diagnose_mic.py        # standalone detection validator (stdlib only)
├── config.yaml
├── .env.example
└── meetrec/
    ├── registry_scan.py   # ConsentStore reader (the only Windows-registry boundary)
    ├── detector.py        # debounced state machine (pure, injectable)
    ├── audio_capture.py   # dual-track WASAPI recorder
    ├── transcribe.py      # faster-whisper wrapper
    ├── diarize.py         # pyannote diarization + speaker embeddings
    ├── voiceprints.py     # persistent voiceprint DB (SQLite + cosine)
    ├── merge.py           # chronological merge of both tracks
    ├── summarize.py       # Gemini / Ollama / Anthropic with fallback chain
    ├── telegram.py        # delivery, message splitting, disk-backed retry queue
    ├── notify.py          # Windows 11 toasts
    ├── label.py           # interactive speaker labeling
    ├── pipeline.py        # processing orchestration
    ├── daemon.py          # detector → recorder → pipeline wiring
    └── cli.py             # the `meetrec` command
```

## License

[MIT](LICENSE)
