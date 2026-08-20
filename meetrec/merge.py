"""Chronological merge of the mic and system tracks into one document.

The mic track is labeled with `self_label` (the user — configurable, e.g.
"ME" or "EU"). System-track segments carry the speaker assigned by
diarization/voiceprint matching. Consecutive blocks from the same speaker
are merged when the gap is small.
"""

MERGE_GAP_S = 2.0
DEFAULT_SELF_LABEL = "ME"

# Echo dedup: when the user listens on speakers, the mic also captures the
# other participants' voices, so the mic track duplicates their speech
# (attributed to the user). A mic segment that overlaps a system segment in
# time AND says nearly the same thing is echo, not the user speaking.
ECHO_WINDOW_SLACK_S = 1.0     # Whisper timestamps are rough; widen the match
ECHO_OVERLAP_MIN = 0.3        # fraction of the mic segment inside the window
ECHO_SIMILARITY_MIN = 0.65    # echo is muffled — transcripts differ slightly


def _normalize(text: str) -> str:
    return "".join(c for c in text.casefold() if c.isalnum() or c.isspace())


def _similarity(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def drop_mic_echo(mic_segments: list[dict],
                  sys_segments: list[dict]) -> list[dict]:
    """Remove mic segments that are speaker-bleed echoes of system audio."""
    kept = []
    for mic in mic_segments:
        duration = max(mic["end"] - mic["start"], 0.1)
        is_echo = False
        for sys in sys_segments:
            overlap = (min(mic["end"], sys["end"] + ECHO_WINDOW_SLACK_S)
                       - max(mic["start"], sys["start"] - ECHO_WINDOW_SLACK_S))
            if overlap / duration < ECHO_OVERLAP_MIN:
                continue
            if _similarity(mic["text"], sys["text"]) >= ECHO_SIMILARITY_MIN:
                is_echo = True
                break
        if not is_echo:
            kept.append(mic)
    return kept


def _fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def merge_tracks(mic_segments: list[dict], sys_segments: list[dict],
                 self_label: str = DEFAULT_SELF_LABEL) -> list[dict]:
    """Return blocks [{speaker, start, end, text}] sorted by start time."""
    entries = []
    for seg in mic_segments:
        if seg["text"]:
            entries.append({"speaker": self_label, "start": seg["start"],
                            "end": seg["end"], "text": seg["text"]})
    for seg in sys_segments:
        if seg["text"]:
            entries.append({"speaker": seg.get("speaker", "SPEAKER_??"),
                            "start": seg["start"], "end": seg["end"],
                            "text": seg["text"]})
    entries.sort(key=lambda e: e["start"])

    blocks: list[dict] = []
    for entry in entries:
        if (blocks
                and blocks[-1]["speaker"] == entry["speaker"]
                and entry["start"] - blocks[-1]["end"] <= MERGE_GAP_S):
            blocks[-1]["text"] += " " + entry["text"]
            blocks[-1]["end"] = max(blocks[-1]["end"], entry["end"])
        else:
            blocks.append(dict(entry))
    return blocks


def to_plain_text(blocks: list[dict]) -> str:
    """One block per line: [HH:MM:SS] Speaker: text"""
    return "\n".join(
        f"[{_fmt_ts(b['start'])}] {b['speaker']}: {b['text']}" for b in blocks
    ) + "\n"


def to_markdown(blocks: list[dict], meta: dict) -> str:
    lines = [f"# Meeting transcript — {meta.get('date', '')}", ""]
    duration = meta.get("duration_s")
    if duration:
        lines.append(f"- **Duration:** {_fmt_ts(duration)}")
    if meta.get("language"):
        lines.append(f"- **Language:** {meta['language']}")
    if meta.get("speakers"):
        lines.append(f"- **Speakers:** {', '.join(meta['speakers'])}")
    lines.append("")
    for b in blocks:
        lines.append(f"**`[{_fmt_ts(b['start'])}]` {b['speaker']}:** {b['text']}")
        lines.append("")
    return "\n".join(lines)
