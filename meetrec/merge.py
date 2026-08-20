"""Chronological merge of the mic and system tracks into one document.

The mic track is labeled with `self_label` (the user — configurable, e.g.
"ME" or "EU"). System-track segments carry the speaker assigned by
diarization/voiceprint matching. Consecutive blocks from the same speaker
are merged when the gap is small.
"""

MERGE_GAP_S = 2.0
DEFAULT_SELF_LABEL = "ME"


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
