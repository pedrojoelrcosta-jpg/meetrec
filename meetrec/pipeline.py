"""Post-meeting processing: transcribe → diarize → identify → merge →
summarize → deliver.

All intermediates are saved to the session directory so `meetrec reprocess`
and `meetrec label` can rerun any stage without the original meeting.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import notify, telegram
from .config import data_dir
from .diarize import (HFTokenMissing, assign_speakers_to_segments,
                      diarize_track, speaker_embeddings)
from .merge import merge_tracks, to_markdown, to_plain_text
from .summarize import summarize
from .transcribe import transcribe_track
from .voiceprints import VoiceprintDB

log = logging.getLogger(__name__)

MYSELF = "EU"


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _mix_to_flac(session_dir: Path) -> None:
    """Mix both tracks into one stereo FLAC: mic on the left, system on the
    right. Keeps the originals' timeline; resamples to the higher rate."""
    import soundfile as sf

    mic_path = session_dir / "track_mic.wav"
    sys_path = session_dir / "track_sys.wav"
    tracks = []
    rates = []
    for path in (mic_path, sys_path):
        if path.exists():
            data, rate = sf.read(path, dtype="float32", always_2d=True)
            tracks.append(data.mean(axis=1))  # mono
            rates.append(rate)
    if not tracks:
        return
    target_rate = max(rates)
    resampled = []
    for data, rate in zip(tracks, rates):
        if rate != target_rate:
            n_out = int(len(data) * target_rate / rate)
            idx = np.linspace(0, len(data) - 1, n_out)
            data = data[idx.round().astype(int)]
        resampled.append(data)
    length = max(len(t) for t in resampled)
    stereo = np.zeros((length, 2), dtype=np.float32)
    stereo[:len(resampled[0]), 0] = resampled[0]
    if len(resampled) > 1:
        stereo[:len(resampled[1]), 1] = resampled[1]
    sf.write(session_dir / "audio.flac", stereo, target_rate)


def _identify_speakers(cfg: dict, session_dir: Path,
                       turns: list[dict]) -> dict[str, str]:
    """Map diarized labels (SPEAKER_00…) to known names via the voiceprint DB.

    Saves per-speaker embeddings to embeddings.npz for `meetrec label`.
    """
    sys_wav = session_dir / "track_sys.wav"
    embeddings = speaker_embeddings(sys_wav, turns)
    if embeddings:
        np.savez(session_dir / "embeddings.npz", **embeddings)

    db = VoiceprintDB(data_dir() / "voiceprints.db")
    threshold = cfg["diarization"]["similarity_threshold"]
    mapping: dict[str, str] = {}
    try:
        for speaker, emb in embeddings.items():
            match = db.match(emb, threshold)
            if match:
                mapping[speaker] = match[0]
                log.info("%s recognized as %s (%.3f)", speaker, *match)
    finally:
        db.close()
    return mapping


def regenerate_transcripts(session_dir: Path) -> dict:
    """Rebuild transcricao.txt/.md (and meta speakers) from saved
    intermediates, applying speaker_map.json. Used by process and label."""
    mic = _read_json(session_dir / "transcript_mic.json") \
        if (session_dir / "transcript_mic.json").exists() \
        else {"segments": [], "language": None, "language_probability": 0}
    sys_tr = _read_json(session_dir / "transcript_sys.json") \
        if (session_dir / "transcript_sys.json").exists() \
        else {"segments": [], "language": None, "language_probability": 0}
    speaker_map = {}
    if (session_dir / "speaker_map.json").exists():
        speaker_map = _read_json(session_dir / "speaker_map.json")

    for seg in sys_tr["segments"]:
        raw = seg.get("speaker", "SPEAKER_??")
        seg["speaker"] = speaker_map.get(raw, raw)

    blocks = merge_tracks(mic["segments"], sys_tr["segments"])
    speakers = sorted({b["speaker"] for b in blocks})
    language = sys_tr["language"] or mic["language"] or "en"

    meta_path = session_dir / "meta.json"
    meta = _read_json(meta_path) if meta_path.exists() else {}
    meta.update({
        "date": meta.get("date") or session_dir.name,
        "language": language,
        "language_probability": max(mic.get("language_probability") or 0,
                                    sys_tr.get("language_probability") or 0),
        "speakers": speakers,
    })
    _write_json(meta_path, meta)

    (session_dir / "transcricao.txt").write_text(
        to_plain_text(blocks), encoding="utf-8")
    (session_dir / "transcricao.md").write_text(
        to_markdown(blocks, meta), encoding="utf-8")
    return {"blocks": blocks, "meta": meta}


def process_session(cfg: dict, session_dir: Path,
                    with_notifications: bool = True) -> None:
    """Full processing of a recorded session directory."""
    started = time.time()
    log.info("Processing %s", session_dir)

    mic_wav = session_dir / "track_mic.wav"
    sys_wav = session_dir / "track_sys.wav"

    # 1. transcription (both tracks, separately)
    for wav, out_name in ((mic_wav, "transcript_mic.json"),
                          (sys_wav, "transcript_sys.json")):
        out = session_dir / out_name
        if wav.exists() and not out.exists():
            _write_json(out, transcribe_track(cfg, wav))

    # 2. diarization + speaker identity (system track only; mic is MYSELF)
    unknown_count = 0
    if sys_wav.exists():
        try:
            diar_path = session_dir / "diarization.json"
            if not diar_path.exists():
                _write_json(diar_path, diarize_track(sys_wav))
            turns = _read_json(diar_path)

            sys_tr = _read_json(session_dir / "transcript_sys.json")
            assign_speakers_to_segments(sys_tr["segments"], turns)
            _write_json(session_dir / "transcript_sys.json", sys_tr)

            mapping = _identify_speakers(cfg, session_dir, turns)
            _write_json(session_dir / "speaker_map.json", mapping)
            all_speakers = {t["speaker"] for t in turns}
            unknown_count = len(all_speakers - set(mapping))
        except HFTokenMissing as exc:
            log.error("%s", exc)
            if with_notifications:
                notify.error("Diarization skipped: HF_TOKEN not configured")
        except Exception:
            log.exception("Diarization failed; continuing without speakers")

    # 3. chronological merge + output documents
    result = regenerate_transcripts(session_dir)
    meta = result["meta"]

    # 4. FLAC with both tracks (mic=left, system=right)
    try:
        if not (session_dir / "audio.flac").exists():
            _mix_to_flac(session_dir)
    except Exception:
        log.exception("FLAC mix failed; WAV tracks kept")

    # 5. summary in the detected language
    summary_text = None
    try:
        transcript_text = (session_dir / "transcricao.txt") \
            .read_text(encoding="utf-8")
        # summary.language: auto = language detected in the meeting; pt/en force
        lang_cfg = cfg["summary"].get("language", "auto")
        summary_lang = meta["language"] if lang_cfg == "auto" else lang_cfg
        summary_text = summarize(cfg, summary_lang, transcript_text)
        (session_dir / "resumo.md").write_text(summary_text, encoding="utf-8")
    except Exception:
        log.exception("Summary failed; transcript is intact")
        if with_notifications:
            notify.error("Summary failed — transcript saved")

    meta["processing_s"] = round(time.time() - started, 1)
    meta["whisper_model"] = cfg["transcription"]["model"]
    meta["summary_backend"] = cfg["summary"]["backend"]
    _write_json(session_dir / "meta.json", meta)

    # 6. delivery
    telegram.flush_queue()
    if summary_text:
        try:
            telegram.deliver(cfg, session_dir, summary_text,
                             f"Meeting {session_dir.name}")
        except telegram.TelegramNotConfigured as exc:
            log.warning("%s", exc)

    if with_notifications:
        notify.processing_done(session_dir)
        if unknown_count:
            notify.speakers_unlabeled(session_dir, unknown_count)
    log.info("Done in %.0fs", meta["processing_s"])


def new_session_dir(cfg: dict) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    return Path(cfg["output"]["dir"]) / stamp
