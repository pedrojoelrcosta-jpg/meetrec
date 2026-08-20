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
from .config import (SUMMARY_MD, TRANSCRIPT_MD, TRANSCRIPT_TXT, data_dir,
                     load_config)
from .diarize import (HFTokenMissing, assign_speakers_to_segments,
                      diarize_track, speaker_embeddings)
from .merge import drop_mic_echo, merge_tracks, to_markdown, to_plain_text
from .summarize import summarize
from .transcribe import transcribe_track
from .voiceprints import VoiceprintDB

log = logging.getLogger(__name__)


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _restore_tracks_from_flac(session_dir: Path) -> None:
    """Rebuild track_mic.wav / track_sys.wav from audio.flac channels
    (mic=left, system=right). Makes `reprocess --full` work on sessions
    whose raw WAVs were cleaned up (output.keep_wav=false)."""
    import soundfile as sf

    flac = session_dir / "audio.flac"
    if not flac.exists():
        return
    targets = [(session_dir / "track_mic.wav", 0),
               (session_dir / "track_sys.wav", 1)]
    if all(path.exists() for path, _ in targets):
        return
    data, rate = sf.read(flac, dtype="int16", always_2d=True)
    for path, channel in targets:
        if not path.exists() and data.shape[1] > channel:
            sf.write(path, data[:, channel], rate, subtype="PCM_16")
            log.info("Restored %s from audio.flac", path.name)


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


def regenerate_transcripts(session_dir: Path,
                           self_label: str | None = None,
                           echo_dedup: bool | None = None) -> dict:
    """Rebuild transcript.txt/.md (and meta speakers) from saved
    intermediates, applying speaker_map.json. Used by process and label."""
    if self_label is None or echo_dedup is None:
        cfg = load_config()
        if self_label is None:
            self_label = cfg["diarization"].get("self_label", "ME")
        if echo_dedup is None:
            echo_dedup = cfg["audio"].get("echo_dedup", True)
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

    mic_segments = mic["segments"]
    if echo_dedup:
        before = len(mic_segments)
        mic_segments = drop_mic_echo(mic_segments, sys_tr["segments"])
        if len(mic_segments) < before:
            log.info("Echo dedup: dropped %d mic segment(s) that duplicated "
                     "system audio (speaker bleed)",
                     before - len(mic_segments))
    blocks = merge_tracks(mic_segments, sys_tr["segments"],
                          self_label=self_label)
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

    (session_dir / TRANSCRIPT_TXT).write_text(
        to_plain_text(blocks), encoding="utf-8")
    (session_dir / TRANSCRIPT_MD).write_text(
        to_markdown(blocks, meta), encoding="utf-8")
    return {"blocks": blocks, "meta": meta}


def process_session(cfg: dict, session_dir: Path,
                    with_notifications: bool = True) -> None:
    """Full processing of a recorded session directory.

    Every stage is timed and its outcome persisted to debug.json (see
    debuglog). Non-fatal failures never lose the transcript, but they are
    never silent either: they count as issues, show up in `meetrec list`
    and `meetrec debug`, and change the final notification.
    """
    from .debuglog import StageRecorder

    started = time.time()
    log.info("Processing %s", session_dir)
    rec = StageRecorder(session_dir,
                        strict=cfg.get("debug", {}).get("strict", False))

    mic_wav = session_dir / "track_mic.wav"
    sys_wav = session_dir / "track_sys.wav"

    # 0. if the raw WAVs were cleaned up but a stage needs them again
    # (reprocess --full), rebuild them from the FLAC channels
    if not (mic_wav.exists() and sys_wav.exists()):
        with rec.stage("restore_tracks"):
            _restore_tracks_from_flac(session_dir)

    # 1. transcription (both tracks, separately). Fatal: without it there
    # is nothing downstream to work with.
    for wav, out_name in ((mic_wav, "transcript_mic.json"),
                          (sys_wav, "transcript_sys.json")):
        out = session_dir / out_name
        stage_name = f"transcribe_{wav.stem.split('_')[-1]}"
        if not wav.exists() and not out.exists():
            rec.skip(stage_name, f"{wav.name} not found")
            continue
        if out.exists():
            rec.skip(stage_name, "already transcribed")
            continue
        with rec.stage(stage_name, fatal=True):
            result = transcribe_track(cfg, wav)
            _write_json(out, result)
            rec.note(stage_name, language=result["language"],
                     segments=len(result["segments"]))

    # 2. diarization + speaker identity (system track only; mic is MYSELF)
    unknown_count = 0
    if not sys_wav.exists() and not (session_dir / "diarization.json").exists():
        rec.skip("diarization", "track_sys.wav not found")
    else:
        with rec.stage("diarization"):
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
                rec.note("diarization", speakers=len(all_speakers),
                         recognized=len(mapping))
            except HFTokenMissing:
                if with_notifications:
                    notify.error("Diarization skipped: HF_TOKEN not configured")
                raise

    # 3. chronological merge + output documents (fatal: this IS the product)
    with rec.stage("merge", fatal=True):
        result = regenerate_transcripts(
            session_dir,
            self_label=cfg["diarization"].get("self_label", "ME"),
            echo_dedup=cfg["audio"].get("echo_dedup", True))
        meta = result["meta"]

    # 4. FLAC with both tracks (mic=left, system=right)
    if (session_dir / "audio.flac").exists():
        rec.skip("flac_mix", "audio.flac already exists")
    else:
        with rec.stage("flac_mix"):
            _mix_to_flac(session_dir)

    # 5. summary in the detected (or forced) language
    summary_text = None
    with rec.stage("summary"):
        transcript_text = (session_dir / TRANSCRIPT_TXT) \
            .read_text(encoding="utf-8")
        lang_cfg = cfg["summary"].get("language", "auto")
        summary_lang = meta["language"] if lang_cfg == "auto" else lang_cfg
        summary_text = summarize(cfg, summary_lang, transcript_text)
        (session_dir / SUMMARY_MD).write_text(summary_text, encoding="utf-8")
        rec.note("summary", language=summary_lang)

    # 6. delivery
    with rec.stage("telegram_delivery"):
        telegram.flush_queue()
        if summary_text is None:
            rec.skip("telegram_delivery", "no summary to send")
        else:
            delivered = telegram.deliver(cfg, session_dir, summary_text,
                                         f"Meeting {session_dir.name}")
            rec.note("telegram_delivery",
                     delivered="now" if delivered else "queued for retry")

    # 7. optional cleanup of the raw WAV tracks (only when the FLAC and the
    # transcripts are safely on disk; `label` falls back to audio.flac)
    if not cfg["output"].get("keep_wav", False):
        if ((session_dir / "audio.flac").exists()
                and (session_dir / TRANSCRIPT_TXT).exists()):
            for name in ("track_mic.wav", "track_sys.wav"):
                (session_dir / name).unlink(missing_ok=True)
            log.info("Raw WAV tracks removed (output.keep_wav=false)")

    meta["processing_s"] = round(time.time() - started, 1)
    meta["whisper_model"] = cfg["transcription"]["model"]
    meta["summary_backend"] = cfg["summary"]["backend"]
    meta["issues"] = rec.issue_count
    _write_json(session_dir / "meta.json", meta)

    notif_cfg = cfg.get("notifications", {})
    if with_notifications:
        if notif_cfg.get("processing_done", True):
            notify.processing_done(session_dir, issues=rec.issue_count)
        if unknown_count and notif_cfg.get("speakers_unlabeled", True):
            notify.speakers_unlabeled(session_dir, unknown_count)
    log.info("Done in %.0fs (%d issue(s))",
             meta["processing_s"], rec.issue_count)


AUDIO_FILES = ("audio.flac", "track_mic.wav", "track_sys.wav")


def cleanup_expired_audio(cfg: dict, dry_run: bool = False) -> list[Path]:
    """Delete audio files from sessions processed more than
    output.audio_retention_h hours ago. Transcripts, summary, metadata and
    label excerpts are never touched. Returns the files (that would be)
    deleted."""
    retention_h = float(cfg["output"].get("audio_retention_h", 0) or 0)
    if retention_h <= 0:
        return []
    cutoff = time.time() - retention_h * 3600
    out_dir = Path(cfg["output"]["dir"])
    if not out_dir.exists():
        return []
    removed: list[Path] = []
    for session in out_dir.iterdir():
        if not session.is_dir():
            continue
        # only expire sessions that are fully processed — never delete the
        # sole copy of an untranscribed recording
        from .config import find_transcript
        if find_transcript(session) is None:
            continue
        for name in AUDIO_FILES:
            path = session / name
            if path.exists() and path.stat().st_mtime < cutoff:
                removed.append(path)
                if not dry_run:
                    path.unlink()
    if removed and not dry_run:
        log.info("Audio retention: deleted %d file(s)", len(removed))
    return removed


def new_session_dir(cfg: dict) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    return Path(cfg["output"]["dir"]) / stamp
