"""Local transcription with faster-whisper.

- language=None: auto-detect (PT-PT and EN meetings, sometimes mixed)
- vad_filter=True: mandatory — Whisper hallucinates on long silences
- word_timestamps=True: needed for the chronological merge
"""

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_model_cache: dict = {}


def resolve_device(cfg: dict) -> tuple[str, str]:
    """(device, compute_type) honoring 'auto' in config."""
    device = cfg["transcription"]["device"]
    compute = cfg["transcription"]["compute_type"]
    if device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def get_model(cfg: dict):
    from faster_whisper import WhisperModel

    device, compute = resolve_device(cfg)
    key = (cfg["transcription"]["model"], device, compute)
    if key not in _model_cache:
        if device == "cpu":
            log.warning(
                "No CUDA GPU detected — transcribing on CPU with int8. "
                "Expect roughly 1-2x the meeting duration for large-v3-turbo.")
        log.info("Loading Whisper model %s (%s/%s)", *key)
        _model_cache[key] = WhisperModel(
            cfg["transcription"]["model"], device=device, compute_type=compute)
    return _model_cache[key]


def transcribe_track(cfg: dict, wav_path: Path) -> dict:
    """Transcribe one track. Returns {language, language_probability, segments}.

    Each segment: {start, end, text, words: [{start, end, word}]}.
    """
    model = get_model(cfg)
    segments_iter, info = model.transcribe(
        str(wav_path),
        language=None,
        vad_filter=True,
        word_timestamps=True,
    )
    segments = []
    for seg in segments_iter:
        segments.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "words": [
                {"start": round(w.start, 2), "end": round(w.end, 2),
                 "word": w.word}
                for w in (seg.words or [])
            ],
        })
    log.info("%s: %d segments, language=%s (p=%.2f)",
             wav_path.name, len(segments), info.language,
             info.language_probability)
    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "segments": segments,
    }
