"""Diarization of the system track with pyannote.audio + speaker embeddings.

Only track_sys.wav is diarized — the user's voice is already isolated on the
microphone track and is labeled "EU" directly.

Requires accepting the model terms on HuggingFace and setting HF_TOKEN:
  1. https://huggingface.co/pyannote/speaker-diarization-3.1  -> accept
  2. https://huggingface.co/pyannote/segmentation-3.0         -> accept
  3. https://huggingface.co/settings/tokens -> create a read token
  4. put HF_TOKEN=... in .env
"""

import logging
from pathlib import Path

import numpy as np

from .config import env

log = logging.getLogger(__name__)

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
EMBEDDING_MODEL = "pyannote/wespeaker-voxceleb-resnet34-LM"

# per speaker, embed up to this many of their longest segments
MAX_SEGMENTS_PER_SPEAKER = 5
MIN_SEGMENT_S = 1.0


class HFTokenMissing(RuntimeError):
    pass


def _require_token() -> str:
    token = env("HF_TOKEN")
    if not token:
        raise HFTokenMissing(
            "HF_TOKEN is not set. Diarization needs a HuggingFace token:\n"
            "  1. Accept the terms at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  2. Accept the terms at "
            "https://huggingface.co/pyannote/segmentation-3.0\n"
            "  3. Create a read token at "
            "https://huggingface.co/settings/tokens\n"
            "  4. Add HF_TOKEN=... to the .env file in the project root")
    return token


def diarize_track(wav_path: Path) -> list[dict]:
    """[{speaker: 'SPEAKER_00', start, end}] sorted by start."""
    import torch
    from pyannote.audio import Pipeline

    token = _require_token()
    pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
    log.info("Diarizing %s ...", wav_path.name)
    annotation = pipeline(str(wav_path))
    turns = [
        {"speaker": speaker, "start": round(turn.start, 2),
         "end": round(turn.end, 2)}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t["start"])
    log.info("%d turns, %d speakers", len(turns),
             len({t['speaker'] for t in turns}))
    return turns


def speaker_embeddings(wav_path: Path, turns: list[dict]) -> dict[str, np.ndarray]:
    """Mean embedding per diarized speaker, from their longest segments."""
    from pyannote.audio import Inference
    from pyannote.core import Segment

    token = _require_token()
    inference = Inference(EMBEDDING_MODEL, window="whole", use_auth_token=token)

    by_speaker: dict[str, list[dict]] = {}
    for turn in turns:
        if turn["end"] - turn["start"] >= MIN_SEGMENT_S:
            by_speaker.setdefault(turn["speaker"], []).append(turn)

    embeddings: dict[str, np.ndarray] = {}
    for speaker, segs in by_speaker.items():
        segs = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)
        vectors = []
        for seg in segs[:MAX_SEGMENTS_PER_SPEAKER]:
            try:
                vec = inference.crop(str(wav_path),
                                     Segment(seg["start"], seg["end"]))
                vectors.append(np.asarray(vec, dtype=np.float32).reshape(-1))
            except Exception:
                log.exception("embedding failed for %s [%s-%s]",
                              speaker, seg["start"], seg["end"])
        if vectors:
            embeddings[speaker] = np.mean(vectors, axis=0)
    return embeddings


def assign_speakers_to_segments(segments: list[dict],
                                turns: list[dict]) -> None:
    """Label each transcription segment with the diarized speaker that
    overlaps it the most (in place, key 'speaker')."""
    for seg in segments:
        best_speaker, best_overlap = None, 0.0
        for turn in turns:
            overlap = (min(seg["end"], turn["end"])
                       - max(seg["start"], turn["start"]))
            if overlap > best_overlap:
                best_speaker, best_overlap = turn["speaker"], overlap
        seg["speaker"] = best_speaker or "SPEAKER_??"
