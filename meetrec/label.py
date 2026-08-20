"""Interactive labeling of unknown speakers (`meetrec label <session>`).

For each unknown speaker, plays up to 3 audio excerpts from track_sys.wav,
asks for a name, and stores the mean embedding in the voiceprint DB so the
speaker is recognized in future meetings.
"""

import json
import wave
import winsound
from pathlib import Path

import numpy as np

from .config import data_dir
from .voiceprints import VoiceprintDB

EXCERPTS_PER_SPEAKER = 3
EXCERPT_MAX_S = 8.0


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_excerpt(wav_path: Path, start: float, end: float,
                     out_path: Path) -> None:
    with wave.open(str(wav_path), "rb") as wav:
        rate = wav.getframerate()
        wav.setpos(int(start * rate))
        frames = wav.readframes(int(min(end - start, EXCERPT_MAX_S) * rate))
        with wave.open(str(out_path), "wb") as out:
            out.setnchannels(wav.getnchannels())
            out.setsampwidth(wav.getsampwidth())
            out.setframerate(rate)
            out.writeframes(frames)


def _extract_excerpt_flac(flac_path: Path, start: float, end: float,
                          out_path: Path) -> None:
    """Fallback when track_sys.wav was cleaned up: the system track lives on
    the right channel of audio.flac."""
    import numpy as np
    import soundfile as sf

    with sf.SoundFile(str(flac_path)) as flac:
        rate = flac.samplerate
        flac.seek(int(start * rate))
        frames = flac.read(int(min(end - start, EXCERPT_MAX_S) * rate),
                           dtype="int16", always_2d=True)
    channel = frames[:, 1] if frames.shape[1] > 1 else frames[:, 0]
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(np.ascontiguousarray(channel).tobytes())


def label_session(cfg: dict, session_dir: Path) -> bool:
    """Returns True if any speaker was labeled."""
    emb_path = session_dir / "embeddings.npz"
    diar_path = session_dir / "diarization.json"
    if not emb_path.exists() or not diar_path.exists():
        print(f"No diarization data in {session_dir} — nothing to label.")
        return False

    embeddings = dict(np.load(emb_path))
    turns = _read_json(diar_path)
    speaker_map_path = session_dir / "speaker_map.json"
    speaker_map: dict = (_read_json(speaker_map_path)
                         if speaker_map_path.exists() else {})
    unknown = [s for s in embeddings if s not in speaker_map]
    if not unknown:
        print("Every speaker in this session is already identified.")
        return False

    sys_wav = session_dir / "track_sys.wav"
    flac = session_dir / "audio.flac"
    excerpt_dir = session_dir / "excerpts"
    excerpt_dir.mkdir(exist_ok=True)
    db = VoiceprintDB(data_dir() / "voiceprints.db")
    labeled = False
    try:
        for speaker in unknown:
            segs = sorted(
                (t for t in turns if t["speaker"] == speaker),
                key=lambda t: t["end"] - t["start"], reverse=True,
            )[:EXCERPTS_PER_SPEAKER]
            print(f"\n=== {speaker} ({len(segs)} excerpts) ===")
            for i, seg in enumerate(segs, 1):
                out = excerpt_dir / f"{speaker}_{i}.wav"
                if sys_wav.exists():
                    _extract_excerpt(sys_wav, seg["start"], seg["end"], out)
                elif flac.exists():
                    _extract_excerpt_flac(flac, seg["start"], seg["end"], out)
                else:
                    print("  (no audio file found — cannot play excerpts)")
                    break
                print(f"  Playing excerpt {i} "
                      f"[{seg['start']:.0f}s–{seg['end']:.0f}s] ...")
                try:
                    winsound.PlaySound(str(out), winsound.SND_FILENAME)
                except RuntimeError:
                    print(f"  (playback failed — listen to {out})")
            name = input(f"Name for {speaker} (empty = skip): ").strip()
            if name:
                db.add(name, embeddings[speaker])
                speaker_map[speaker] = name
                labeled = True
                print(f"  {speaker} -> {name} saved to the voiceprint DB")
    finally:
        db.close()

    if labeled:
        speaker_map_path.write_text(
            json.dumps(speaker_map, ensure_ascii=False, indent=2),
            encoding="utf-8")
        from .pipeline import regenerate_transcripts
        regenerate_transcripts(session_dir)
        print("\nTranscripts regenerated with the new names.")
    return labeled
