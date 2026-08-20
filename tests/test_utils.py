import os
import time

import pytest

from meetrec.cli import _parse_duration
from meetrec.pipeline import cleanup_expired_audio
from meetrec.transcribe import collapse_repetitions


def seg(text, start=0.0):
    return {"start": start, "end": start + 1, "text": text, "words": []}


def test_collapse_keeps_normal_dialogue():
    segments = [seg("Hello"), seg("How are you?"), seg("Fine.")]
    assert collapse_repetitions(segments) == segments


def test_collapse_drops_hallucinated_loop():
    segments = [seg("Ok.")] + [seg("Thank you.") for _ in range(10)]
    result = collapse_repetitions(segments)
    assert len(result) == 3  # "Ok." + 2 kept repeats
    assert result[0]["text"] == "Ok."


def test_collapse_allows_repeats_when_separated():
    segments = [seg("Yes."), seg("No."), seg("Yes."), seg("No.")]
    assert len(collapse_repetitions(segments)) == 4


def test_collapse_ignores_case_and_punctuation():
    segments = [seg("thank you"), seg("Thank you."), seg("THANK YOU!"),
                seg("Thank you")]
    assert len(collapse_repetitions(segments)) == 2


def _make_session(root, name, age_h, with_transcript=True):
    session = root / name
    session.mkdir()
    audio = session / "audio.flac"
    audio.write_bytes(b"fake")
    if with_transcript:
        (session / "transcript.txt").write_text("x", encoding="utf-8")
    old = time.time() - age_h * 3600
    os.utime(audio, (old, old))
    return audio


def test_cleanup_deletes_only_expired_processed_audio(tmp_path):
    cfg = {"output": {"dir": str(tmp_path), "audio_retention_h": 24}}
    expired = _make_session(tmp_path, "old", age_h=48)
    fresh = _make_session(tmp_path, "new", age_h=1)
    unprocessed = _make_session(tmp_path, "raw", age_h=48,
                                with_transcript=False)
    removed = cleanup_expired_audio(cfg)
    assert removed == [expired]
    assert not expired.exists()
    assert fresh.exists()
    assert unprocessed.exists()  # never delete the only copy of a recording


def test_cleanup_disabled_by_default(tmp_path):
    cfg = {"output": {"dir": str(tmp_path), "audio_retention_h": 0}}
    audio = _make_session(tmp_path, "old", age_h=999)
    assert cleanup_expired_audio(cfg) == []
    assert audio.exists()


def test_cleanup_dry_run_deletes_nothing(tmp_path):
    cfg = {"output": {"dir": str(tmp_path), "audio_retention_h": 24}}
    audio = _make_session(tmp_path, "old", age_h=48)
    removed = cleanup_expired_audio(cfg, dry_run=True)
    assert removed == [audio]
    assert audio.exists()


def test_deep_merge_empty_yaml_section_keeps_defaults():
    from meetrec.config import _deep_merge
    base = {"telegram": {"enabled": True, "send_full_transcript": False}}
    # "telegram:" with all keys deleted parses from YAML as None
    merged = _deep_merge(base, {"telegram": None})
    assert merged["telegram"] == base["telegram"]


def test_parse_duration():
    assert _parse_duration("30m") == 1800
    assert _parse_duration("2h") == 7200
    assert _parse_duration("1h30m") == 5400
    for bad in ("", "abc", "10", "5x"):
        with pytest.raises(ValueError):
            _parse_duration(bad)
