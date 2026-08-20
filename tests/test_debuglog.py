import json

import pytest

from meetrec.debuglog import StageError, StageRecorder, load_debug


def test_ok_stage_records_duration(tmp_path):
    rec = StageRecorder(tmp_path)
    with rec.stage("transcribe"):
        pass
    data = load_debug(tmp_path)
    assert data["stages"]["transcribe"]["ok"] is True
    assert "seconds" in data["stages"]["transcribe"]
    assert rec.issue_count == 0


def test_failed_stage_is_recorded_and_swallowed(tmp_path):
    rec = StageRecorder(tmp_path)
    with rec.stage("summary"):
        raise ValueError("backend down")
    data = load_debug(tmp_path)
    assert data["stages"]["summary"]["ok"] is False
    assert "backend down" in data["stages"]["summary"]["error"]
    assert rec.issue_count == 1
    assert "traceback" in data["issues"][0]


def test_fatal_stage_reraises_but_still_records(tmp_path):
    rec = StageRecorder(tmp_path)
    with pytest.raises(StageError):
        with rec.stage("merge", fatal=True):
            raise RuntimeError("boom")
    assert load_debug(tmp_path)["stages"]["merge"]["ok"] is False


def test_strict_mode_reraises_nonfatal(tmp_path):
    rec = StageRecorder(tmp_path, strict=True)
    with pytest.raises(StageError):
        with rec.stage("summary"):
            raise ValueError("x")


def test_note_inside_stage_block_survives_exit(tmp_path):
    rec = StageRecorder(tmp_path)
    with rec.stage("transcribe"):
        rec.note("transcribe", language="pt", segments=7)
    data = load_debug(tmp_path)
    assert data["stages"]["transcribe"]["language"] == "pt"
    assert data["stages"]["transcribe"]["ok"] is True


def test_skip_and_note_are_visible(tmp_path):
    rec = StageRecorder(tmp_path)
    rec.skip("diarization", "track_sys.wav not found")
    with rec.stage("transcribe_mic"):
        pass
    rec.note("transcribe_mic", language="pt", segments=42)
    data = load_debug(tmp_path)
    assert data["stages"]["diarization"]["skipped"] == "track_sys.wav not found"
    assert data["stages"]["transcribe_mic"]["language"] == "pt"


def test_recorder_reload_keeps_stage_history(tmp_path):
    rec1 = StageRecorder(tmp_path)
    with rec1.stage("transcribe"):
        pass
    rec2 = StageRecorder(tmp_path)  # e.g. a reprocess run
    with rec2.stage("summary"):
        pass
    data = load_debug(tmp_path)
    assert set(data["stages"]) == {"transcribe", "summary"}
    assert data["issues"] == []  # issues reset per run


def test_corrupt_debug_file_does_not_crash(tmp_path):
    (tmp_path / "debug.json").write_text("{not json", encoding="utf-8")
    rec = StageRecorder(tmp_path)
    with rec.stage("merge"):
        pass
    assert load_debug(tmp_path)["stages"]["merge"]["ok"] is True


def test_load_debug_missing(tmp_path):
    assert load_debug(tmp_path) is None
    assert json is not None  # silence linters about unused import
