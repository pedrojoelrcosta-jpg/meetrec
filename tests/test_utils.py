import pytest

from meetrec.cli import _parse_duration
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


def test_parse_duration():
    assert _parse_duration("30m") == 1800
    assert _parse_duration("2h") == 7200
    assert _parse_duration("1h30m") == 5400
    for bad in ("", "abc", "10", "5x"):
        with pytest.raises(ValueError):
            _parse_duration(bad)
