from meetrec.merge import drop_mic_echo, merge_tracks, to_plain_text
from meetrec.telegram import MAX_MESSAGE_CHARS, split_message


def seg(start, end, text, speaker=None):
    s = {"start": start, "end": end, "text": text}
    if speaker:
        s["speaker"] = speaker
    return s


def test_merge_is_chronological():
    mic = [seg(10.0, 12.0, "I agree.")]
    sys = [seg(0.0, 8.0, "Shall we start?", "SPEAKER_00"),
           seg(14.0, 20.0, "Great, moving on.", "SPEAKER_00")]
    blocks = merge_tracks(mic, sys)
    assert [b["speaker"] for b in blocks] == ["SPEAKER_00", "ME", "SPEAKER_00"]
    assert blocks[0]["start"] < blocks[1]["start"] < blocks[2]["start"]


def test_merge_self_label_is_configurable():
    blocks = merge_tracks([seg(0.0, 2.0, "Hello")], [], self_label="EU")
    assert blocks[0]["speaker"] == "EU"


def test_merge_joins_adjacent_same_speaker():
    sys = [seg(0.0, 4.0, "First part", "Ana"),
           seg(5.0, 9.0, "second part.", "Ana")]
    blocks = merge_tracks([], sys)
    assert len(blocks) == 1
    assert blocks[0]["text"] == "First part second part."
    assert blocks[0]["end"] == 9.0


def test_merge_respects_gap_limit():
    sys = [seg(0.0, 4.0, "First.", "Ana"), seg(10.0, 12.0, "Later.", "Ana")]
    assert len(merge_tracks([], sys)) == 2


def test_plain_text_format():
    text = to_plain_text([{"speaker": "ME", "start": 65.0, "end": 70.0,
                           "text": "Hello"}])
    assert text == "[00:01:05] ME: Hello\n"


def test_echo_dedup_drops_bleed():
    sys = [seg(10.0, 14.0, "We should ship the release on Friday.", "Ana")]
    mic = [seg(10.3, 14.2, "we should ship the release on friday")]  # bleed
    assert drop_mic_echo(mic, sys) == []


def test_echo_dedup_keeps_real_speech_during_overlap():
    sys = [seg(10.0, 14.0, "We should ship the release on Friday.", "Ana")]
    mic = [seg(10.5, 13.0, "I completely disagree with that plan.")]
    assert drop_mic_echo(mic, sys) == mic


def test_echo_dedup_keeps_same_text_far_apart():
    # repeating someone's words minutes later is quoting, not echo
    sys = [seg(10.0, 14.0, "We should ship the release on Friday.", "Ana")]
    mic = [seg(300.0, 304.0, "We should ship the release on Friday.")]
    assert drop_mic_echo(mic, sys) == mic


def test_echo_dedup_tolerates_muffled_transcription():
    sys = [seg(10.0, 14.0, "We should ship the release on Friday.", "Ana")]
    mic = [seg(10.4, 14.1, "we should ship the release friday")]  # lossy echo
    assert drop_mic_echo(mic, sys) == []


def test_split_message_short_is_untouched():
    assert split_message("hello") == ["hello"]


def test_split_message_respects_limit_and_lines():
    lines = "\n".join(f"line {i:04d} " + "x" * 80 for i in range(200))
    chunks = split_message(lines)
    assert all(len(c) <= MAX_MESSAGE_CHARS for c in chunks)
    assert "".join(chunks) == lines


def test_split_message_pathological_single_line():
    text = "a" * (MAX_MESSAGE_CHARS * 2 + 100)
    chunks = split_message(text)
    assert all(len(c) <= MAX_MESSAGE_CHARS for c in chunks)
    assert "".join(chunks) == text
