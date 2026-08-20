from meetrec.detector import MeetingDetector, State
from meetrec.registry_scan import MicUsage

CHROME = MicUsage(
    app_id=r"C:#Program Files#Google#Chrome#Application#chrome.exe",
    exe_path=r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    hive="HKCU", packaged=False, start=1, stop=0,
)
CHROME_IDLE = MicUsage(
    app_id=CHROME.app_id, exe_path=CHROME.exe_path,
    hive="HKCU", packaged=False, start=1, stop=2,
)
RECORDER = MicUsage(
    app_id="Microsoft.WindowsSoundRecorder_8wekyb3d8bbwe",
    exe_path="Microsoft.WindowsSoundRecorder_8wekyb3d8bbwe",
    hive="HKCU", packaged=True, start=1, stop=0,
)
TEAMS = MicUsage(
    app_id="MSTeams_8wekyb3d8bbwe",
    exe_path="MSTeams_8wekyb3d8bbwe",
    hive="HKCU", packaged=True, start=1, stop=0,
)


def make_detector(**kwargs):
    events = []
    detector = MeetingDetector(
        scan_fn=lambda: [],
        clock=lambda: 0.0,
        start_debounce_s=15.0,
        stop_debounce_s=30.0,
        on_meeting_started=lambda apps: events.append(("started", apps)),
        on_meeting_ended=lambda duration: events.append(("ended", duration)),
        **kwargs,
    )
    return detector, events


def run_ticks(detector, spec):
    """spec: list of (timestamp, usages)."""
    for now, usages in spec:
        detector.tick(now, usages)


def test_does_not_start_before_debounce():
    detector, events = make_detector()
    run_ticks(detector, [(t, [CHROME]) for t in range(0, 15, 2)])
    assert events == []
    assert detector.state is State.CANDIDATE


def test_starts_after_continuous_capture():
    detector, events = make_detector()
    run_ticks(detector, [(t, [CHROME]) for t in range(0, 17, 2)])
    assert events == [("started", {CHROME.exe_path})]
    assert detector.state is State.ACTIVE


def test_interruption_during_candidate_resets():
    detector, events = make_detector()
    run_ticks(detector, [(t, [CHROME]) for t in range(0, 11, 2)])
    detector.tick(12, [CHROME_IDLE])  # capture stopped at 12s
    assert detector.state is State.IDLE
    # capture resumes: must only start 15s after the NEW beginning
    run_ticks(detector, [(t, [CHROME]) for t in range(14, 28, 2)])
    assert events == []
    detector.tick(30, [CHROME])
    assert events == [("started", {CHROME.exe_path})]


def test_stops_only_after_stop_debounce():
    detector, events = make_detector()
    run_ticks(detector, [(t, [CHROME]) for t in range(0, 101, 2)])
    run_ticks(detector, [(t, []) for t in range(102, 131, 2)])
    assert [e for e in events if e[0] == "ended"] == []
    detector.tick(132, [])
    # duration = cooldown start (first tick without capture, t=102) - capture
    # start (t=0); with 2s polling the ±2s at the edge is expected
    assert ("ended", 102) in events
    assert detector.state is State.IDLE


def test_short_blip_does_not_end_meeting():
    detector, events = make_detector()
    run_ticks(detector, [(t, [CHROME]) for t in range(0, 101, 2)])
    run_ticks(detector, [(t, []) for t in range(102, 107, 2)])  # ~5s blip
    detector.tick(108, [CHROME])
    assert detector.state is State.ACTIVE
    assert [e for e in events if e[0] == "ended"] == []


def test_ignored_app_never_triggers():
    detector, events = make_detector(
        ignore_apps=["Microsoft.WindowsSoundRecorder_8wekyb3d8bbwe"])
    run_ticks(detector, [(t, [RECORDER]) for t in range(0, 60, 2)])
    assert events == []
    assert detector.state is State.IDLE


def test_ignored_exe_name_case_insensitive():
    detector, events = make_detector(ignore_apps=["CHROME.EXE"])
    run_ticks(detector, [(t, [CHROME]) for t in range(0, 60, 2)])
    assert events == []
    assert detector.state is State.IDLE


def test_ignored_plus_real_app_triggers():
    detector, events = make_detector(
        ignore_apps=["Microsoft.WindowsSoundRecorder_8wekyb3d8bbwe"])
    run_ticks(detector, [(t, [RECORDER, TEAMS]) for t in range(0, 17, 2)])
    assert events == [("started", {TEAMS.exe_path})]


def test_events_carry_all_apps_seen():
    detector, events = make_detector()
    detector.tick(0, [CHROME])
    run_ticks(detector, [(t, [CHROME, TEAMS]) for t in range(2, 17, 2)])
    assert events == [("started", {CHROME.exe_path, TEAMS.exe_path})]


def test_inactive_entries_do_not_trigger():
    detector, events = make_detector()
    run_ticks(detector, [(t, [CHROME_IDLE]) for t in range(0, 60, 2)])
    assert events == []
    assert detector.state is State.IDLE
