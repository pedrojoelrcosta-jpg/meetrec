# -*- coding: utf-8 -*-
"""Standalone diagnostic for the microphone-usage detection mechanism.

Reads the Windows ConsentStore (HKCU and HKLM) and shows which applications
have used — or are currently using — the microphone. Stdlib only.

Usage:
  py diagnose_mic.py            # single snapshot
  py diagnose_mic.py --watch    # continuous polling, prints transitions
"""

import argparse
import sys
import time
import winreg
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

CONSENT_STORE = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager"
    r"\ConsentStore\microphone"
)

ROOTS = (
    (winreg.HKEY_CURRENT_USER, "HKCU"),
    (winreg.HKEY_LOCAL_MACHINE, "HKLM"),
)

# FILETIME: 100 ns intervals since 1601-01-01 UTC
_FILETIME_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)


@dataclass
class MicUsage:
    app_id: str        # subkey name (AppUserModelId or '#'-encoded path)
    exe_path: str      # readable path (NonPackaged) or app_id (packaged)
    hive: str          # HKCU / HKLM
    packaged: bool
    start: int         # raw FILETIME (0 = never)
    stop: int          # raw FILETIME (0 = capturing NOW, if start != 0)

    @property
    def is_active(self) -> bool:
        return self.start != 0 and self.stop == 0

    @property
    def anomaly(self) -> bool:
        # start > stop with stop != 0 should not happen; report if it does
        return self.stop != 0 and self.start > self.stop


def filetime_to_local(ft: int) -> str:
    if ft == 0:
        return "-"
    dt = _FILETIME_EPOCH + timedelta(microseconds=ft // 10)
    return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _read_qword(key, name: str) -> int:
    try:
        value, vtype = winreg.QueryValueEx(key, name)
        return int(value)
    except FileNotFoundError:
        return 0


def _scan_branch(hive, hive_name: str, subpath: str, packaged: bool):
    results = []
    try:
        base = winreg.OpenKey(hive, subpath, 0,
                              winreg.KEY_READ | winreg.KEY_WOW64_64KEY)
    except OSError:
        return results
    with base:
        i = 0
        while True:
            try:
                name = winreg.EnumKey(base, i)
            except OSError:
                break
            i += 1
            if packaged and name == "NonPackaged":
                continue
            try:
                with winreg.OpenKey(base, name, 0,
                                    winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as sub:
                    start = _read_qword(sub, "LastUsedTimeStart")
                    stop = _read_qword(sub, "LastUsedTimeStop")
            except OSError:
                continue  # unreadable subkey (permissions) — skip
            exe = name.replace("#", "\\") if not packaged else name
            results.append(MicUsage(name, exe, hive_name, packaged, start, stop))
    return results


def scan() -> list:
    results = []
    for hive, hive_name in ROOTS:
        results += _scan_branch(hive, hive_name, CONSENT_STORE, packaged=True)
        results += _scan_branch(hive, hive_name, CONSENT_STORE + r"\NonPackaged",
                                packaged=False)
    return results


def snapshot() -> None:
    rows = scan()
    if not rows:
        print("No ConsentStore entries found (unexpected).")
        return
    rows.sort(key=lambda r: (not r.is_active, -r.start))
    active = [r for r in rows if r.is_active]
    print(f"{len(rows)} entries ({len(active)} ACTIVE)\n")
    fmt = "{:<8} {:<10} {:<21} {:<21} {:<8} {}"
    print(fmt.format("Hive", "Type", "Start", "Stop", "State", "Application"))
    print("-" * 110)
    for r in rows:
        state = "ACTIVE" if r.is_active else ("ANOMALY" if r.anomaly else "")
        if r.start == 0 and r.stop == 0:
            state = "(never)"
        print(fmt.format(r.hive, "packaged" if r.packaged else "win32",
                         filetime_to_local(r.start), filetime_to_local(r.stop),
                         state, r.exe_path))
    print()
    if active:
        print("Capturing the microphone RIGHT NOW:")
        for r in active:
            print(f"  -> {r.exe_path}  [{r.hive}]")
    else:
        print("No application is capturing the microphone at the moment.")


def watch(interval: float) -> None:
    print(f"Watching the ConsentStore every {interval:g}s. Ctrl+C to exit.\n"
          "Join and leave a meeting with the microphone on and watch the "
          "transitions:\n")
    previous = set()
    while True:
        now = {(r.hive, r.exe_path) for r in scan() if r.is_active}
        ts = datetime.now().strftime("%H:%M:%S")
        for hive, exe in sorted(now - previous):
            print(f"[{ts}] STARTED capturing: {exe}  [{hive}]")
        for hive, exe in sorted(previous - now):
            print(f"[{ts}] STOPPED capturing: {exe}  [{hive}]")
        previous = now
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--watch", action="store_true",
                        help="continuous polling; prints only transitions")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="polling interval in seconds (default: 2)")
    args = parser.parse_args()
    try:
        watch(args.interval) if args.watch else snapshot()
    except KeyboardInterrupt:
        print("\nDone.")
        sys.exit(0)


if __name__ == "__main__":
    main()
