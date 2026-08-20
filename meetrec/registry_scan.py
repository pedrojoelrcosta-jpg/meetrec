"""Read the Windows ConsentStore: which apps are using the microphone.

Mechanism validated on a real machine (2026-08-20): LastUsedTimeStop == 0
with LastUsedTimeStart != 0 means the app is capturing right now.
"""

import winreg
from dataclasses import dataclass

CONSENT_STORE = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager"
    r"\ConsentStore\microphone"
)

_ROOTS = (
    (winreg.HKEY_CURRENT_USER, "HKCU"),
    (winreg.HKEY_LOCAL_MACHINE, "HKLM"),
)


@dataclass(frozen=True)
class MicUsage:
    app_id: str    # subkey name: AppUserModelId (packaged) or '#'-encoded path
    exe_path: str  # readable path (win32) or app_id (packaged)
    hive: str      # HKCU / HKLM
    packaged: bool
    start: int     # raw FILETIME (0 = never used)
    stop: int      # raw FILETIME (0 = capturing now, if start != 0)

    @property
    def is_active(self) -> bool:
        return self.start != 0 and self.stop == 0

    @property
    def exe_name(self) -> str:
        return self.exe_path.rsplit("\\", 1)[-1]


def _read_qword(key, name: str) -> int:
    try:
        value, _ = winreg.QueryValueEx(key, name)
        return int(value)
    except FileNotFoundError:
        return 0


def _scan_branch(hive, hive_name: str, subpath: str, packaged: bool) -> list[MicUsage]:
    results: list[MicUsage] = []
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
                continue  # unreadable subkey (permissions)
            exe = name if packaged else name.replace("#", "\\")
            results.append(MicUsage(name, exe, hive_name, packaged, start, stop))
    return results


def scan_mic_usage() -> list[MicUsage]:
    """Every ConsentStore entry across HKCU/HKLM, packaged and win32."""
    results: list[MicUsage] = []
    for hive, hive_name in _ROOTS:
        results += _scan_branch(hive, hive_name, CONSENT_STORE, packaged=True)
        results += _scan_branch(hive, hive_name, CONSENT_STORE + r"\NonPackaged",
                                packaged=False)
    return results
