# -*- coding: utf-8 -*-
"""Diagnóstico standalone do mecanismo de deteção de uso do microfone.

Lê o ConsentStore do Windows (HKCU e HKLM) e mostra que aplicações usaram
ou estão a usar o microfone. Só usa a stdlib — não requer instalação.

Uso:
  py diagnose_mic.py            # snapshot único
  py diagnose_mic.py --watch    # polling contínuo, imprime transições
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

# FILETIME: intervalos de 100 ns desde 1601-01-01 UTC
_FILETIME_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)


@dataclass
class MicUsage:
    app_id: str        # nome da subchave (AppUserModelId ou caminho com #)
    exe_path: str      # caminho legível (NonPackaged) ou app_id (empacotada)
    hive: str          # HKCU / HKLM
    packaged: bool
    start: int         # FILETIME cru (0 = nunca)
    stop: int          # FILETIME cru (0 = a capturar AGORA, se start != 0)

    @property
    def is_active(self) -> bool:
        return self.start != 0 and self.stop == 0

    @property
    def anomaly(self) -> bool:
        # start > stop com stop != 0 não devia acontecer; reportar se ocorrer
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
                continue  # subchave ilegível (permissões) — ignorar
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
        print("Nenhuma entrada encontrada no ConsentStore (inesperado).")
        return
    rows.sort(key=lambda r: (not r.is_active, -r.start))
    active = [r for r in rows if r.is_active]
    print(f"{len(rows)} entradas ({len(active)} ATIVAS)\n")
    fmt = "{:<8} {:<10} {:<21} {:<21} {:<8} {}"
    print(fmt.format("Origem", "Tipo", "Início", "Fim", "Estado", "Aplicação"))
    print("-" * 110)
    for r in rows:
        state = "ATIVO" if r.is_active else ("ANOMALIA" if r.anomaly else "")
        if r.start == 0 and r.stop == 0:
            state = "(nunca)"
        print(fmt.format(r.hive, "packaged" if r.packaged else "win32",
                         filetime_to_local(r.start), filetime_to_local(r.stop),
                         state, r.exe_path))
    print()
    if active:
        print("A capturar o microfone AGORA:")
        for r in active:
            print(f"  -> {r.exe_path}  [{r.hive}]")
    else:
        print("Nenhuma aplicação a capturar o microfone neste momento.")


def watch(interval: float) -> None:
    print(f"A vigiar o ConsentStore a cada {interval:g}s. Ctrl+C para sair.\n"
          "Entra e sai de uma reunião com o microfone ligado e observa as "
          "transições:\n")
    previous = set()
    while True:
        now = {(r.hive, r.exe_path) for r in scan() if r.is_active}
        ts = datetime.now().strftime("%H:%M:%S")
        for hive, exe in sorted(now - previous):
            print(f"[{ts}] COMEÇOU a capturar: {exe}  [{hive}]")
        for hive, exe in sorted(previous - now):
            print(f"[{ts}] PAROU de capturar:  {exe}  [{hive}]")
        previous = now
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--watch", action="store_true",
                        help="polling contínuo; imprime só transições")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="intervalo de polling em segundos (default: 2)")
    args = parser.parse_args()
    try:
        watch(args.interval) if args.watch else snapshot()
    except KeyboardInterrupt:
        print("\nTerminado.")
        sys.exit(0)


if __name__ == "__main__":
    main()
